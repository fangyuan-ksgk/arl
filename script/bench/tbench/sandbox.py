"""proot userspace sandbox for Terminal-Bench tasks — no Docker, no privileges.

Port of the SkyRL terminal env sandbox (see arl/skyrl_terminal/HANDOFF.md) for
the TRL GRPOTrainer pipeline.

The core trick
--------------
This box blocks user namespaces (no CAP_SYS_ADMIN) so Docker / rootless /
bubblewrap all fail. `proot` (userspace ptrace chroot, zero privilege) binds a
per-rollout temp dir to guest `/app`:

    proot -b $SANDBOX:/app -w /app bash /host/path/script.sh

The agent script's writes and the task's pytest verifier's absolute `/app/...`
checks resolve to the same isolated dir. Tests stay on the host (never copied
into the sandbox); the verifier runs the host pytest (from ~/tbench-venv)
under the same proot binding.

MANDATORY reaping fix (do not remove)
-------------------------------------
Agent-generated bash that loops/forks/backgrounds must be reaped, or orphans
accumulate across rollouts: a previous machine died at 2,965 live bash procs
holding ~50 GB RSS. Every subprocess here is launched with
`start_new_session=True` (child pid == process-group id) and the WHOLE group
is `os.killpg(SIGKILL)`-ed on timeout AND on normal exit (`_run_reaped`).
Known tradeoff: under proot, a script that backgrounds a process waits out its
exec timeout (proot ptrace-traces every descendant, so it only exits when they
do) and is then reaped — correct and safe.

Graceful degradation
--------------------
If `proot` or the verifier venv is missing, `check_available()` raises
`SandboxUnavailable` with the exact setup commands. Setup prerequisites:

    sudo apt-get install -y proot
    python3 -m venv ~/tbench-venv && ~/tbench-venv/bin/pip install pytest pyyaml
    git clone --depth 1 https://github.com/laude-institute/terminal-bench \
        ~/terminal-bench   # tasks live in ~/terminal-bench/original-tasks
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

PROOT = os.environ.get("TBENCH_PROOT", shutil.which("proot") or "/usr/bin/proot")
VERIFIER_PYTHON = os.environ.get(
    "TBENCH_VERIFIER_PYTHON", os.path.expanduser("~/tbench-venv/bin/python")
)
DEFAULT_TASKS_DIR = os.environ.get(
    "TBENCH_TASKS_DIR", os.path.expanduser("~/terminal-bench/original-tasks")
)
# Where per-rollout sandbox temp dirs are created.
SBX_ROOT = os.environ.get("TBENCH_SBX_ROOT", tempfile.gettempdir())

SETUP_INSTRUCTIONS = """\
Terminal-Bench sandbox prerequisites are missing. To set up:
  1. proot (userspace chroot; no privileges needed to RUN, apt needs sudo):
       sudo apt-get update && sudo apt-get install -y proot
  2. verifier venv (host-side pytest that runs each task's tests under proot):
       python3 -m venv ~/tbench-venv
       ~/tbench-venv/bin/pip install pytest pyyaml
  3. tasks (curated set in arl/skyrl_terminal/local_tasks.json points here):
       git clone --depth 1 https://github.com/laude-institute/terminal-bench ~/terminal-bench
Override paths with TBENCH_PROOT / TBENCH_VERIFIER_PYTHON / TBENCH_TASKS_DIR."""


class SandboxUnavailable(RuntimeError):
    """Raised when proot / the verifier venv is missing. Message = setup steps."""


def missing_prerequisites(task_path: Optional[str] = None) -> list[str]:
    """Return a list of human-readable missing prerequisites (empty = all good)."""
    missing = []
    if not (PROOT and os.path.exists(PROOT)):
        missing.append(f"proot binary not found (looked at {PROOT!r})")
    if not os.path.exists(VERIFIER_PYTHON):
        missing.append(f"verifier python not found at {VERIFIER_PYTHON!r}")
    if task_path is not None and not (Path(task_path) / "task.yaml").exists():
        missing.append(f"task dir {task_path!r} missing (no task.yaml)")
    return missing


def check_available(task_path: Optional[str] = None) -> None:
    missing = missing_prerequisites(task_path)
    if missing:
        raise SandboxUnavailable(
            "cannot run the terminal sandbox:\n  - "
            + "\n  - ".join(missing)
            + "\n\n"
            + SETUP_INSTRUCTIONS
        )


# ---------------------------------------------------------------------------
# Reaped subprocess execution (THE fix — see module docstring)
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    secs: float


def _killpg(proc: subprocess.Popen) -> None:
    """SIGKILL the whole process group (pgid == child pid via start_new_session)."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _run_reaped(
    cmd: Sequence[str],
    timeout: float,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
) -> RunResult:
    """subprocess.run(timeout=) replacement that reaps ALL descendants.

    start_new_session=True puts the child in its own process group; on timeout
    AND on normal exit we os.killpg(SIGKILL) the group, so backgrounded /
    forked grandchildren can never orphan and accumulate across rollouts.
    """
    t0 = time.time()
    proc = subprocess.Popen(
        list(cmd),
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    out, err = "", ""
    try:
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _killpg(proc)
            try:
                out, err = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:  # pathological: pipe still held
                proc.kill()
                out, err = proc.communicate()
    finally:
        # Reap on NORMAL exit too: bash may have backgrounded children that
        # survived their parent (they escape proot only if proot itself died).
        _killpg(proc)
    return RunResult(
        returncode=proc.returncode if proc.returncode is not None else -9,
        stdout=out or "",
        stderr=err or "",
        timed_out=timed_out,
        secs=round(time.time() - t0, 2),
    )


# ---------------------------------------------------------------------------
# proot sandbox: run agent script, then run the task's own pytest verifier
# ---------------------------------------------------------------------------

def _proot_cmd(app_dir: str, inner: Sequence[str]) -> list[str]:
    """proot binds the sandbox dir to guest /app and cds there.

    Everything else on the host filesystem stays visible (that is how the
    host-side script file, verifier python and the task's tests/ resolve)."""
    return [PROOT, "-b", f"{app_dir}:/app", "-w", "/app", *inner]


def run_script(script_text: str, app_dir: str, timeout: float = 45.0) -> RunResult:
    """Execute one bash script inside the proot sandbox (cwd = /app).

    The script file lives on the HOST side (a sibling of app_dir, invoked by
    absolute path) so it never pollutes /app — several tasks' tests inspect
    the /app directory contents.
    """
    check_available()
    script_path = Path(app_dir).parent / "agent_script.sh"
    script_path.write_text(script_text)
    return _run_reaped(_proot_cmd(app_dir, ["bash", str(script_path)]), timeout=timeout)


# ---------------------------------------------------------------------------
# Task-file staging: replicate the Dockerfile's COPY/ADD of task inputs to /app
# ---------------------------------------------------------------------------

_COPY_RE = re.compile(r"^\s*(COPY|ADD)\s+(.*)$", re.IGNORECASE)
_WORKDIR_RE = re.compile(r"^\s*WORKDIR\s+(\S+)", re.IGNORECASE)


def stage_task_files(task_path: str, app_dir: str) -> None:
    """Materialize the task's input files into the sandbox /app dir.

    Terminal-Bench tasks ship their input data (logs, csv, code, ...) in the
    task dir and `COPY` it to /app in the Dockerfile. We have no Docker, so we
    parse the Dockerfile's COPY/ADD lines (relative to the task dir) and copy
    the same files into the sandbox before the agent script runs. Docker
    semantics honored: a trailing-slash / multi-src dst is a directory; a
    directory src has its CONTENTS copied into dst. Destinations outside /app
    are ignored (curated tasks never need them)."""
    dockerfile = Path(task_path) / "Dockerfile"
    if not dockerfile.exists():
        return
    task_dir = Path(task_path)
    app = Path(app_dir)
    workdir = "/"
    # join Dockerfile line continuations
    text = re.sub(r"\\\s*\n", " ", dockerfile.read_text())
    for line in text.splitlines():
        m = _WORKDIR_RE.match(line)
        if m:
            workdir = m.group(1)
            continue
        m = _COPY_RE.match(line)
        if not m:
            continue
        tokens = [t for t in m.group(2).split() if t]
        tokens = [t for t in tokens if not t.startswith("--")]  # --chown/--from etc.
        if len(tokens) < 2:
            continue
        srcs, dst = tokens[:-1], tokens[-1]
        dst_is_dir = dst.endswith("/") or len(srcs) > 1
        if not dst.startswith("/"):
            dst = os.path.normpath(os.path.join(workdir, dst))
        dst = os.path.normpath(dst)
        if dst != "/app" and not dst.startswith("/app/"):
            continue  # only /app is sandboxed
        rel = os.path.relpath(dst, "/app")
        host_dst = app if rel == "." else app / rel
        for src_pat in srcs:
            pat = (src_pat[2:] if src_pat.startswith("./") else src_pat).rstrip("/")
            for src in sorted(task_dir.glob(pat)):
                if src.is_dir():
                    # docker copies the CONTENTS of a dir src into dst
                    shutil.copytree(src, host_dst, dirs_exist_ok=True)
                elif host_dst == app or dst_is_dir or host_dst.is_dir():
                    host_dst.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, host_dst / src.name)
                else:
                    host_dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, host_dst)


@dataclass
class VerifyResult:
    reward: float          # fraction of the task's pytest tests that passed (0..1)
    n_passed: int
    n_total: int
    all_passed: bool
    detail: str            # short human-readable diagnostic

    @property
    def error(self) -> Optional[str]:
        return None if self.n_total > 0 else self.detail


_PYTEST_COUNT_RE = re.compile(r"(\d+) (passed|failed|errors?|xpassed|xfailed)")


def _parse_pytest_counts(output: str) -> dict:
    """Parse the pytest summary line, e.g. '1 failed, 3 passed in 0.15s'."""
    counts: dict = {}
    for line in reversed(output.splitlines()):
        found = _PYTEST_COUNT_RE.findall(line)
        if found:
            for n, kind in found:
                counts[kind.rstrip("s")] = int(n)
            break
    return counts


def verify(app_dir: str, task_path: str, timeout: float = 60.0) -> VerifyResult:
    """Run the task's own pytest verifier against the sandbox, on the host.

    The tests stay at <task>/tests/test_outputs.py on the host; their absolute
    /app/... assertions resolve inside proot to app_dir. Reward = fraction of
    tests passed.
    """
    check_available(task_path)
    test_file = Path(task_path) / "tests" / "test_outputs.py"
    if not test_file.exists():
        return VerifyResult(0.0, 0, 0, False, f"no verifier at {test_file}")
    cmd = _proot_cmd(
        app_dir,
        [
            VERIFIER_PYTHON, "-m", "pytest", str(test_file),
            "-q", "--tb=no", "-p", "no:cacheprovider",
        ],
    )
    res = _run_reaped(cmd, timeout=timeout, env={**os.environ, "TEST_DIR": str(test_file.parent)})
    if res.timed_out:
        return VerifyResult(0.0, 0, 0, False, f"verifier timed out after {timeout}s")
    counts = _parse_pytest_counts(res.stdout + "\n" + res.stderr)
    n_passed = counts.get("passed", 0)
    n_total = n_passed + counts.get("failed", 0) + counts.get("error", 0)
    if n_total == 0:
        tail = (res.stdout + res.stderr).strip().splitlines()[-3:]
        return VerifyResult(0.0, 0, 0, False, "no tests ran: " + " | ".join(tail))
    reward = n_passed / n_total
    return VerifyResult(reward, n_passed, n_total, n_passed == n_total,
                        f"{n_passed}/{n_total} passed in {res.secs}s")


def score_completion(
    script_text: str,
    task_path: str,
    exec_timeout: float = 45.0,
    verify_timeout: float = 60.0,
) -> VerifyResult:
    """One full rollout scoring: fresh sandbox -> run agent bash -> verify.

    Always verifies, even if the agent script timed out (it may have done
    enough before being reaped — partial credit is the whole point of the
    fraction-passed reward)."""
    check_available(task_path)
    sbx = tempfile.mkdtemp(prefix="tbench_sbx_", dir=SBX_ROOT)
    try:
        app_dir = os.path.join(sbx, "app")
        os.makedirs(app_dir, exist_ok=True)
        stage_task_files(task_path, app_dir)  # Dockerfile COPY inputs -> /app
        run_script(script_text, app_dir, timeout=exec_timeout)
        return verify(app_dir, task_path, timeout=verify_timeout)
    finally:
        shutil.rmtree(sbx, ignore_errors=True)


def validate_task(
    task_path: str, solve_timeout: float = 120.0, verify_timeout: float = 120.0
) -> VerifyResult:
    """Run the task's OWN reference solution.sh through sandbox + verifier.

    Used for curation and smoke testing: a kept task's canonical solution must
    score 1.0, guaranteeing the RL reward signal is correct. Supports both
    solution.sh and the step-list solution.yaml format."""
    solution = Path(task_path) / "solution.sh"
    if solution.exists():
        script = solution.read_text()
    elif (Path(task_path) / "solution.yaml").exists():
        import yaml

        steps = yaml.safe_load((Path(task_path) / "solution.yaml").read_text())
        script = "\n".join(str(s["command"]) for s in steps if isinstance(s, dict))
    else:
        return VerifyResult(0.0, 0, 0, False, f"no solution.sh/.yaml in {task_path}")
    return score_completion(
        script, task_path,
        exec_timeout=solve_timeout, verify_timeout=verify_timeout,
    )
