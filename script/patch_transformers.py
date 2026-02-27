"""
Patch transformers 5.2.0 for compatibility with huggingface-hub>=1.0.

Run after `pip install -r requirements.txt`:
    python script/patch_transformers.py

Fixes:
  1. dependency_versions_table.py: remove <1.0 upper bound on huggingface-hub
  2. utils/hub.py: catch EntryNotFoundError in list_repo_templates() so models
     without additional_chat_templates/ (e.g. Qwen3-1.7B) don't crash
"""

import importlib
import re
import sys
from pathlib import Path


def get_transformers_dir() -> Path:
    import transformers
    return Path(transformers.__file__).parent


def patch_dependency_versions_table(tf_dir: Path) -> bool:
    path = tf_dir / "dependency_versions_table.py"
    text = path.read_text()
    old = '"huggingface-hub>=0.34.0,<1.0"'
    new = '"huggingface-hub>=0.34.0"'
    if old not in text:
        print(f"  [skip] upper bound already removed or pattern not found")
        return False
    path.write_text(text.replace(old, new))
    print(f"  [ok]   removed <1.0 upper bound on huggingface-hub")
    return True


def patch_list_repo_templates(tf_dir: Path) -> bool:
    path = tf_dir / "utils" / "hub.py"
    text = path.read_text()

    # Check if already patched
    if "except EntryNotFoundError:" in text:
        # Make sure it's the one after the offline/connection block
        # (there may be other uses elsewhere in the file)
        block = (
            "except (HTTPError, OfflineModeIsEnabled, requests.exceptions.ConnectionError):\n"
            "            pass  # offline mode, internet down, etc. => try local files\n"
            "        except EntryNotFoundError:\n"
            "            return []"
        )
        if block in text:
            print(f"  [skip] list_repo_templates already patched")
            return False

    old = (
        "except (HTTPError, OfflineModeIsEnabled, requests.exceptions.ConnectionError):\n"
        "            pass  # offline mode, internet down, etc. => try local files"
    )
    new = (
        "except (HTTPError, OfflineModeIsEnabled, requests.exceptions.ConnectionError):\n"
        "            pass  # offline mode, internet down, etc. => try local files\n"
        "        except EntryNotFoundError:\n"
        "            return []  # repo has no additional_chat_templates directory"
    )
    if old not in text:
        print(f"  [skip] list_repo_templates pattern not found (different version?)")
        return False
    path.write_text(text.replace(old, new))
    print(f"  [ok]   added EntryNotFoundError handler in list_repo_templates()")
    return True


def main():
    tf_dir = get_transformers_dir()
    import transformers
    version = transformers.__version__
    print(f"Patching transformers {version} at {tf_dir}")

    n = 0
    n += patch_dependency_versions_table(tf_dir)
    n += patch_list_repo_templates(tf_dir)

    if n == 0:
        print("Nothing to patch — already up to date.")
    else:
        print(f"Applied {n} patch(es). Done.")


if __name__ == "__main__":
    main()
