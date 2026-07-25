import json, statistics as st
P = "logs/ablate_game24_velreward/Qwen__Qwen3-0.6B/pt-velocity_adv-token_seed-0"

rows = [json.loads(l) for l in open(f"{P}/reward_stats.jsonl")]
print("reward_stats: rows=", len(rows))
m_n  = sum(r["marker"]["n"]    for r in rows)
nm_n = sum(r["no_marker"]["n"] for r in rows)
print(f"  marker rollouts total={m_n}  no_marker rollouts total={nm_n}")
m_RT  = [r["marker"]["RT_mean"]    for r in rows if r["marker"]["n"]>0]
nm_RT = [r["no_marker"]["RT_mean"] for r in rows if r["no_marker"]["n"]>0]
m_len = [r["marker"]["len_mean"]   for r in rows if r["marker"]["n"]>0]
nm_len= [r["no_marker"]["len_mean"]for r in rows if r["no_marker"]["n"]>0]
print(f"  marker    : RT mean={st.mean(m_RT):+.2f}  len mean={st.mean(m_len):.0f}  RT>100 frac={sum(x>100 for x in m_RT)/len(m_RT):.3f}")
print(f"  no_marker : RT mean={st.mean(nm_RT):+.2f}  len mean={st.mean(nm_len):.0f}  RT<0 frac={sum(x<0 for x in nm_RT)/len(nm_RT):.3f}")

recs = [json.loads(l) for l in open(f"{P}/velocity_log.jsonl")]
print(f"velocity_log: rollouts={len(recs)}")
n_corr = sum(r["correct"] for r in recs)
print(f"  correct: {n_corr}/{len(recs)} ({100*n_corr/len(recs):.2f}%)")
clens = [r["comp_len"] for r in recs]; olens = [r["o_len"] for r in recs]
print(f"  comp_len==1024 frac: {sum(x==1024 for x in clens)/len(clens):.3f}")
print(f"  o_len   ==1024 frac: {sum(x==1024 for x in olens)/len(olens):.3f}")
print(f"  has_marker (comp_len>o_len) frac: {sum(r['comp_len']>r['o_len'] for r in recs)/len(recs):.3f}")
trunc = [r for r in recs if r["o_len"] >= 1024]
ok    = [r for r in recs if r["o_len"] <  1024]
def s(xs): return f"n={len(xs)} mean={st.mean(xs):+.2f} median={st.median(xs):+.2f}" if xs else "n=0"
print(f"  truncated (o_len==1024)    R_T: {s([r['R_T'] for r in trunc])}")
print(f"  non-truncated (o_len<1024) R_T: {s([r['R_T'] for r in ok])}")
print(f"  truncated  R_T>0 frac: {sum(r['R_T']>0 for r in trunc)/max(1,len(trunc)):.3f}")
print(f"  truncated  R_per_token: {s([r['R_per_token'] for r in trunc])}")
print(f"  non-trunc  R_per_token: {s([r['R_per_token'] for r in ok])}")

# length-quartile within non-truncated: does shorter -> higher R/token?
ok_sorted = sorted(ok, key=lambda r: r["o_len"])
import numpy as np
qs = np.array_split(ok_sorted, 4)
print("  among o_len<1024, length quartiles:")
for i,q in enumerate(qs):
    if not len(q): continue
    ls = [r["o_len"] for r in q]; rts=[r["R_T"] for r in q]; rpts=[r["R_per_token"] for r in q]
    print(f"    Q{i+1} o_len[{min(ls)},{max(ls)}] n={len(q)}: R_T={st.mean(rts):+.2f}  R/tok={st.mean(rpts):+.4f}")
