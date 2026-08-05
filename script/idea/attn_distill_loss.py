"""Attention distillation with chunked cross-layer head matching.

Teacher layers are split into Ls contiguous chunks (equal depths -> identity);
each student head is Hungarian-matched to one (teacher_layer, teacher_head)
inside its chunk by min mean-KL, then trained to imitate it. Load both models
with attn_implementation="eager" and fp32/bf16 (fp16 attn probs underflow).

    loss = attn_distill_loss(teacher, student, rollouts)  # rollouts: (B,L) ids
    loss.backward()
"""
import torch
from scipy.optimize import linear_sum_assignment

EPS = 1e-9

# Comment 1. the n_samples appear redundant, we probably want a better matching result, using all the attn we obtain via eager mode anyway
#            also, just to confirm, we are obtaining the avg. teacher head - student head matching result right? 
#            how come the 'sp' is not sliced as the tp -> t operation? we are doing it in batch? 
# Comment 2. we need to ensure layer matching (or at least layer chunk mathching when student & teacher model got different number of layers)
#            otherwise it makes no sense to match heads between different layers
# Comment 3. In practice, it is important to sweep on the choice of layers / chunks for the matching
# Comment 4. masking should be tested, it might be that the attention to the query is critical enough, or that we need the full attention pattern 
#            to be matched here
# Comment 5. during training, we would periodically update the matching table to avoid per-step head matching here


@torch.no_grad()
def chunk_head_map(t_attn, chunk, s_attn):
    """Injectively match each student head of one layer to a (teacher_layer,
    teacher_head) pair drawn from anywhere in `chunk`, by min KL averaged
    over batch rows and query positions. Returns list of pairs, length Hs."""
    Ht = t_attn[chunk[0]].shape[1]
    tp = torch.cat([t_attn[l] for l in chunk], 1).float().clamp_min(EPS)
    sp = s_attn.float().clamp_min(EPS)
    cost = torch.stack([                                    # (len(chunk)*Ht, Hs)
        (tp[:, i:i + 1] * (tp[:, i:i + 1].log() - sp.log())).sum(-1).mean(dim=(0, 2))
        for i in range(tp.shape[1])])
    assert cost.shape[0] >= sp.shape[1], "teacher pool smaller than student heads"
    rows, cols = linear_sum_assignment(cost.cpu().numpy())
    pairs = [None] * sp.shape[1]
    for i, j in zip(rows, cols):
        pairs[j] = (chunk[i // Ht], i % Ht)
    return pairs


def attn_distill_loss(teacher, student, rollouts, query_mask=None,
                      attention_mask=None):
    """KL(teacher-attn || student-attn) under chunked head matching, averaged
    over layers, heads, and (optionally query_mask-restricted) positions."""
    with torch.no_grad():
        t_attn = teacher(rollouts, attention_mask=attention_mask,
                         output_attentions=True).attentions
    if torch.isnan(t_attn[-1]).any():
        raise ValueError("teacher attentions contain NaN — load the model "
                         "with torch_dtype=torch.float32 (or bfloat16)")
    s_attn = student(rollouts, attention_mask=attention_mask,
                     output_attentions=True).attentions
    Ls, Lt = len(s_attn), len(t_attn)
    b = [round(i * Lt / Ls) for i in range(Ls + 1)]         # chunk boundaries
    loss = rollouts.new_zeros((), dtype=torch.float32)
    for sl in range(Ls):
        sp = s_attn[sl]                                     # (B,Hs,L,L)
        pairs = chunk_head_map(t_attn, list(range(b[sl], b[sl + 1])), sp)
        tp = torch.stack([t_attn[l][:, h] for l, h in pairs], 1)
        tp = tp.float().clamp_min(EPS)
        sp = sp.float().clamp_min(EPS)
        kl = (tp * (tp.log() - sp.log())).sum(-1)[:, :, 1:]  # (B,Hs,L-1)
        if query_mask is not None:
            kl = kl[:, :, query_mask[1:]]
        loss = loss + kl.mean()
    return loss / Ls
