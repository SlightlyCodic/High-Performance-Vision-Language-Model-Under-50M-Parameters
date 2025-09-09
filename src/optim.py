import math
import torch

def cosine_decay(step, total_steps, base_lr, min_lr=1e-6):
    if total_steps <= 0: return base_lr
    t = min(step/total_steps, 1.0)
    return float(min_lr + 0.5*(base_lr - min_lr)*(1 + math.cos(math.pi*t)))

def build_optimizers(student, lr_enc, lr_head, wd):
    heads, encs = [], []
    for n,p in student.named_parameters():
        if not p.requires_grad: continue
        if n.startswith(("v_proj","t_proj","logit_scale")): heads.append(p)
        else: encs.append(p)
    opt = torch.optim.AdamW(
        [{"params": encs, "lr": lr_enc, "weight_decay": wd},
         {"params": heads,"lr": lr_head,"weight_decay": wd}]
    )
    for pg in opt.param_groups: pg.setdefault("initial_lr", pg["lr"])
    return opt
