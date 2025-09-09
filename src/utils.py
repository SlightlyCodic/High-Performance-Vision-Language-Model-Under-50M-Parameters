import random
import torch

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum=0.0; self.cnt=0
    def update(self, val, n=1): self.sum += float(val)*n; self.cnt += n
    @property
    def avg(self): return self.sum/max(self.cnt,1)

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_trainable_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
