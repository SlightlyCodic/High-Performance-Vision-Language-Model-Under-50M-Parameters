import math
import torch
import torch.nn.functional as F

def clip_contrastive_loss_multi(z_img, z_txt, K, logit_scale, soft_i2t=True):
    """
    z_img: [B,d], z_txt: [B*K,d]
    I->T: soft multi-positive (uniform over K); T->I: CE with repeated labels.
    """
    B = z_img.size(0)
    s = logit_scale.exp().clamp(1e-3, 100.0)
    logits = s * (z_img @ z_txt.t())  # [B, B*K]

    # I->T
    if soft_i2t:
        with torch.no_grad():
            targets = torch.zeros_like(logits)
            for i in range(B):
                cols = slice(i*K, i*K+K)
                targets[i, cols] = 1.0 / K
        loss_i = F.kl_div(F.log_softmax(logits, dim=1), targets, reduction="batchmean")
    else:
        labels_i = torch.arange(B, device=logits.device) * K
        loss_i = F.cross_entropy(logits, labels_i)

    # T->I
    logits_t = logits.t().contiguous()  # [B*K, B]
    labels_t = torch.arange(B, device=logits.device).repeat_interleave(K)
    loss_t = F.cross_entropy(logits_t, labels_t)

    return 0.5*(loss_i+loss_t), logits

def distill_losses_dimaware(z_i_s, z_t_s, z_i_t, z_t_t, tau_t: float = 1.0, tau_s: float = 1.0):
    # feature L2 (normalised targets)
    z_i_t_ = F.normalize(z_i_t, dim=-1); z_t_t_ = F.normalize(z_t_t, dim=-1)
    L_feat = 0.5*(F.mse_loss(z_i_s, z_i_t_) + F.mse_loss(z_t_s, z_t_t_))
    # row/col KL on similarities
    S_s = (z_i_s @ z_t_s.t()) / max(tau_s,1e-6)
    S_t = (z_i_t @ z_t_t.t()) / max(tau_t,1e-6)
    P_t_row = F.softmax(S_t, dim=1).detach(); P_s_row = F.log_softmax(S_s, dim=1)
    P_t_col = F.softmax(S_t.t(), dim=1).detach(); P_s_col = F.log_softmax(S_s.t(), dim=1)
    L_row = F.kl_div(P_s_row, P_t_row, reduction="batchmean")
    L_col = F.kl_div(P_s_col, P_t_col, reduction="batchmean")
    L_logit = 0.5*(L_row+L_col)
    L_aff = L_logit
    return L_feat, L_logit, L_aff
