import os, argparse, math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import open_clip

from .utils import set_seed, AverageMeter, count_trainable_params
from .models import build_student
from .data import CocoCaptionTrainMulti, collate_train, build_image_transform
from .optim import build_optimizers, cosine_decay
from .losses import clip_contrastive_loss_multi, distill_losses_dimaware

def load_teacher(arch, pretrained, device):
    teacher, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device=device)
    for p in teacher.parameters(): p.requires_grad=False
    teacher.eval()
    return teacher

def parse_args():
    ap = argparse.ArgumentParser()
    # Student/recipe
    ap.add_argument("--recipe", default="tinyvit_11m_minilm_l6",
                    choices=["vit_s16_minilm_l6","tinyvit_11m_minilm_l6","effnet_b0_tinybert_4l_312"])
    ap.add_argument("--embed_dim", type=int, default=512)

    # Data
    ap.add_argument("--image_dir_train", type=str, required=True)
    ap.add_argument("--ann_train", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="outputs_tinyclip_convffn_mc")
    ap.add_argument("--save_every", type=int, default=1)

    # Optional quick-val (not used by script itself)
    ap.add_argument("--image_dir_val", type=str, default="")
    ap.add_argument("--ann_val", type=str, default="")

    # Training
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--mixed_precision", action="store_true", default=True)
    ap.add_argument("--log_every", type=int, default=10)

    # Optim
    ap.add_argument("--lr_enc", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)

    # Text/batching
    ap.add_argument("--max_tokens", type=int, default=32)
    ap.add_argument("--caps_per_image", type=int, default=2)
    ap.add_argument("--soft_i2t", action="store_true", default=True)

    # Distill
    ap.add_argument("--teacher_arch", default="ViT-B-32")
    ap.add_argument("--teacher_ckpt", default="laion2b_s34b_b79k")
    ap.add_argument("--alpha_feat_warmup", type=float, default=0.50)
    ap.add_argument("--alpha_feat_base",   type=float, default=0.30)
    ap.add_argument("--beta_logit", type=float, default=0.50)
    ap.add_argument("--gamma_aff",  type=float, default=1.00)

    # ConvFFN
    ap.add_argument("--convffn_hidden_mult", type=int, default=2)
    ap.add_argument("--convffn_kernel", type=int, default=3)
    ap.add_argument("--convffn_dropout", type=float, default=0.10)

    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # Student
    tok, student = build_student(
        recipe=args.recipe, embed_dim=args.embed_dim, eval_mode=False,
        hidden_mult=args.convffn_hidden_mult, kernel_size=args.convffn_kernel, dropout=args.convffn_dropout
    )
    student = student.to(device)
    print("Student params (M):", f"{count_trainable_params(student)/1e6:.2f}")

    # Teacher
    teacher = load_teacher(args.teacher_arch, args.teacher_ckpt, device)
    assert args.embed_dim == 512, "This config expects 512-d teacher/student embeddings."

    # Data
    tfm = build_image_transform(train=True, size=224)
    ds = CocoCaptionTrainMulti(args.image_dir_train, args.ann_train, tfm, tok,
                               max_tokens=args.max_tokens, caps_per_image=args.caps_per_image)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    collate_fn=collate_train, drop_last=True, pin_memory=True)

    # Optim/AMP
    opt = build_optimizers(student, args.lr_enc, args.lr_head, args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision and device.type=="cuda"))
    steps_per_epoch = max(len(dl)//max(args.grad_accum,1), 1)
    total_steps = args.epochs * steps_per_epoch
    global_step = 0
    opt.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs+1):
        student.train(); teacher.eval()

        # heads-only warmup
        heads_only = (epoch <= args.warmup_epochs)
        for n,p in student.named_parameters():
            if n.startswith(("vision.","text.")):
                p.requires_grad = not heads_only
        if heads_only:
            for n,p in student.named_parameters():
                if n.startswith(("v_proj","t_proj","logit_scale")):
                    assert p.requires_grad, f"Head unexpectedly frozen: {n}"

        # feature distill weight schedule
        ALPHA_FEAT = args.alpha_feat_warmup if epoch <= args.warmup_epochs else args.alpha_feat_base

        loss_m=AverageMeter(); ce_m=AverageMeter(); feat_m=AverageMeter(); logit_m=AverageMeter(); aff_m=AverageMeter()
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{args.epochs} ({'heads' if heads_only else 'full'})", ncols=100)

        for it, (pix, ids, mask, caps_raw) in enumerate(pbar, 1):
            B, K, L = ids.size()
            pix   = pix.to(device, non_blocking=True)
            ids_f = ids.to(device, non_blocking=True).view(B*K, L)
            msk_f = mask.to(device, non_blocking=True).view(B*K, L)
            caps_flat = [c for capsK in caps_raw for c in capsK]

            with torch.cuda.amp.autocast(enabled=(args.mixed_precision and device.type=='cuda')):
                # Student
                z_i_s = student.encode_image(pix)              # [B, d]
                z_t_s = student.encode_text(ids_f, msk_f)      # [B*K, d]
                assert z_i_s.requires_grad and z_t_s.requires_grad, "Embeddings lack grad; check no_grad."

                # Teacher
                with torch.no_grad():
                    z_i_t = F.normalize(teacher.encode_image(pix), dim=-1)           # [B, 512]
                    tok_t  = open_clip.get_tokenizer(args.teacher_arch)
                    txt_tokens = tok_t(caps_flat).to(device)
                    z_t_t = F.normalize(teacher.encode_text(txt_tokens), dim=-1)     # [B*K, 512]

                # Contrastive (multi-positive)
                ce_loss, _ = clip_contrastive_loss_multi(z_i_s, z_t_s, K=K,
                                                         logit_scale=student.logit_scale,
                                                         soft_i2t=args.soft_i2t)

                # Temperatures
                try:  tau_t = float(1.0 / teacher.logit_scale.exp().item())
                except: tau_t = 1.0
                try:  tau_s = float(1.0 / student.logit_scale.exp().item())
                except: tau_s = 1.0

                # Distill
                L_feat, L_logit, L_aff = distill_losses_dimaware(z_i_s, z_t_s, z_i_t, z_t_t, tau_t=tau_t, tau_s=tau_s)
                total_loss = ce_loss + ALPHA_FEAT*L_feat + args.beta_logit*L_logit + args.gamma_aff*L_aff

            # AMP + grad accum
            scaler.scale(total_loss / max(args.grad_accum,1)).backward()
            if (it % max(args.grad_accum,1)) == 0:
                with torch.no_grad(): student.logit_scale.data.clamp_(-2.5, 5.5)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                global_step += 1
                # cosine LR
                for pg in opt.param_groups:
                    base = pg.get("initial_lr", pg["lr"])
                    pg["lr"] = cosine_decay(global_step, total_steps, base)

            bs = pix.size(0)
            loss_m.update(total_loss.item(), bs); ce_m.update(ce_loss.item(), bs)
            feat_m.update(L_feat.item(), bs); logit_m.update(L_logit.item(), bs); aff_m.update(L_aff.item(), bs)
            if it % args.log_every == 0 or it == 1:
                pbar.set_postfix(loss=f"{loss_m.avg:.3f}", ce=f"{ce_m.avg:.3f}",
                                 feat=f"{feat_m.avg:.3f}", logit=f"{logit_m.avg:.3f}",
                                 aff=f"{aff_m.avg:.3f}",
                                 temp_s=f"{float(student.logit_scale.exp().item()):.1f}")

        if (epoch % args.save_every)==0:
            ckpt = {
                "recipe": args.recipe,
                "epoch": epoch,
                "student_state": student.state_dict(),
                "embed_dim": args.embed_dim,
                "convffn": {"hidden_mult": args.convffn_hidden_mult,
                            "kernel": args.convffn_kernel,
                            "dropout": args.convffn_dropout},
                "caps_per_image": args.caps_per_image,
                "soft_i2t": args.soft_i2t,
            }
            path = os.path.join(args.output_dir, f"tinyclip_{args.recipe}_convffn_mc_e{epoch}.pt")
            torch.save(ckpt, path); print("Saved:", path)

    print("Training complete.")

if __name__ == "__main__":
    main()
