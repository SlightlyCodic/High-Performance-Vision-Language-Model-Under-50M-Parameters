import os, json, argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models import build_student
from .data import (
    build_eval_transform, load_coco_karpathy, load_f30k_karpathy,
    ImageEvalCOCODS, ImageEvalF30KDS
)

def recall_at_k(ranks:torch.Tensor, ks=(1,5,10)):
    res = {}
    n = ranks.numel()
    for k in ks:
        res[f"R@{k}"] = float((ranks <= k).sum().item() * 100.0 / n)
    res["MedR"] = float(ranks.median().item())
    return res

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", choices=["coco","f30k"], required=True)

    # COCO
    ap.add_argument("--karpathy_coco_json", type=str, default="")
    ap.add_argument("--img_dir_val", type=str, default="")
    ap.add_argument("--img_dir_train", type=str, default="")

    # F30K
    ap.add_argument("--karpathy_f30k_json", type=str, default="")
    ap.add_argument("--f30k_img_dir", type=str, default="")

    # Common
    ap.add_argument("--embed_dim", type=int, default=512)
    ap.add_argument("--max_tokens", type=int, default=32)
    ap.add_argument("--batch_size_img", type=int, default=256)
    ap.add_argument("--batch_size_txt", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--save_json", action="store_true")
    ap.add_argument("--out_json", type=str, default="retrieval_results.json")
    return ap.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Build model with convffn hyperparams from checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    convffn_cfg = ckpt.get("convffn", {"hidden_mult":2, "kernel":3, "dropout":0.1})
    tok, student = build_student(
        args.recipe, embed_dim=args.embed_dim, eval_mode=True,
        hidden_mult=convffn_cfg.get("hidden_mult",2),
        kernel_size=convffn_cfg.get("kernel",3),
        dropout=convffn_cfg.get("dropout",0.1)
    )
    student.to(device).eval()
    miss, unexp = student.load_state_dict(ckpt["student_state"], strict=False)
    print("[warn] Missing keys:", miss)
    print("[warn] Unexpected keys:", unexp)

    # 2) Load dataset
    tfm = build_eval_transform()
    if args.dataset == "coco":
        assert args.karpathy_coco_json and args.img_dir_val and args.img_dir_train
        img_recs, cap_texts, cap_owner = load_coco_karpathy(args.karpathy_coco_json)
        img_ds = ImageEvalCOCODS(img_recs, tfm, args.img_dir_val, args.img_dir_train)
    else:
        assert args.karpathy_f30k_json and args.f30k_img_dir
        img_recs, cap_texts, cap_owner = load_f30k_karpathy(args.karpathy_f30k_json)
        img_ds = ImageEvalF30KDS(img_recs, tfm, args.f30k_img_dir)

    N_img, N_cap = len(img_recs), len(cap_texts)
    print(f"[info] {args.dataset.upper()} Karpathy: {N_img} images, {N_cap} captions")

    # 3) Encode images
    img_dl = DataLoader(img_ds, batch_size=args.batch_size_img, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    I = torch.empty((N_img, args.embed_dim), dtype=torch.float32)
    for idxs, pix in tqdm(img_dl, desc="Encode images"):
        z = student.encode_image(pix.to(device))
        I[idxs] = z.detach().cpu()

    # 4) Encode texts (with attention_mask)
    T = torch.empty((N_cap, args.embed_dim), dtype=torch.float32)
    bs = args.batch_size_txt
    for s in tqdm(range(0, N_cap, bs), desc="Encode texts"):
        chunk = cap_texts[s:s+bs]
        enc = tok(chunk, padding="max_length", truncation=True, max_length=args.max_tokens, return_tensors="pt")
        ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        z = student.encode_text(ids, mask)
        T[s:s+bs] = z.detach().cpu()

    sims = I @ T.t()  # [N_img, N_cap]

    # I->T
    ranks_i2t = []
    for i, rec in enumerate(img_recs):
        start, K = rec["cap_start"], rec["cap_num"]
        order = torch.argsort(sims[i], descending=True)
        r = min((order == p).nonzero(as_tuple=True)[0].item() for p in range(start, start+K)) + 1
        ranks_i2t.append(r)
    I2T = recall_at_k(torch.tensor(ranks_i2t, dtype=torch.int32))

    # T->I
    ranks_t2i = []
    for j, owner in enumerate(cap_owner):
        order = torch.argsort(sims[:, j], descending=True)
        r = (order == owner).nonzero(as_tuple=True)[0].item() + 1
        ranks_t2i.append(r)
    T2I = recall_at_k(torch.tensor(ranks_t2i, dtype=torch.int32))

    res = {"dataset": args.dataset.upper(), "recipe": args.recipe, "checkpoint": args.checkpoint,
           "embed_dim": args.embed_dim, "convffn": convffn_cfg, "I2T": I2T, "T2I": T2I}
    print("I2T:", I2T)
    print("T2I:", T2I)

    if args.save_json:
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=2)
        print("Saved:", args.out_json)

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
