# TinyCLIP-ConvFFN (34.4M)

**Compact Image–Text Retrieval with Multi-Caption Training & Distillation**

**TL;DR** — A compact CLIP-style dual encoder (**TinyViT-11M** vision + **MiniLM-L6** text; **\~34.4M params**) with a lightweight **ConvFFN** text head. Trained on **MS-COCO 2014** with **K-positive (K=2)** multi-caption contrastive learning and **TinyCLIP-style, temperature-aware distillation** from **OpenCLIP ViT-B/32**. Evaluated on **Karpathy COCO-5k** and **Flickr30k-1k** via a reproducible pipeline.

---

## Contents

* [Highlights](#highlights)
* [Repo Structure](#repo-structure)
* [Setup](#setup)
* [Data](#data)
* [Training](#training)
* [Evaluation (Karpathy)](#evaluation-karpathy)
* [Reproduced Results](#reproduced-results)
* [Tips & Troubleshooting](#tips--troubleshooting)
* [Reproducibility Notes](#reproducibility-notes)
* [Use the Checkpoint Programmatically](#use-the-checkpoint-programmatically)
* [License](#license)
* [References](#references)

---

## Highlights

* **Student**: TinyViT-11M (vision) + MiniLM-L6 (text) → shared **512-d** space
* **Text head**: **ConvFFN** = Linear → depthwise 1-D token conv → GELU/Dropout → **mask-aware mean pooling** → Linear + LayerNorm
* **Objective**: CLIP contrastive with **K-positive** (K=2) supervision

  * Soft targets for image→text; standard CE for text→image
* **Distillation**: from **OpenCLIP ViT-B/32** (frozen) with

  * **Feature L2** (on normalised embeddings)
  * **Row/column KL** over similarity matrices (**temperature-aware**)
* **Stability**: heads-only warm-up (3 ep) → full fine-tune (to 30), **logit\_scale clamp**, cosine LR, AMP, grad-accum

---

## Repo Structure

```
tinyclip-convffn/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
└── src/
    ├── __init__.py
    ├── utils.py            # meters, seed, param counting
    ├── models.py           # ConvFFN head + student builders
    ├── data.py             # COCO train loader, Karpathy loaders
    ├── losses.py           # multi-caption contrast + KD losses
    ├── optim.py            # AdamW + cosine decay
    ├── train.py            # training CLI
    └── eval_karpathy.py    # COCO/F30K retrieval CLI
```

---

## Setup

```bash
git clone <your-repo-url> tinyclip-convffn
cd tinyclip-convffn
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**requirements.txt**

```
torch>=2.2
torchvision>=0.17
timm>=0.9.12
transformers>=4.40
open_clip_torch>=2.24.0
tqdm>=4.66
pillow>=10.2
numpy>=1.26
```

> The first run downloads the MiniLM text encoder from Hugging Face. If you use private/gated models, run `huggingface-cli login`.

---

## Data
Kaparthy Split can be downloaded from here - https://www.kaggle.com/datasets/shtvkumar/karpathy-splits
### MS-COCO 2014 (training + Karpathy eval)

* Images:

  * `train2014/`
  * `val2014/`
* Official COCO captions:

  * `annotations/captions_train2014.json`
  * `annotations/captions_val2014.json`
* **Karpathy split JSON**: `dataset_coco.json` (must include `"images"` with `"split": "test"` and `sentences: [{"raw": "..."}]`).

### Flickr30k (Karpathy eval)

* Images: `flickr30k-images/`
* **Karpathy split JSON**: `dataset_flickr30k.json` (same schema as above).

> Place paths anywhere; pass them via CLI flags below.

---

## Training

Train on COCO with K=2 multi-caption supervision and OpenCLIP ViT-B/32 teacher:

```bash
python -m src.train \
  --recipe tinyvit_11m_minilm_l6 \
  --embed_dim 512 \
  --image_dir_train /path/to/train2014 \
  --ann_train /path/to/annotations/captions_train2014.json \
  --output_dir outputs_tinyclip_convffn_mc \
  --epochs 30 --warmup_epochs 3 \
  --batch_size 256 --grad_accum 4 \
  --caps_per_image 2 \
  --teacher_arch ViT-B-32 --teacher_ckpt laion2b_s34b_b79k
```

**Key flags**

* `--caps_per_image` (default **2**): K-positive multi-caption
* `--soft_i2t` (default **on**): soft targets for I→T
* `--lr_enc 1e-4 --lr_head 5e-4 --weight_decay 0.01` (defaults)
* AMP on by default if CUDA is available

**Checkpoints** are saved each epoch, e.g.:

```
outputs_tinyclip_convffn_mc/tinyclip_tinyvit_11m_minilm_l6_convffn_mc_e30.pt
```

and include metadata:

```json
{
  "recipe": "...",
  "epoch": 30,
  "embed_dim": 512,
  "convffn": {"hidden_mult": 2, "kernel": 3, "dropout": 0.1},
  "caps_per_image": 2,
  "soft_i2t": true,
  "student_state": {...}
}
```

---

## Evaluation (Karpathy)

### COCO-5k

```bash
python -m src.eval_karpathy \
  --recipe tinyvit_11m_minilm_l6 \
  --checkpoint outputs_tinyclip_convffn_mc/tinyclip_tinyvit_11m_minilm_l6_convffn_mc_e30.pt \
  --dataset coco \
  --karpathy_coco_json /path/to/dataset_coco.json \
  --img_dir_val /path/to/val2014 \
  --img_dir_train /path/to/train2014 \
  --save_json
```

### Flickr30k-1k

```bash
python -m src.eval_karpathy \
  --recipe tinyvit_11m_minilm_l6 \
  --checkpoint outputs_tinyclip_convffn_mc/tinyclip_tinyvit_11m_minilm_l6_convffn_mc_e30.pt \
  --dataset f30k \
  --karpathy_f30k_json /path/to/dataset_flickr30k.json \
  --f30k_img_dir /path/to/flickr30k-images \
  --save_json
```

The script prints **I→T / T→I Recall@{1,5,10}** and **Median Rank** and writes `retrieval_results.json` if `--save_json` is set.

---

## Reproduced Results

**Training**: MS-COCO 2014.
**Evaluation**: Karpathy splits.

### MS-COCO (Karpathy 5k)

| Model                                            | Train setting |  I→T R\@1 |      R\@5 |     R\@10 |  T→I R\@1 |      R\@5 |     R\@10 |  MedR |
| ------------------------------------------------ | ------------- | --------: | --------: | --------: | --------: | --------: | --------: | ----: |
| **Ours (TinyViT-11M + MiniLM-L6, ConvFFN, K=2)** | COCO-trained  | **42.32** | **70.56** | **80.92** | **29.28** | **58.80** | **71.48** | **2** |

### Flickr30k (Karpathy 1k)

| Model                                            | Train setting         |  I→T R\@1 |      R\@5 |     R\@10 |  T→I R\@1 |      R\@5 |     R\@10 |  MedR |
| ------------------------------------------------ | --------------------- | --------: | --------: | --------: | --------: | --------: | --------: | ----: |
| **Ours (TinyViT-11M + MiniLM-L6, ConvFFN, K=2)** | **COCO-trained only** | **56.10** | **83.50** | **90.00** | **40.66** | **71.38** | **80.38** | **2** |

> We observed that **ConvFFN > MLP** and **K=2 > K=1** consistently; multi-caption notably boosts **I→T**, while ConvFFN improves both directions.

---

## Tips & Troubleshooting

* **Tokenizer/head mismatch at eval**
  The evaluator **rebuilds** the **ConvFFN** head and **MiniLM tokenizer** using the checkpoint metadata. Warnings like

  ```
  [warn] Missing keys: []
  [warn] Unexpected keys: ['logit_scale']
  ```

  are expected: `logit_scale` is a training-time parameter.

* **OOM / memory**
  Reduce `--batch_size`, increase `--grad_accum`, or disable AMP (`--mixed_precision False`). In eval, lower `--batch_size_img` / `--batch_size_txt`.

* **Karpathy JSON**
  Ensure `"images"` entries have `"split": "test"` and `sentences: [{"raw": "..."}]`. For COCO, the evaluator resolves filenames across `val2014/` and `train2014/`.

* **Hugging Face auth**
  If MiniLM loading fails with 401, run `huggingface-cli login`, or switch to one of the fallback repos in `TEXT_MODEL_CANDIDATES` (see `src/models.py`).

---

## Reproducibility Notes

* Fixed seeds and per-epoch dataloader shuffles.
* CLIP-style image normalisation shared by student & teacher.
* Checkpoints store **ConvFFN hyper-params** and **embed dim**; the evaluator restores the exact encoder stack.
* Embeddings are computed once per modality; retrieval uses the full similarity matrix for fair ranking.

---

## Use the Checkpoint Programmatically

```python
import torch
from src.models import build_student

ckpt_path = "outputs_tinyclip_convffn_mc/tinyclip_tinyvit_11m_minilm_l6_convffn_mc_e30.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")

tok, model = build_student(
    "tinyvit_11m_minilm_l6",
    embed_dim=ckpt.get("embed_dim", 512),
    eval_mode=True,
    hidden_mult=ckpt["convffn"]["hidden_mult"],
    kernel_size=ckpt["convffn"]["kernel"],
    dropout=ckpt["convffn"]["dropout"]
)
model.load_state_dict(ckpt["student_state"], strict=False)
model.eval()

# Encode
# z_i = model.encode_image(pixel_batch)                 # [B,512] (L2-normalised)
# z_t = model.encode_text(input_ids, attention_mask)    # [B,512] (L2-normalised)
```

---

## License

MIT — see `LICENSE`.
Datasets keep their original licenses (MS-COCO, Flickr30k). This repo does **not** redistribute images or annotations.

---

## References

* **CLIP** — Radford et al., 2021. *Learning Transferable Visual Models From Natural Language Supervision.*
* **OpenCLIP** — Ilharco et al., 2021–2023. *OpenCLIP: Reproducing CLIP Training at Open Scale.*
* **TinyCLIP** — Wu et al., 2023. *TinyCLIP: CLIP Distillation via Affinity Mimicking and Weight Inheritance.*
* **MobileCLIP** — Luo et al., 2023. *MobileCLIP: Fast Image-Text Models for Mobile Devices.*
* **SigLIP** — Zhai et al., 2023. *Sigmoid Loss for Language-Image Pre-Training.*
* **TinyViT** — Wu et al., 2022. *TinyViT: Fast Pretraining Distillation for Small Vision Transformers.*
* **MiniLM** — Wang et al., 2020. *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers.*
