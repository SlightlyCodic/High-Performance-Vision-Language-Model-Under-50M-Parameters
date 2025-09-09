# TinyViT-11M-MiniLM-L6 With ConvFFN and MC(Multi-caption) (34.4M) — Reproducible Training & Karpathy Retrieval Eval

A compact CLIP-style dual encoder (TinyViT-11M + MiniLM-L6, ~34.4M params) with a **ConvFFN** text head, **K-positive** multi-caption training, and **TinyCLIP-style** distillation from **OpenCLIP ViT-B/32**.

- **Train** on MS-COCO 2014 captions
- **Evaluate** on **Karpathy COCO-5k** and **Flickr30k-1k**
- Checkpoints include ConvFFN config & embed dim; the evaluator **rebuilds** the exact text head & tokenizer

2) Data

MS-COCO 2014: Images in train2014/ and val2014/ folders, and COCO captions jsons:
-annotations/captions_train2014.json
-annotations/captions_val2014.json

Karpathy splits:
-dataset_coco.json (Karpathy 5k split format: "images" with "split": "test" and "sentences")
-dataset_flickr30k.json (Karpathy 1k split)

Flickr30k images: flickr30k-images/
-You can place paths anywhere; pass them via CLI flags.

3) Train (COCO)
Example-
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
