import os, json, random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def build_image_transform(train=True, size=224):
    if train:
        return transforms.Compose([
            transforms.Resize(int(size*1.15), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop(size, scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std =(0.26862954, 0.26130258, 0.27577711)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(size*1.15), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std =(0.26862954, 0.26130258, 0.27577711)),
        ])

class CocoCaptionTrainMulti(Dataset):
    """
    Returns: pixel [3,H,W], input_ids [K,L], attention_mask [K,L], raw_caps [K]
    """
    def __init__(self, image_dir, ann_json, transform, tokenizer, max_tokens=32, caps_per_image=2):
        self.image_dir = image_dir
        self.data = json.load(open(ann_json, "r"))
        self.transform = transform
        self.tok = tokenizer
        self.max_tokens = max_tokens
        self.K = max(1, int(caps_per_image))
        # id -> filename
        self.id2file = {img["id"]: f"COCO_train2014_{img['id']:012d}.jpg" for img in self.data["images"]}
        caps_by_img = {}
        for ann in self.data["annotations"]:
            caps_by_img.setdefault(ann["image_id"], []).append(ann["caption"])
        self.samples = [(img_id, caps) for img_id, caps in caps_by_img.items()]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_id, caps = self.samples[idx]
        path = os.path.join(self.image_dir, self.id2file[img_id])
        with Image.open(path) as im: im = im.convert("RGB")
        pixel = self.transform(im)
        # choose K captions; tokenize to fixed max length so stacking is safe
        picked = random.sample(caps, self.K) if len(caps) >= self.K else [random.choice(caps) for _ in range(self.K)]
        enc = self.tok(picked, padding="max_length", truncation=True, max_length=self.max_tokens, return_tensors="pt")
        return pixel, enc["input_ids"], enc["attention_mask"], picked

def collate_train(batch):
    pixels, ids, masks, caps_raw = zip(*batch)
    pixels = torch.stack(pixels, 0)     # [B,3,H,W]
    ids    = torch.stack(ids, 0)        # [B,K,L]
    masks  = torch.stack(masks, 0)      # [B,K,L]
    return pixels, ids, masks, list(caps_raw)

# -------- Karpathy eval helpers --------
def build_eval_transform(size=224):
    return build_image_transform(train=False, size=size)

def load_coco_karpathy(json_path):
    data = json.load(open(json_path, "r"))
    imgs = [im for im in data["images"] if im.get("split") == "test"]
    assert len(imgs) == 5000, f"Expected 5k test, got {len(imgs)}"
    img_records, cap_texts, cap_owner = [], [], []
    for i, im in enumerate(imgs):
        fname = im.get("filename") or im.get("filepath", "")
        if not fname.endswith(".jpg") and "COCO_" in fname:
            fname += ".jpg"
        sents = im.get("sentences", [])
        texts = [s.get("raw","") for s in sents]
        start = len(cap_texts)
        cap_texts.extend(texts)
        cap_owner.extend([i]*len(texts))
        img_records.append({
            "filename": fname,
            "filepath": im.get("filepath", ""),
            "cap_start": start,
            "cap_num": len(texts)
        })
    return img_records, cap_texts, cap_owner

def resolve_coco_path(filename, filepath, val_dir, train_dir):
    if isinstance(filepath, str) and filepath:
        if "val" in filepath:   return os.path.join(val_dir, filename)
        if "train" in filepath: return os.path.join(train_dir, filename)
    if "val2014" in filename:   return os.path.join(val_dir, filename)
    if "train2014" in filename: return os.path.join(train_dir, filename)
    p1 = os.path.join(val_dir, filename)
    return p1 if os.path.exists(p1) else os.path.join(train_dir, filename)

def load_f30k_karpathy(json_path):
    data = json.load(open(json_path, "r"))
    imgs = [im for im in data["images"] if im.get("split") == "test"]
    assert len(imgs) == 1000, f"Expected 1k test, got {len(imgs)}"
    img_records, cap_texts, cap_owner = [], [], []
    for i, im in enumerate(imgs):
        fname = im.get("filename")
        sents = im.get("sentences", [])
        texts = [s.get("raw","") for s in sents]
        start = len(cap_texts)
        cap_texts.extend(texts)
        cap_owner.extend([i]*len(texts))
        img_records.append({"filename": fname, "cap_start": start, "cap_num": len(texts)})
    return img_records, cap_texts, cap_owner

class ImageEvalCOCODS(Dataset):
    def __init__(self, img_recs, tfm, val_dir, train_dir):
        self.img_recs = img_recs; self.tfm = tfm
        self.val_dir = val_dir; self.train_dir = train_dir
    def __len__(self): return len(self.img_recs)
    def __getitem__(self, i):
        from PIL import Image
        rec = self.img_recs[i]
        path = resolve_coco_path(rec["filename"], rec.get("filepath",""), self.val_dir, self.train_dir)
        with Image.open(path) as im: im = im.convert("RGB")
        return i, self.tfm(im)

class ImageEvalF30KDS(Dataset):
    def __init__(self, img_recs, tfm, root_dir):
        self.img_recs = img_recs; self.tfm = tfm; self.root = root_dir
    def __len__(self): return len(self.img_recs)
    def __getitem__(self, i):
        from PIL import Image
        rec = self.img_recs[i]
        path = os.path.join(self.root, rec["filename"])
        with Image.open(path) as im: im = im.convert("RGB")
        return i, self.tfm(im)
