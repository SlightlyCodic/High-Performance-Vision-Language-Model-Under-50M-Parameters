import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoTokenizer, AutoModel

# --- ConvFFN head ---
def masked_mean_pool(x, mask):
    if mask is None: return x.mean(dim=1)
    mask = mask.float()
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / denom

class ConvFFNTextHead(nn.Module):
    """
    Linear(H->H') → DepthwiseConv1d over tokens → GELU+Dropout →
    masked mean pool → Linear(H'->D) + LayerNorm(D)
    """
    def __init__(self, in_dim, out_dim, hidden_mult=2, kernel_size=3, dropout=0.1):
        super().__init__()
        h = max(out_dim * hidden_mult, out_dim)
        self.pre  = nn.Linear(in_dim, h)
        self.dw   = nn.Conv1d(h, h, kernel_size=kernel_size, padding=kernel_size//2, groups=h)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(h, out_dim)
        self.ln   = nn.LayerNorm(out_dim)
    def forward(self, seq, mask=None):
        y = self.pre(seq)                         # [B,L,H']
        y = self.dw(y.transpose(1,2)).transpose(1,2)
        y = self.act(self.drop(y))
        y = masked_mean_pool(y, mask)             # [B,H']
        return self.ln(self.proj(y))              # [B,D]

# --- Text backbones ---
TEXT_MODEL_CANDIDATES = {
    "minilm_l6_h384": [
        "microsoft/MiniLM-L6-H384-uncased",
        "nreimers/MiniLM-L6-H384-uncased",
        "microsoft/MiniLM-L12-H384-uncased",
        "sentence-transformers/all-MiniLM-L6-v2",
    ],
    "tinybert_4l_312": [
        "huawei-noah/TinyBERT_General_4L_312D",
        "prajjwal1/bert-mini",
        "prajjwal1/bert-tiny",
    ],
}

def load_text_stack(kind:str):
    if kind == "minilm_l6":
        last=None
        for repo in TEXT_MODEL_CANDIDATES["minilm_l6_h384"]:
            try:
                return AutoTokenizer.from_pretrained(repo, use_fast=True), AutoModel.from_pretrained(repo)
            except Exception as e: last=e
        raise RuntimeError(f"Failed to load MiniLM-L6-H384. Last error: {last}")
    elif kind == "tinybert_4l_312":
        last=None
        for repo in TEXT_MODEL_CANDIDATES["tinybert_4l_312"]:
            try:
                return AutoTokenizer.from_pretrained(repo, use_fast=True), AutoModel.from_pretrained(repo)
            except Exception as e: last=e
        raise RuntimeError(f"Failed to load TinyBERT-4L-312. Last error: {last}")
    else:
        raise ValueError(kind)

# --- Students ---
class TinyCLIPTrain(nn.Module):
    def __init__(self, vision_model, text_model, embed_dim=512,
                 hidden_mult=2, kernel_size=3, dropout=0.1):
        super().__init__()
        self.vision = vision_model
        self.text   = text_model
        if hasattr(self.vision, "num_features"): v_dim = self.vision.num_features
        elif hasattr(self.vision, "num_features_final"): v_dim = self.vision.num_features_final
        else: raise ValueError("Cannot infer vision features.")
        t_dim = self.text.config.hidden_size
        self.v_proj = nn.Sequential(nn.Linear(v_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.t_proj = ConvFFNTextHead(t_dim, embed_dim,
                                      hidden_mult=hidden_mult,
                                      kernel_size=kernel_size,
                                      dropout=dropout)
        self.logit_scale = nn.Parameter(torch.tensor(4.6052))  # ln(100)

    def encode_image(self, pixel_values):
        z = self.vision(pixel_values)
        if isinstance(z, (list,tuple)): z = z[-1]
        return F.normalize(self.v_proj(z), dim=-1)

    def encode_text(self, input_ids, attention_mask):
        out = self.text(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        seq = out.last_hidden_state
        z   = self.t_proj(seq, attention_mask)
        return F.normalize(z, dim=-1)

    def forward(self, pixel_values, input_ids, attention_mask):
        return self.encode_image(pixel_values), self.encode_text(input_ids, attention_mask)

class TinyCLIPEval(nn.Module):
    def __init__(self, vision_model, text_model, embed_dim=512,
                 hidden_mult=2, kernel_size=3, dropout=0.1):
        super().__init__()
        self.vision = vision_model
        self.text   = text_model
        if hasattr(self.vision, "num_features"): v_dim = self.vision.num_features
        elif hasattr(self.vision, "num_features_final"): v_dim = self.vision.num_features_final
        else: raise ValueError("Cannot infer vision features.")
        t_dim = self.text.config.hidden_size
        self.v_proj = nn.Sequential(nn.Linear(v_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.t_proj = ConvFFNTextHead(t_dim, embed_dim,
                                      hidden_mult=hidden_mult,
                                      kernel_size=kernel_size,
                                      dropout=dropout)

    def encode_image(self, pixel_values):
        z = self.vision(pixel_values)
        if isinstance(z, (list,tuple)): z = z[-1]
        return F.normalize(self.v_proj(z), dim=-1)

    def encode_text(self, input_ids, attention_mask):
        out = self.text(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        seq = out.last_hidden_state
        z   = self.t_proj(seq, attention_mask)
        return F.normalize(z, dim=-1)

def build_student(recipe:str, embed_dim:int=512, eval_mode=False,
                  hidden_mult=2, kernel_size=3, dropout=0.1):
    r = recipe.lower()
    if r == "vit_s16_minilm_l6":
        vision = timm.create_model("vit_small_patch16_224", pretrained=True if not eval_mode else False, num_classes=0)
        tok, txt = load_text_stack("minilm_l6")
    elif r == "effnet_b0_tinybert_4l_312":
        vision = timm.create_model("efficientnet_b0", pretrained=True if not eval_mode else False, num_classes=0, global_pool="avg")
        tok, txt = load_text_stack("tinybert_4l_312")
    elif r == "tinyvit_11m_minilm_l6":
        vision = timm.create_model("tiny_vit_11m_224", pretrained=True if not eval_mode else False, num_classes=0)
        tok, txt = load_text_stack("minilm_l6")
    else:
        raise ValueError(r)
    if eval_mode:
        model = TinyCLIPEval(vision, txt, embed_dim=embed_dim,
                             hidden_mult=hidden_mult, kernel_size=kernel_size, dropout=dropout)
    else:
        model = TinyCLIPTrain(vision, txt, embed_dim=embed_dim,
                              hidden_mult=hidden_mult, kernel_size=kernel_size, dropout=dropout)
    return tok, model
