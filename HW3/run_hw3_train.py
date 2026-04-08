#!/usr/bin/env python3
"""Headless training for HW3: generates figures for LaTeX report. Run from HW3/."""
import math
import os
import urllib.request
import random
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
DATA_DIR = Path("data")
FIG_DIR = Path(".")
DATA_DIR.mkdir(exist_ok=True)

# --- data ---
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = DATA_DIR / "input.txt"
if not DATA_PATH.exists():
    urllib.request.urlretrieve(URL, DATA_PATH)

full_text = DATA_PATH.read_text(encoding="utf-8")
split_idx = int(len(full_text) * 0.8)
train_text, val_text = full_text[:split_idx], full_text[split_idx:]

VOCAB_SIZE = 500
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()
trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=["<unk>", "<pad>"],
)
tokenizer.train_from_iterator([train_text], trainer=trainer)
tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"), pad_token="<pad>")
actual_vocab = tokenizer.get_vocab_size()
assert actual_vocab <= VOCAB_SIZE


def encode(text):
    return tokenizer.encode(text).ids


train_ids, val_ids = encode(train_text), encode(val_text)

BLOCK_SIZE, STRIDE = 65, 32


class ShakespeareDataset(Dataset):
    def __init__(self, token_ids, block_size, stride):
        self.samples = []
        for i in range(0, len(token_ids) - block_size + 1, stride):
            chunk = token_ids[i : i + block_size]
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


train_ds = ShakespeareDataset(train_ids, BLOCK_SIZE, STRIDE)
val_ds = ShakespeareDataset(val_ids, BLOCK_SIZE, STRIDE)
BATCH_SIZE = 32
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# --- model ---


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self._last_attn = None

    def forward(self, x):
        B, T, D = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        self._last_attn = attn.detach()
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=8, n_layers=2, d_ff=512, max_len=4096, dropout=0.1, use_pe=True):
        super().__init__()
        self.d_model = d_model
        self.use_pe = use_pe
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len) if use_pe else None
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        x = self.tok_emb(idx) * math.sqrt(self.d_model)
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)

    def get_attention_from_layer(self, layer_idx):
        return self.blocks[layer_idx].attn._last_attn


D_MODEL, N_HEADS, N_LAYERS, D_FF = 128, 8, 2, 512
DROPOUT, LR = 0.1, 3e-4
EPOCHS = int(os.environ.get("HW3_EPOCHS", "22"))
ABLATE_EPOCHS = int(os.environ.get("HW3_ABLATE", "12"))

criterion = nn.CrossEntropyLoss()


def evaluate_ppl(loader, model):
    model.eval()
    total_loss, total_tok = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits.view(-1, actual_vocab), yb.view(-1))
            total_loss += loss.item() * yb.numel()
            total_tok += yb.numel()
    mean_ce = total_loss / total_tok
    return mean_ce, math.exp(mean_ce)


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss, total_tok = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits.view(-1, actual_vocab), yb.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * yb.numel()
        total_tok += yb.numel()
    return total_loss / total_tok


def main():
    model = TinyTransformerLM(
        vocab_size=actual_vocab,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_len=BLOCK_SIZE + 32,
        dropout=DROPOUT,
        use_pe=True,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    history = {"train_ce": [], "val_ce": [], "val_ppl": []}
    for epoch in range(1, EPOCHS + 1):
        tr_ce = train_one_epoch(model, train_loader, optimizer)
        val_ce, val_ppl = evaluate_ppl(val_loader, model)
        history["train_ce"].append(tr_ce)
        history["val_ce"].append(val_ce)
        history["val_ppl"].append(val_ppl)
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(f"Epoch {epoch:3d} | train CE {tr_ce:.4f} | val CE {val_ce:.4f} | val PPL {val_ppl:.2f}")

    final_ce, final_ppl = history["val_ce"][-1], history["val_ppl"][-1]
    Path("hw3_metrics.txt").write_text(f"final_val_ce={final_ce}\nfinal_val_ppl={final_ppl}\n", encoding="utf-8")

    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10})
    fig, ax = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
    epochs = range(1, len(history["train_ce"]) + 1)
    ax[0].plot(epochs, history["train_ce"], label="Train CE", color="#2E86AB")
    ax[0].plot(epochs, history["val_ce"], label="Val CE", color="#E94F37")
    ax[0].set_ylabel("Cross-entropy (nats)")
    ax[0].legend()
    ax[0].set_title("Training and validation cross-entropy")
    ax[1].plot(epochs, history["val_ppl"], color="#44AF69", marker="o", markersize=3)
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Perplexity")
    ax[1].set_title("Validation perplexity = exp(val CE)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_01_loss_ppl.png", bbox_inches="tight")
    plt.close()
    print("Saved fig_01_loss_ppl.png")

    def ids_to_labels(ids_list):
        return [tokenizer.decode([i]) for i in ids_list]

    @torch.no_grad()
    def plot_attention(m, x_sample, layer_idx, head_idx, max_len, fname):
        m.eval()
        x = x_sample[:max_len].unsqueeze(0).to(DEVICE)
        _ = m(x)
        attn = m.get_attention_from_layer(layer_idx)[0, head_idx].cpu().numpy()
        T = attn.shape[0]
        labels = ids_to_labels(x_sample[:T].tolist())
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(attn, cmap="Blues", vmin=0, vmax=max(attn.max(), 1e-8))
        ax.set_xticks(np.arange(T))
        ax.set_yticks(np.arange(T))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        ax.set_title(f"Causal attention — layer {layer_idx}, head {head_idx}")
        plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        plt.savefig(FIG_DIR / fname, bbox_inches="tight")
        plt.close()

    xb, _ = next(iter(val_loader))
    plot_attention(model, xb[0], 0, 0, 28, "fig_02_attention_L0H0.png")
    plot_attention(model, xb[0], 1, 2, 28, "fig_03_attention_L1H2.png")
    print("Saved attention figures")

    # PE ablation
    model_no_pe = TinyTransformerLM(
        actual_vocab, D_MODEL, N_HEADS, N_LAYERS, D_FF, BLOCK_SIZE + 32, DROPOUT, use_pe=False
    ).to(DEVICE)
    opt2 = torch.optim.AdamW(model_no_pe.parameters(), lr=LR, weight_decay=0.01)
    hist_no_pe = {"val_ppl": []}
    for epoch in range(1, ABLATE_EPOCHS + 1):
        train_one_epoch(model_no_pe, train_loader, opt2)
        _, val_ppl = evaluate_ppl(val_loader, model_no_pe)
        hist_no_pe["val_ppl"].append(val_ppl)
        if epoch == 1 or epoch % 4 == 0 or epoch == ABLATE_EPOCHS:
            print(f"[No PE] Epoch {epoch} | val PPL {val_ppl:.2f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, ABLATE_EPOCHS + 1), hist_no_pe["val_ppl"], label="No PE", color="#E94F37")
    ax.plot(range(1, ABLATE_EPOCHS + 1), history["val_ppl"][:ABLATE_EPOCHS], label="With sinusoidal PE", color="#2E86AB")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation perplexity")
    ax.legend()
    ax.set_title("Ablation: positional encoding")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_04_ablation_pe.png", bbox_inches="tight")
    plt.close()
    print("Saved fig_04_ablation_pe.png")

    # Short context
    BLOCK_SHORT, STRIDE_S = 33, 16
    train_ds_s = ShakespeareDataset(train_ids, BLOCK_SHORT, STRIDE_S)
    val_ds_s = ShakespeareDataset(val_ids, BLOCK_SHORT, STRIDE_S)
    tl_s = DataLoader(train_ds_s, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    vl_s = DataLoader(val_ds_s, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    model_short = TinyTransformerLM(
        actual_vocab, D_MODEL, N_HEADS, N_LAYERS, D_FF, BLOCK_SHORT + 16, DROPOUT, use_pe=True
    ).to(DEVICE)
    opt_s = torch.optim.AdamW(model_short.parameters(), lr=LR, weight_decay=0.01)
    SHORT_EPOCHS = int(os.environ.get("HW3_SHORT_EP", "10"))
    hist_short = []
    for epoch in range(1, SHORT_EPOCHS + 1):
        train_one_epoch(model_short, tl_s, opt_s)
        _, vp = evaluate_ppl(vl_s, model_short)
        hist_short.append(vp)
    Path("hw3_short_ctx_ppl.txt").write_text(f"short_ctx_final_ppl={hist_short[-1]}\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, SHORT_EPOCHS + 1), hist_short, marker="o", color="#F18F01")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val perplexity")
    ax.set_title("Shorter context (block=33 tokens, 32 positions)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_05_short_context.png", bbox_inches="tight")
    plt.close()
    print("Saved fig_05_short_context.png")

    print("Done. Final val PPL (main):", final_ppl)


if __name__ == "__main__":
    main()
