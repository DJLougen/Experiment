#!/usr/bin/env python3
# dvx_mpk_dorsal_ventral_text.py
# Full text-only DVX architecture with MPK (Magno/Parvo/Konio) and Dorsal/Ventral streams.
# - No vision filters/transforms. Pure character LM on Tiny Shakespeare.
# - M (magno): long-range, coarse temporal integration (dilated causal conv)
# - P (parvo): short-range, fine detail (short causal conv)
# - K (konio): modulatory gates (channel SE + token-wise mixing)
# - Dorsal stream: integrates M-dominant features (sequence dynamics)
# - Ventral stream: integrates P-dominant features (content/detail)
# - Streams are cross-talked with K and fused per token.
#
# Usage:
#   python dvx_mpk_dorsal_ventral_text.py --steps 800 --batch_size 64 --seq_len 256
#   (add --layers, --channels, etc. as needed)
#
# macOS (M-series) friendly via MPS; falls back to CPU.
#
# This is designed to be easy to swap into your DVX experiments:
# - The "DVXTextMPK" class is the model.
# - The training loop is minimal and prints tokens/sec.
# - Autodownloads Tiny Shakespeare.
#
import os, time, math, argparse, urllib.request, random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------
# Utilities
# ---------------------------------
SHAKES_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def ensure_shakespeare(path: str = "input.txt") -> str:
    if not os.path.exists(path):
        print("Downloading tiny Shakespeare...")
        urllib.request.urlretrieve(SHAKES_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def build_vocab(text: str):
    vocab = sorted(list(set(text)))
    stoi = {ch:i for i,ch in enumerate(vocab)}
    itos = {i:ch for ch,i in stoi.items()}
    return vocab, stoi, itos

def encode(text: str, stoi: dict) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)

def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(0, data.size(0) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)

# ---------------------------------
# Core layers
# ---------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return self.scale * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

class CausalDWConv1d(nn.Module):
    """Depthwise 1D conv with left-only padding (causal)."""
    def __init__(self, channels: int, kernel_size: int, dilation: int = 1, bias: bool = True):
        super().__init__()
        self.ks = kernel_size
        self.dil = dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation,
                              groups=channels, bias=bias)
    def forward(self, x):
        # (B,T,C) -> (B,C,T)
        x = x.transpose(1,2)
        pad_left = self.dil * (self.ks - 1)
        x = F.pad(x, (pad_left, 0))
        y = self.conv(x)
        return y.transpose(1,2)  # (B,T,C)

class SEGate(nn.Module):
    """Konio-like modulatory gate:
       - Channel SE (global over time)
       - Token-wise mixing heads (e.g., M vs P; Dorsal vs Ventral)"""
    def __init__(self, channels: int, hidden: Optional[int] = None, num_mix_heads: int = 2):
        super().__init__()
        if hidden is None:
            hidden = max(64, channels // 8)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.mix = nn.Linear(channels, num_mix_heads)  # token-wise logits
    def forward(self, x):
        # x: (B,T,C)
        g = x.mean(dim=1)               # (B,C)
        g = F.gelu(self.fc1(g))
        g = torch.sigmoid(self.fc2(g))  # (B,C) in [0,1]
        logits = self.mix(x)            # (B,T,num_mix_heads)
        weights = F.softmax(logits, dim=-1)
        return g.unsqueeze(1), weights   # (B,1,C), (B,T,num_mix_heads)

# ---------------------------------
# MPK LGN stage
# ---------------------------------
class LGN_MPK(nn.Module):
    """Text analog of an LGN split into M/P/K feature maps from a shared embedding.
       - M: long context via dilated DW conv + PW MLP
       - P: local context via short DW conv + PW MLP
       - K: modulatory gate over channels + mix(M,P)
    """
    def __init__(self, channels: int, k_long: int = 9, k_short: int = 3, dil_long: int = 2, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm = RMSNorm(channels)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Magno (M)
        self.m_dw  = CausalDWConv1d(channels, kernel_size=k_long, dilation=dil_long)
        self.m_pw1 = nn.Linear(channels, channels * expand)
        self.m_pw2 = nn.Linear(channels * expand, channels)

        # Parvo (P)
        self.p_dw  = CausalDWConv1d(channels, kernel_size=k_short, dilation=1)
        self.p_pw1 = nn.Linear(channels, channels * expand)
        self.p_pw2 = nn.Linear(channels * expand, channels)

        # Konio (K): mix M/P + channel gating
        self.k = SEGate(channels, num_mix_heads=2)

        # Output projection
        self.out = nn.Linear(channels, channels)

    def forward(self, x):
        # x: (B,T,C)
        h = self.norm(x)

        m = self.m_dw(h)
        m = F.gelu(self.m_pw1(m)); m = self.drop(self.m_pw2(m))

        p = self.p_dw(h)
        p = F.gelu(self.p_pw1(p)); p = self.drop(self.p_pw2(p))

        ch_scale, mix_mp = self.k(h)  # mix_mp: (B,T,2) -> [M, P]
        m = m * ch_scale
        p = p * ch_scale
        fused = mix_mp[...,0:1] * m + mix_mp[...,1:2] * p

        return self.out(fused), (m, p, ch_scale, mix_mp)

# ---------------------------------
# Dorsal/Ventral cortical stage
# ---------------------------------
class DorsalBlock(nn.Module):
    """Dorsal favors M (sequence/temporal pattern)."""
    def __init__(self, channels: int, k_long: int = 9, dil: int = 2, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm = RMSNorm(channels)
        self.dw   = CausalDWConv1d(channels, kernel_size=k_long, dilation=dil)
        self.pw1  = nn.Linear(channels, channels * expand)
        self.pw2  = nn.Linear(channels * expand, channels)
        self.out  = nn.Linear(channels, channels)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    def forward(self, x):
        h = self.norm(x)
        y = self.dw(h)
        y = F.gelu(self.pw1(y))
        y = self.drop(self.pw2(y))
        return x + self.out(y)

class VentralBlock(nn.Module):
    """Ventral favors P (content/features)."""
    def __init__(self, channels: int, k_short: int = 3, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm = RMSNorm(channels)
        self.dw   = CausalDWConv1d(channels, kernel_size=k_short, dilation=1)
        self.pw1  = nn.Linear(channels, channels * expand)
        self.pw2  = nn.Linear(channels * expand, channels)
        self.out  = nn.Linear(channels, channels)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    def forward(self, x):
        h = self.norm(x)
        y = self.dw(h)
        y = F.gelu(self.pw1(y))
        y = self.drop(self.pw2(y))
        return x + self.out(y)

class DV_Fusion(nn.Module):
    """Fuse Dorsal and Ventral with Konio gating (per-token mix) + residual."""
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.k = SEGate(channels, num_mix_heads=2)  # [D, V]
        self.out = nn.Linear(channels, channels)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    def forward(self, d, v):
        # d, v: (B,T,C)
        base = (d + v) / 2
        ch_scale, mix_dv = self.k(base)  # (B,1,C), (B,T,2)
        d = d * ch_scale
        v = v * ch_scale
        fused = mix_dv[...,0:1] * d + mix_dv[...,1:2] * v
        return base + self.out(self.drop(fused))

# ---------------------------------
# Full stacked DVX MPK model
# ---------------------------------
class DVXTextMPK(nn.Module):
    """
    Text LM with MPK LGN and Dorsal/Ventral cortical stacks.
    Pipeline per layer:
      1) LGN_MPK: split & modulate -> fused LGN features
      2) DorsalBlock on (LGN + prev_dorsal)
      3) VentralBlock on (LGN + prev_ventral)
      4) DV_Fusion to mix D/V -> residual to backbone
    """
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        channels: int = 512,
        layers: int = 6,
        k_long: int = 9,
        k_short: int = 3,
        dil_base: int = 2,
        expand: int = 2,
        dropout: float = 0.0,
        tie_embeddings: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels

        self.tok = nn.Embedding(vocab_size, channels)
        self.pos = nn.Embedding(seq_len, channels)

        self.lgn_layers = nn.ModuleList([
            LGN_MPK(channels, k_long=k_long, k_short=k_short, dil_long=(dil_base + (i % 3)), expand=expand, dropout=dropout)
            for i in range(layers)
        ])
        self.dorsal_layers = nn.ModuleList([
            DorsalBlock(channels, k_long=k_long, dil=(dil_base + (i % 3)), expand=expand, dropout=dropout)
            for i in range(layers)
        ])
        self.ventral_layers = nn.ModuleList([
            VentralBlock(channels, k_short=k_short, expand=expand, dropout=dropout)
            for _ in range(layers)
        ])
        self.fuse_layers = nn.ModuleList([
            DV_Fusion(channels, dropout=dropout) for _ in range(layers)
        ])

        self.norm = RMSNorm(channels)
        self.head = nn.Linear(channels, vocab_size, bias=False)

        if tie_embeddings:
            self.head.weight = self.tok.weight

        # init
        nn.init.normal_(self.tok.weight, std=0.02)
        if not tie_embeddings:
            nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, idx: torch.Tensor):
        B, T = idx.shape
        if T > self.seq_len:
            raise ValueError(f"Input length {T} exceeds configured seq_len {self.seq_len}.")

        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok(idx) + self.pos(pos)   # (B,T,C)

        d = x  # dorsal state
        v = x  # ventral state

        for lgn, dblk, vblk, fuse in zip(self.lgn_layers, self.dorsal_layers, self.ventral_layers, self.fuse_layers):
            lgn_out, (m, p, ch_scale, mix_mp) = lgn(x)
            d = dblk(d + lgn_out)   # dorsal receives lgn + residual
            v = vblk(v + lgn_out)   # ventral receives lgn + residual
            x = fuse(d, v)          # fuse into backbone token features

        x = self.norm(x)
        logits = self.head(x)
        return logits

# ---------------------------------
# Sampling (optional)
# ---------------------------------
@torch.no_grad()
def generate(model: nn.Module, idx: torch.Tensor, steps: int, temperature: float = 1.0, top_k: Optional[int] = None):
    model.eval()
    for _ in range(steps):
        if idx.size(1) > model.seq_len:
            idx = idx[:, -model.seq_len:]
        logits = model(idx)[:, -1, :] / max(1e-8, temperature)
        if top_k is not None:
            v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
            thresh = v[:, -1].unsqueeze(-1)
            logits = torch.where(logits < thresh, torch.full_like(logits, -1e10), logits)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx

# ---------------------------------
# Train CLI
# ---------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--channels", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--k_long", type=int, default=9)
    ap.add_argument("--k_short", type=int, default=3)
    ap.add_argument("--dil_base", type=int, default=2)
    ap.add_argument("--expand", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--save", type=str, default="dvx_mpk_dv_text.pt")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    print(f"Device: {device}")

    # Data
    text = ensure_shakespeare("input.txt")
    vocab, stoi, itos = build_vocab(text)
    data = encode(text, stoi)
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size} | Data chars: {len(text):,}")

    # Model
    model = DVXTextMPK(
        vocab_size=vocab_size,
        seq_len=args.seq_len,
        channels=args.channels,
        layers=args.layers,
        k_long=args.k_long,
        k_short=args.k_short,
        dil_base=args.dil_base,
        expand=args.expand,
        dropout=args.dropout,
        tie_embeddings=False,
    ).to(device)

    # Optim
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Warmup
    model.train()
    for _ in range(args.warmup):
        xb, yb = get_batch(data, args.batch_size, args.seq_len, device)
        logits = model(xb)
        loss = loss_fn(logits.view(-1, vocab_size), yb.view(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    # Timed loop
    tokens = 0
    start = time.time()
    log_every = max(1, args.steps // 20)
    for step in range(1, args.steps + 1):
        xb, yb = get_batch(data, args.batch_size, args.seq_len, device)
        logits = model(xb)
        loss = loss_fn(logits.view(-1, vocab_size), yb.view(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tokens += xb.numel()

        if step % log_every == 0 or step == 1:
            tps = int(tokens / max(1e-8, (time.time() - start)))
            print(f"step {step:4d}/{args.steps} | loss {loss.item():.3f} | tokens/s ~ {tps:,}")

    # Save
    ckpt = {
        "model": model.state_dict(),
        "vocab": ''.join(vocab),
        "seq_len": args.seq_len,
        "channels": args.channels,
        "layers": args.layers,
        "stoi": {k:int(v) for k,v in stoi.items()},
        "itos": {int(k):v for k,v in itos.items()},
    }
    torch.save(ckpt, args.save)
    print(f"Saved checkpoint: {args.save}")

    # Sample
    with torch.no_grad():
        seed_char = random.choice(list(stoi.keys()))
        idx0 = torch.tensor([[stoi[seed_char]]], dtype=torch.long, device=device)
        out = generate(model, idx0, steps=args.sample, temperature=args.temperature, top_k=args.top_k)

    itos_map = {i:c for i,c in enumerate(vocab)}
    sample_text = ''.join(itos_map[int(i)] for i in out[0].tolist())
    print("\n=== SAMPLE ===")
    print(sample_text)
    print("==============")

if __name__ == "__main__":
    main()
