#!/usr/bin/env python3
# dvx_text_bench_local_cached.py
# DVX (Magno/Parvo/Konio + Dorsal/Ventral) with:
# - local loaders for enwik8/text8
# - Bits-per-Character eval
# - streaming generation CACHE for fast autoregressive decoding
# - device picker with ThinkPad/CUDA + AMP support

import argparse, math, os, time, random, gzip, zipfile, urllib.request
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== Dataset I/O (local, cached on disk) =====================
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "dvx_datasets")
os.makedirs(CACHE_DIR, exist_ok=True)

def _download(url: str, out_path: str):
    if not os.path.exists(out_path):
        print(f"Downloading {url} -> {out_path}")
        urllib.request.urlretrieve(url, out_path)

def load_enwik8_local():
    zip_path = os.path.join(CACHE_DIR, "enwik8.zip")
    raw_path = os.path.join(CACHE_DIR, "enwik8")
    if not os.path.exists(raw_path):
        _download("http://mattmahoney.net/dc/enwik8.zip", zip_path)
        print("Unzipping enwik8.zip...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extract("enwik8", path=CACHE_DIR)
    with open(raw_path, "rb") as f:
        raw = f.read()
    data = torch.tensor(list(raw), dtype=torch.long)
    train = data[:90_000_000]
    valid = data[90_000_000:95_000_000]
    test  = data[95_000_000:100_000_000]
    return train, valid, test, 256

def load_text8_local():
    raw_path = os.path.join(CACHE_DIR, "text8")
    if not os.path.exists(raw_path):
        zip_path = os.path.join(CACHE_DIR, "text8.zip")
        gz_path  = os.path.join(CACHE_DIR, "text8.gz")
        try:
            _download("http://mattmahoney.net/dc/text8.zip", zip_path)
            print("Unzipping text8.zip...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extract("text8", path=CACHE_DIR)
        except zipfile.BadZipFile:
            _download("http://mattmahoney.net/dc/text8.gz", gz_path)
            print("Decompressing text8.gz...")
            with gzip.open(gz_path, "rb") as gzf, open(raw_path, "wb") as out:
                out.write(gzf.read())
    with open(raw_path, "rt", encoding="utf-8") as f:
        txt = f.read()
    charset = list("abcdefghijklmnopqrstuvwxyz ")
    stoi = {c:i for i,c in enumerate(charset)}
    ids = torch.tensor([stoi.get(ch, 26) for ch in txt], dtype=torch.long)
    train = ids[:90_000_000]
    valid = ids[90_000_000:95_000_000]
    test  = ids[95_000_000:100_000_000]
    return train, valid, test, 27

# ===================== Helpers =====================
def pick_device(name: Optional[str] = None):
    if name:
        if name == "cuda" and torch.cuda.is_available(): return torch.device("cuda")
        if name == "mps"  and torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def bpc_from_loss(loss_nats: float) -> float:
    return loss_nats / math.log(2.0)

def chunk_ids(buf: torch.Tensor, seq_len: int):
    L = (buf.numel() - 1) // seq_len
    buf = buf[: L * seq_len + 1]
    x = buf[:-1].view(L, seq_len)
    y = buf[1: ].view(L, seq_len)
    return x, y

def get_batch(stream_x: torch.Tensor, stream_y: torch.Tensor, batch_size: int, device):
    N = stream_x.size(0)
    idx = torch.randint(0, N, (batch_size,))
    return stream_x[idx].to(device), stream_y[idx].to(device)

# ===================== Core layers (with step() cache) =====================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return self.scale * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

class CausalDWConv1d(nn.Module):
    """
    Depthwise 1D conv (causal). Provides:
      - forward(x): full sequence
      - step(x_t, state): single-step with streaming cache
    state shape: (B, hist, C), hist = dilation*(kernel_size-1)
    """
    def __init__(self, channels: int, kernel_size: int, dilation: int = 1, bias: bool = True):
        super().__init__()
        self.ks = kernel_size
        self.dil = dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                              dilation=dilation, groups=channels, bias=bias)

    def cache_len(self) -> int:
        return self.dil * (self.ks - 1)

    def forward(self, x):
        # x: (B,T,C)
        x = x.transpose(1, 2)  # (B,C,T)
        x = F.pad(x, (self.cache_len(), 0))
        y = self.conv(x)
        return y.transpose(1, 2)  # (B,T,C)

    def step(self, x_t: torch.Tensor, state: Optional[torch.Tensor]):
        # x_t: (B,1,C); state: (B, hist, C) or None
        B, one, C = x_t.shape
        hist = self.cache_len()
        if hist == 0:
            x_seq = x_t.transpose(1,2)          # (B,C,1)
        else:
            if state is None:
                state = torch.zeros(B, hist, C, device=x_t.device, dtype=x_t.dtype)
            x_seq = torch.cat([state, x_t], dim=1).transpose(1,2)  # (B,C,hist+1)
        y = self.conv(x_seq)                    # (B,C,1)
        y = y.transpose(1,2)                    # (B,1,C)
        # update state (drop oldest, append x_t)
        if hist > 0:
            new_state = torch.cat([state[:, 1:], x_t], dim=1)
        else:
            new_state = None
        return y, new_state

class SEGate(nn.Module):
    def __init__(self, channels: int, hidden: Optional[int] = None, num_mix_heads: int = 2):
        super().__init__()
        if hidden is None:
            hidden = max(64, channels // 8)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.mix = nn.Linear(channels, num_mix_heads)
    def forward(self, x):
        g = x.mean(dim=1)
        g = F.gelu(self.fc1(g))
        g = torch.sigmoid(self.fc2(g))      # (B,C)
        logits = self.mix(x)                # (B,T,H)
        weights = F.softmax(logits, dim=-1) # (B,T,H)
        return g.unsqueeze(1), weights

# ===================== MPK + D/V blocks (cache-aware step) =====================
class LGN_MPK(nn.Module):
    def __init__(self, channels: int, k_long: int = 9, k_short: int = 3, dil_long: int = 2, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm = RMSNorm(channels)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.m_dw  = CausalDWConv1d(channels, kernel_size=k_long, dilation=dil_long)
        self.m_pw1 = nn.Linear(channels, channels * expand)
        self.m_pw2 = nn.Linear(channels * expand, channels)
        self.p_dw  = CausalDWConv1d(channels, kernel_size=k_short, dilation=1)
        self.p_pw1 = nn.Linear(channels, channels * expand)
        self.p_pw2 = nn.Linear(channels * expand, channels)
        self.k = SEGate(channels, num_mix_heads=2)
        self.out = nn.Linear(channels, channels)

    def forward(self, x):
        h = self.norm(x)
        m = self.m_dw(h);  m = F.gelu(self.m_pw1(m));  m = self.drop(self.m_pw2(m))
        p = self.p_dw(h);  p = F.gelu(self.p_pw1(p));  p = self.drop(self.pw2(p))  # fixed below (typo)
        return NotImplemented

    # fixed full forward (typo fix)
    def forward(self, x):  # noqa
        h = self.norm(x)
        m = self.m_dw(h);  m = F.gelu(self.m_pw1(m));  m = self.drop(self.m_pw2(m))
        p = self.p_dw(h);  p = F.gelu(self.p_pw1(p));  p = self.drop(self.p_pw2(p))
        ch_scale, mix = self.k(h)
        fused = mix[...,0:1]*m*ch_scale + mix[...,1:2]*p*ch_scale
        return self.out(fused)

    # one-token step with cache dict {"m":state, "p":state}
    def step(self, x_t: torch.Tensor, cache: Optional[dict]):
        # x_t: (B,1,C)
        h_t = self.norm(x_t)                 # per-token rmsnorm is fine
        m_y, m_state = self.m_dw.step(h_t, None if cache is None else cache.get("m"))
        p_y, p_state = self.p_dw.step(h_t, None if cache is None else cache.get("p"))
        # PW MLPs
        m_y = F.gelu(self.m_pw1(m_y)); m_y = self.m_pw2(m_y)
        p_y = F.gelu(self.p_pw1(p_y)); p_y = self.p_pw2(p_y)
        # K gates (per-token)
        # for K we need a token-wise mix; use single-token head
        g = h_t.mean(dim=1)                       # (B,C)
        g = F.gelu(self.k.fc1(g))
        g = torch.sigmoid(self.k.fc2(g)).unsqueeze(1)  # (B,1,C)
        mix = F.softmax(self.k.mix(h_t), dim=-1)       # (B,1,2)
        fused = mix[...,0:1]*m_y*g + mix[...,1:2]*p_y*g
        y_t = self.out(fused)                     # (B,1,C)
        new_cache = {"m": m_state, "p": p_state}
        return y_t, new_cache

class DorsalBlock(nn.Module):
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
        y = self.dw(h); y = F.gelu(self.pw1(y)); y = self.drop(self.pw2(y))
        return x + self.out(y)
    def step(self, x_t, cache):
        h_t = self.norm(x_t)
        y_t, st = self.dw.step(h_t, cache)
        y_t = F.gelu(self.pw1(y_t)); y_t = self.pw2(y_t)
        return x_t + self.out(y_t), st

class VentralBlock(nn.Module):
    def __init__(self, channels: int, k_short: int = 3, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm = RMSNorm(channels)
        self.dw   = CausalDWConv1d(channels, kernel_size=k_short, dilation=1)
        self.pw1  = nn.Linear(channels, channels * expand)
        self.pw2  = nn.Linear(channels * expand, channels)  # <- corrected
        self.out  = nn.Linear(channels, channels)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    def forward(self, x):
        h = self.norm(x)
        y = self.dw(h); y = F.gelu(self.pw1(y)); y = self.drop(self.pw2(y))
        return x + self.out(y)
    def step(self, x_t, cache):
        h_t = self.norm(x_t)
        y_t, st = self.dw.step(h_t, cache)
        y_t = F.gelu(self.pw1(y_t)); y_t = self.pw2(y_t)
        return x_t + self.out(y_t), st

class DV_Fusion(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.k = SEGate(channels, num_mix_heads=2)
        self.out = nn.Linear(channels, channels)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    def forward(self, d, v):
        base = (d + v) / 2
        ch_scale, mix = self.k(base)
        fused = mix[...,0:1]*d*ch_scale + mix[...,1:2]*v*ch_scale
        return base + self.out(self.drop(fused))
    def step(self, d_t, v_t):
        base = (d_t + v_t) / 2
        # K on single token
        g = base.mean(dim=1)
        g = F.gelu(self.k.fc1(g))
        g = torch.sigmoid(self.k.fc2(g)).unsqueeze(1)
        mix = F.softmax(self.k.mix(base), dim=-1)
        fused = mix[...,0:1]*d_t*g + mix[...,1:2]*v_t*g
        return base + self.out(self.drop(fused))

# ===================== DVX model with streaming cache =====================
class DVXTextMPK(nn.Module):
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
        self.fuse_layers = nn.ModuleList([DV_Fusion(channels, dropout=dropout) for _ in range(layers)])

        self.norm = RMSNorm(channels)
        self.head = nn.Linear(channels, vocab_size, bias=False)

        if tie_embeddings:
            self.head.weight = self.tok.weight

        nn.init.normal_(self.tok.weight, std=0.02)
        if not tie_embeddings:
            nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, idx: torch.Tensor):
        B, T = idx.shape
        if T > self.seq_len:
            raise ValueError(f"Input length {T} exceeds configured seq_len {self.seq_len}.")
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok(idx) + self.pos(pos)
        d, v = x, x
        for lgn, dblk, vblk, fuse in zip(self.lgn_layers, self.dorsal_layers, self.ventral_layers, self.fuse_layers):
            lgn_out = lgn(x)
            d = dblk(d + lgn_out)
            v = vblk(v + lgn_out)
            x = fuse(d, v)
        x = self.norm(x)
        return self.head(x)

    # -------- streaming generation with cache --------
    def init_cache(self, batch_size: int, device) -> List[dict]:
        caches = []
        for lgn, dblk, vblk in zip(self.lgn_layers, self.dorsal_layers, self.ventral_layers):
            caches.append({
                "lgn_m": None, "lgn_p": None,
                "d": None, "v": None,
            })
        return caches

    @torch.no_grad()
    def generate_stream(self, seed_ids: torch.Tensor, steps: int, temperature: float = 1.0, top_k: Optional[int] = None):
        """
        seed_ids: (B,T0) int64
        Returns: (B,T0+steps)
        """
        self.eval()
        device = seed_ids.device
        B, T0 = seed_ids.shape

        # prime the model on the seed (no cache yet, do full forward)
        logits = self.forward(seed_ids)
        last_token = seed_ids[:, -1:]

        # init caches
        caches = []
        for lgn, dblk, vblk in zip(self.lgn_layers, self.dorsal_layers, self.ventral_layers):
            caches.append({
                "lgn": {"m": None, "p": None},
                "d":  None,
                "v":  None,
            })

        out_ids = [seed_ids]

        for _ in range(steps):
            # do one streaming step using caches
            # embed + last position
            pos_id = torch.tensor([[0]], device=device)  # position not used in step; keep absolute 0 to avoid OOB
            x_t = self.tok(last_token) + self.pos(pos_id.expand(B,1))
            d_t, v_t = x_t, x_t

            for i, (lgn, dblk, vblk, fuse) in enumerate(zip(self.lgn_layers, self.dorsal_layers, self.ventral_layers, self.fuse_layers)):
                y_t, new_lgn = lgn.step(x_t, caches[i]["lgn"])
                d_t, new_d = dblk.step(d_t + y_t, caches[i]["d"])
                v_t, new_v = vblk.step(v_t + y_t, caches[i]["v"])
                x_t = fuse.step(d_t, v_t)
                caches[i]["lgn"] = new_lgn
                caches[i]["d"]   = new_d
                caches[i]["v"]   = new_v

            x_t = self.norm(x_t)
            logits_t = self.head(x_t)[:, -1, :] / max(1e-8, temperature)

            if top_k is not None:
                v, _ = torch.topk(logits_t, k=min(top_k, logits_t.size(-1)))
                logits_t = torch.where(logits_t < v[..., -1:].expand_as(logits_t),
                                       torch.full_like(logits_t, -1e10), logits_t)
            probs = F.softmax(logits_t, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            out_ids.append(next_id)
            last_token = next_id

        return torch.cat(out_ids, dim=1)

# ===================== Eval =====================
@torch.no_grad()
def eval_bpc(model, x_stream, y_stream, batch_size, device, amp=False):
    model.eval()
    total_loss, total_tok = 0.0, 0
    scaler_ctx = torch.cuda.amp.autocast if (amp and device.type == "cuda") else torch.cpu.amp.autocast
    # torch.cpu.amp.autocast is safe as a no-op on non-CPU types in PyTorch 2.8
    for i in range(0, x_stream.size(0), batch_size):
        xb = x_stream[i:i+batch_size].to(device)
        yb = y_stream[i:i+batch_size].to(device)
        with scaler_ctx():
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), reduction="sum")
        total_loss += loss.item()
        total_tok += yb.numel()
    return bpc_from_loss(total_loss / total_tok)

# ===================== Main =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["enwik8","text8"], required=True)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--channels", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--k_long", type=int, default=9)
    ap.add_argument("--k_short", type=int, default=3)
    ap.add_argument("--dil_base", type=int, default=2)
    ap.add_argument("--expand", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--save", type=str, default="dvx_text_cached.pt")
    ap.add_argument("--device", type=str, choices=["cuda","mps","cpu"], default=None)
    ap.add_argument("--amp", action="store_true", help="use AMP on CUDA")
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"Device: {device} | Dataset: {args.dataset} | batch_size {args.batch_size} | AMP {args.amp}")

    # Data
    if args.dataset == "enwik8":
        train_ids, valid_ids, test_ids, vocab_size = load_enwik8_local()
    else:
        train_ids, valid_ids, test_ids, vocab_size = load_text8_local()

    tx, ty = chunk_ids(train_ids, args.seq_len)
    vx, vy = chunk_ids(valid_ids, args.seq_len)
    qx, qy = chunk_ids(test_ids,  args.seq_len)

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
        tie_embeddings=True,  # small quality bump
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type=="cuda"))

    # Warmup
    model.train()
    for _ in range(args.warmup):
        xb, yb = get_batch(tx, ty, args.batch_size, device)
        with torch.cuda.amp.autocast(enabled=(args.amp and device.type=="cuda")):
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()

    # Train loop
    tokens = 0
    start = time.time()
    log_every = max(1, args.steps // 50)
    for step in range(1, args.steps + 1):
        xb, yb = get_batch(tx, ty, args.batch_size, device)
        with torch.cuda.amp.autocast(enabled=(args.amp and device.type=="cuda")):
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        tokens += xb.numel()

        if step % log_every == 0 or step == 1:
            elapsed = max(1e-8, time.time() - start)
            tps = int(tokens / elapsed)
            with torch.no_grad():
                vxb, vyb = get_batch(vx, vy, min(8, args.batch_size), device)
                v_logits = model(vxb)
                v_loss = F.cross_entropy(v_logits.view(-1, v_logits.size(-1)), vyb.view(-1))
                v_bpc = bpc_from_loss(v_loss.item())
            print(f"step {step:5d}/{args.steps} | train CE {loss.item():.3f} | val BPC {v_bpc:.3f} | tok/s ~ {tps:,}")

    # Save
    ckpt = {
        "model": model.state_dict(),
        "vocab_size": vocab_size,
        "seq_len": args.seq_len,
        "cfg": vars(args),
    }
    torch.save(ckpt, args.save)
    print(f"Saved checkpoint: {args.save}")
    print("Params (M):", sum(p.numel() for p in model.parameters())/1e6)

    # Final eval
    val_bpc = eval_bpc(model, vx, vy, batch_size=args.batch_size, device=device, amp=(args.amp and device.type=="cuda"))
    test_bpc = eval_bpc(model, qx, qy, batch_size=args.batch_size, device=device, amp=(args.amp and device.type=="cuda"))
    print(f"\nFinal: val BPC = {val_bpc:.3f} | test BPC = {test_bpc:.3f}")

if __name__ == "__main__":
    main()
