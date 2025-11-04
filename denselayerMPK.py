#!/usr/bin/env python3
import os, platform

# macOS safety first: spawn, headless plotting, fewer threads
if platform.system() == "Darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

"""
MPK → Linear Tower for CIFAR-100 with TensorBoard logging
Includes "hum" analytics: DFA fractal metric, recurrence plot, spectrum, and π-lock scores.
Swap StubMPK with your real MPKNet that returns a dict {'mag','par','kon'}.
"""
import math
import random
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as T
from torchvision.datasets import CIFAR100

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

# -----------------------------
# MPK Front End (stub)
# -----------------------------
class StubMPK(nn.Module):
    def __init__(self, in_ch=3, mag_ch=64, par_ch=64, kon_ch=32):
        super().__init__()
        self.mag = nn.Sequential(
            nn.Conv2d(in_ch, mag_ch, 7, padding=3, bias=False),
            nn.BatchNorm2d(mag_ch),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
        )
        self.par = nn.Sequential(
            nn.Conv2d(in_ch, par_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(par_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(par_ch, par_ch, 3, padding=1, groups=par_ch, bias=False),
            nn.ReLU(inplace=True),
        )
        self.kon = nn.Sequential(
            nn.Conv2d(in_ch, kon_ch, 1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return {"mag": self.mag(x), "par": self.par(x), "kon": self.kon(x)}

# -----------------------------
# Helpers
# -----------------------------
class GAP(nn.Module):
    def forward(self, x):
        if x.ndim == 4:
            return F.adaptive_avg_pool2d(x, 1).flatten(1)
        elif x.ndim == 2:
            return x
        else:
            raise ValueError("Expected [B,C,H,W] or [B,C]")

class MPKLinearTower(nn.Module):
    def __init__(
        self,
        mpk_frontend: nn.Module,
        mag_dim: int,
        par_dim: int,
        kon_dim: int,
        hidden_dims=(512, 512, 512),
        num_classes=100,
        k_injection: str = "concat",  # "concat" or "film"
        dropout: float = 0.0,
    ):
        super().__init__()
        assert len(hidden_dims) == 3, "Use exactly 3 dense layers"
        self.mpk = mpk_frontend
        self.pool = GAP()
        base_in = mag_dim + par_dim

        self.fc1 = nn.Linear(base_in, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.act = nn.GELU()
        self.do = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.k_injection = k_injection
        if k_injection == "concat":
            self.k_proj = nn.Linear(kon_dim, hidden_dims[2] // 2)
            clf_in = hidden_dims[2] + hidden_dims[2] // 2
            self.classifier = nn.Linear(clf_in, num_classes)
        elif k_injection == "film":
            mod_dim = hidden_dims[2]
            self.k_to_gamma = nn.Linear(kon_dim, mod_dim)
            self.k_to_beta = nn.Linear(kon_dim, mod_dim)
            self.classifier = nn.Linear(mod_dim, num_classes)
        else:
            raise ValueError("k_injection must be 'concat' or 'film'")

        self._last_shapes = {}

    def forward(self, x):
        s = self.mpk(x)
        mag = self.pool(s["mag"])
        par = self.pool(s["par"])
        kon = self.pool(s["kon"])

        base = torch.cat([mag, par], dim=1)
        h1 = self.do(self.act(self.fc1(base)))
        h2 = self.do(self.act(self.fc2(h1)))
        h3 = self.act(self.fc3(h2))

        if self.k_injection == "concat":
            kvec = self.act(self.k_proj(kon))
            joint = torch.cat([h3, kvec], dim=1)
            logits = self.classifier(joint)
        else:  # FiLM
            gamma = self.k_to_gamma(kon)
            beta = self.k_to_beta(kon)
            h3_mod = h3 * (1 + gamma) + beta
            logits = self.classifier(self.act(h3_mod))

        self._last_shapes = {
            "mag": tuple(mag.shape),
            "par": tuple(par.shape),
            "kon": tuple(kon.shape),
            "h1": tuple(h1.shape),
            "h2": tuple(h2.shape),
            "h3": tuple(h3.shape),
            "logits": tuple(logits.shape),
        }
        return logits

    @torch.no_grad()
    def feature_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return self._last_shapes

# -----------------------------
# Hum / Fractal analytics
# -----------------------------
def _windowed_linear_detrend(x: np.ndarray, s: int) -> float:
    n = len(x)
    if s < 2 or s > n:
        return np.nan
    y = np.cumsum(x - np.mean(x))
    m = n // s
    if m < 2:
        return np.nan
    y = y[: m * s].reshape(m, s)
    t = np.arange(s)
    F2 = []
    for seg in y:
        A = np.vstack([t, np.ones_like(t)]).T
        a, b = np.linalg.lstsq(A, seg, rcond=None)[0]
        trend = a * t + b
        res = seg - trend
        F2.append(np.mean(res**2))
    return np.sqrt(np.mean(F2))

def dfa_alpha(x: np.ndarray, s_min: int = 4, s_max: int = None) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    n = len(x)
    if s_max is None:
        s_max = max(8, n // 4)
    scales = np.unique(np.floor(np.geomspace(s_min, s_max, num=12)).astype(int))
    Fs = [_windowed_linear_detrend(x, s) for s in scales]
    mask = np.isfinite(Fs) & (np.array(Fs) > 0)
    scales = scales[mask]
    Fs = np.array(Fs)[mask]
    if len(Fs) < 2:
        return float("nan"), float("nan")
    lx = np.log(scales)
    ly = np.log(Fs)
    a = np.polyfit(lx, ly, 1)[0]
    D = 2.0 - a
    return float(a), float(D)

def recurrence_plot(
    x: np.ndarray, m: int = 2, tau: int = 1, eps_percentile: float = 10.0, max_points: int = 200
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) > max_points:
        idx = np.linspace(0, len(x) - 1, max_points).astype(int)
        x = x[idx]
    L = len(x) - (m - 1) * tau
    if L <= 2:
        return np.zeros((2, 2), dtype=np.uint8)
    Y = np.stack([x[i : i + L] for i in range(0, m * tau, tau)], axis=1)
    d = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2))
    eps = np.percentile(d, eps_percentile)
    R = (d <= eps).astype(np.uint8)
    return R

def spectrum_metrics(x: np.ndarray, kmax: int = 64) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    n = len(x)
    if n < 8:
        return {k: float("nan") for k in ["f_dom", "T_dom", "omega_dom", "hum_index", "pi_lock_period", "pi_lock_omega"]}
    spec = np.fft.rfft(x)
    amps = np.abs(spec)
    freqs = np.fft.rfftfreq(n, d=1.0)
    amps[0] = 0.0
    if amps.max() <= 0:
        return {k: float("nan") for k in ["f_dom", "T_dom", "omega_dom", "hum_index", "pi_lock_period", "pi_lock_omega"]}
    idx = int(np.argmax(amps))
    f_dom = freqs[idx]
    T_dom = (1.0 / f_dom) if f_dom > 0 else float("inf")
    omega_dom = 2 * math.pi * f_dom
    topk = np.sort(amps)[-5:]
    hum_index = float(topk[-1] / (topk.sum() + 1e-9))

    def _pi_lock(val: float) -> float:
        if not np.isfinite(val) or val <= 0:
            return float("nan")
        ks = np.arange(1, kmax + 1)
        d = np.min(np.abs(val - ks * math.pi))
        return float(max(0.0, 1.0 - d / (val + 1e-9)))

    pi_lock_period = _pi_lock(T_dom)
    pi_lock_omega = _pi_lock(omega_dom)
    return {
        "f_dom": float(f_dom),
        "T_dom": float(T_dom),
        "omega_dom": float(omega_dom),
        "hum_index": float(hum_index),
        "pi_lock_period": float(pi_lock_period),
        "pi_lock_omega": float(pi_lock_omega),
    }

# -----------------------------
# Data
# -----------------------------
def get_data(batch_size: int, workers: int):
    if platform.system() == "Darwin":
        workers = 0
    CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
    CIFAR100_STD = [0.2675, 0.2565, 0.2761]
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    train_ds = CIFAR100(root="./data", train=True, download=True, transform=train_tf)
    test_ds = CIFAR100(root="./data", train=False, download=True, transform=test_tf)

    # mac-safe loader settings
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=False, persistent_workers=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=False, persistent_workers=False
    )
    return train_loader, test_loader

# -----------------------------
# Logging helpers
# -----------------------------
def log_hum(writer: SummaryWriter, tag: str, series: List[float], global_step: int):
    arr = np.array(series, dtype=np.float64)
    if len(arr) < 16:
        return
    alpha, D = dfa_alpha(arr)
    writer.add_scalar(f"{tag}/dfa_alpha", alpha, global_step)
    writer.add_scalar(f"{tag}/fractal_dim", D, global_step)

    sp = spectrum_metrics(arr)
    for k, v in sp.items():
        writer.add_scalar(f"{tag}/{k}", v, global_step)

    R = recurrence_plot(arr, m=2, tau=1, eps_percentile=10.0, max_points=200)
    img = torch.from_numpy(R.astype(np.float32))[None, :, :]
    writer.add_image(f"{tag}/recurrence", img, global_step)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = arr - arr.mean()
        n = len(x)
        s_max = max(8, n // 4)
        scales = np.unique(np.floor(np.geomspace(4, s_max, num=12)).astype(int))
        Fs = [_windowed_linear_detrend(x, s) for s in scales]
        mask = np.isfinite(Fs) & (np.array(Fs) > 0)
        scales = scales[mask]; Fs = np.array(Fs)[mask]
        if len(Fs) >= 2:
            plt.figure()
            plt.loglog(scales, Fs, marker="o")
            plt.title("DFA fluctuation function")
            plt.xlabel("scale s"); plt.ylabel("F(s)")
            writer.add_figure(f"{tag}/dfa_curve", plt.gcf(), global_step)
            plt.close()
    except Exception:
        pass

# -----------------------------
# Train
# -----------------------------
def train(args):
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and not args.cpu:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    set_seed(args.seed)
    train_loader, test_loader = get_data(args.batch_size, args.workers)

    mag_ch, par_ch, kon_ch = args.mag_ch, args.par_ch, args.kon_ch
    mpk = StubMPK(in_ch=3, mag_ch=mag_ch, par_ch=par_ch, kon_ch=kon_ch)

    model = MPKLinearTower(
        mpk_frontend=mpk,
        mag_dim=mag_ch,
        par_dim=par_ch,
        kon_dim=kon_ch,
        hidden_dims=tuple(args.hidden_dims),
        num_classes=100,
        k_injection=args.k_injection,
        dropout=args.dropout,
    ).to(device)

    n_params = count_params(model)
    os.makedirs(args.logdir, exist_ok=True)
writer = None
if not args.no_tb:
    writer = SummaryWriter(log_dir=args.logdir)
    writer.add_text("config", str(vars(args)))

    writer.add_text("params", f"{n_params}", 0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)
    criterion = nn.CrossEntropyLoss()
    state = TrainState()

    # Warmup one pass to log feature shapes
    with torch.no_grad():
        xb, yb = next(iter(train_loader))
        xb = xb.to(device)
        _ = model(xb)
        shapes = model.feature_shapes()
        for k, shp in shapes.items():
            writer.add_text("feature_shapes", f"{k}: {shp}", 0)

    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = running_acc = running_grad = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=not args.no_amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    g = p.grad.detach().data.norm(2)
                    total_norm += g.item() ** 2
            total_norm = total_norm ** 0.5

            scaler.step(optimizer)
            scaler.update()

            acc = accuracy(logits, yb)
            running_loss += loss.item()
            running_acc += acc
            running_grad += total_norm

            state.loss_series.append(loss.item())
            state.acc_series.append(acc)
            state.grad_series.append(total_norm)

            if step % args.log_every == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/acc", acc, step)
                writer.add_scalar("train/grad_norm", total_norm, step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
            step += 1

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_acc / len(train_loader)
        epoch_grad = running_grad / len(train_loader)
        writer.add_scalar("epoch/train_loss", epoch_loss, epoch)
        writer.add_scalar("epoch/train_acc", epoch_acc, epoch)
        writer.add_scalar("epoch/train_grad", epoch_grad, epoch)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item()
                val_acc += accuracy(logits, yb)
        val_loss /= len(test_loader)
        val_acc /= len(test_loader)
        writer.add_scalar("epoch/val_loss", val_loss, epoch)
        writer.add_scalar("epoch/val_acc", val_acc, epoch)

        if args.hum_window > 0:
            for name, series in [("loss", state.loss_series), ("acc", state.acc_series), ("grad", state.grad_series)]:
                window = series[-args.hum_window:] if len(series) > args.hum_window else series
                log_hum(writer, f"hum/{name}", window, epoch)

        sched.step()
        print(f"Epoch {epoch:03d} | Train Acc {epoch_acc:.4f} | Val Acc {val_acc:.4f} | Train Loss {epoch_loss:.4f} | Val Loss {val_loss:.4f}")

    writer.add_hparams(
        {
            "lr": args.lr,
            "wd": args.wd,
            "hidden": str(tuple(args.hidden_dims)),
            "dropout": args.dropout,
            "k_inj": args.k_injection,
            "mag_ch": args.mag_ch,
            "par_ch": args.par_ch,
            "kon_ch": args.kon_ch,
        },
        {"hparams/val_acc": val_acc, "hparams/val_loss": val_loss},
    )
    writer.close()

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="MPK → 3xDense with K-injection on CIFAR-100 + hum metrics")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=0)  # safest on macOS
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--hidden-dims", type=int, nargs=3, default=[512, 512, 512])
    p.add_argument("--k-injection", type=str, default="concat", choices=["concat", "film"])
    p.add_argument("--mag-ch", type=int, default=64)
    p.add_argument("--par-ch", type=int, default=64)
    p.add_argument("--kon-ch", type=int, default=32)
    p.add_argument("--seed", type=int, default=111)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--logdir", type=str, default="runs/mpk_linear_cifar100")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--hum-window", type=int, default=1200, help="how many recent steps to analyze for hum metrics")
    p.add_argument("--no-tb", action="store_true")
    p.add_argument("--no-hum", action="store_true")

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
