#!/usr/bin/env python3
# mpk_twoeyes_v1binocular_tb.py
# Architecture:
#   Left eye:  CellPop(4x4) -> K_eye -> MPKExact(Px4, Mx2)
#   Right eye: CellPop(4x4) -> K_eye -> MPKExact(Px4, Mx2)
#   Shared V1 (binocular): disparity energy + rivalry oscillator + tiny tilewise disparity prior (EMA logits)
# Constraints: no backprop anywhere; only V1 has a bounded EMA prior (local counts); oscillations protected.
# CIFAR-100 with symmetric binocular offsets; TensorBoard logs: DFA alpha, |r|, disparity acc/MAE, confidence, throughput.
import os
# keep every native lib single-threaded (prevents mutex weirdness)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# MPS fallback is safe on Mac
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import os, time, math, random, argparse
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.datasets import CIFAR100

from torch.utils.tensorboard import SummaryWriter

# ----------------- utils -----------------
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

# ----------------- fixed conv helpers -----------------
def fixed_depthwise_conv(in_ch, k, stride, pad, ker2d):
    conv = nn.Conv2d(in_ch, in_ch, k, stride=stride, padding=pad, groups=in_ch, bias=False)
    with torch.no_grad():
        w = conv.weight
        ker = ker2d.clone().float() if isinstance(ker2d, torch.Tensor) else torch.tensor(ker2d, dtype=torch.float32)
        for c in range(in_ch):
            w[c,0,:,:] = ker
    for p in conv.parameters(): p.requires_grad=False
    return conv

def gaussian_kernel2d(size, sigma):
    ax = torch.arange(size) - (size-1)/2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    ker = torch.exp(-(xx**2+yy**2)/(2*sigma**2))
    ker = ker/ker.sum()
    return ker

def highpass_3x3():
    ker = torch.tensor([[0.,-1.,0.],[-1.,4.,-1.],[0.,-1.,0.]])
    ker = ker/ker.abs().sum().clamp_min(1e-6)
    return ker

# ----------------- CellPop (tile LN) -----------------
class CellPop(nn.Module):
    def __init__(self, grid=4, eps=1e-5):
        super().__init__(); self.grid=grid; self.eps=eps
    def forward(self,x):
        B,C,H,W = x.shape; g=self.grid
        gh,gw = max(1,H//g), max(1,W//g)
        Hc,Wc = gh*g, gw*g; x = x[:,:,:Hc,:Wc]
        xt = x.unfold(2,gh,gh).unfold(3,gw,gw)  # B,C,g,g,gh,gw
        mu = xt.mean(dim=(1,4,5), keepdim=True)
        sd = xt.std(dim=(1,4,5), keepdim=True) + self.eps
        xt = (xt - mu)/sd
        return xt.contiguous().view(B,C,Hc,Wc)

# ----------------- K_eye (per-eye Konio modulation) -----------------
class KEye(nn.Module):
    """
    Fixed, local modulatory gate per eye (no learning).
    Uses tilewise contrast to produce a gain map in [gmin,gmax].
    """
    def __init__(self, gmin=0.8, gmax=1.2, grid=4, eps=1e-6):
        super().__init__(); self.gmin=gmin; self.gmax=gmax; self.grid=grid; self.eps=eps
    def forward(self, x):
        B,C,H,W = x.shape
        L = x.mean(dim=1, keepdim=True)          # luminance-ish
        g=self.grid; gh,gw = max(1,H//g), max(1,W//g)
        Hc,Wc = gh*g, gw*g; Lc = L[:,:,:Hc,:Wc]
        Lt = Lc.unfold(2,gh,gh).unfold(3,gw,gw)  # B,1,g,g,gh,gw
        std = Lt.std(dim=(1,4,5), keepdim=True)  # B,1,g,g,1,1
        std = std.expand(-1,-1,-1,-1,gh,gw).contiguous().view(B,1,Hc,Wc)
        # normalize 0..1 per image
        s = std
        s = (s - s.amin(dim=(2,3), keepdim=True)) / (s.amax(dim=(2,3), keepdim=True) - s.amin(dim=(2,3), keepdim=True) + self.eps)
        gain = self.gmin + (self.gmax - self.gmin) * s
        G = torch.ones_like(L); G[:,:,:Hc,:Wc]=gain
        return x*G

# ----------------- MPKExact (per eye) -----------------
class MPKExact(nn.Module):
    # Parvo: 4 × (k=3, s=1, p=1); Magno: k=9,s=3 then k=7,s=2; depthwise; fixed
    def __init__(self, in_ch=3):
        super().__init__()
        hp3 = highpass_3x3()
        self.p1=fixed_depthwise_conv(in_ch,3,1,1,hp3)
        self.p2=fixed_depthwise_conv(in_ch,3,1,1,hp3)
        self.p3=fixed_depthwise_conv(in_ch,3,1,1,hp3)
        self.p4=fixed_depthwise_conv(in_ch,3,1,1,hp3)
        g9=gaussian_kernel2d(9,9/3.0); g7=gaussian_kernel2d(7,7/3.0)
        self.m1=fixed_depthwise_conv(in_ch,9,3,4,g9)
        self.m2=fixed_depthwise_conv(in_ch,7,2,3,g7)
        self.eps=1e-6
        for p in self.parameters(): p.requires_grad=False
    def _norm(self,x):
        mu=x.mean(dim=(2,3),keepdim=True); sd=x.std(dim=(2,3),keepdim=True)+self.eps
        return (x-mu)/sd
    def forward(self,x):
        with torch.no_grad():
            p=F.relu(self._norm(self.p1(x)))
            p=F.relu(self._norm(self.p2(p)))
            p=F.relu(self._norm(self.p3(p)))
            p=F.relu(self._norm(self.p4(p)))    # B,C,H,W
            m=F.relu(self._norm(self.m1(x)))    # B,C,H/3,W/3
            m=F.relu(self._norm(self.m2(m)))    # B,C,H/6,W/6
            m_up=F.interpolate(m, size=p.shape[-2:], mode='bilinear', align_corners=False)
            out=torch.cat([p,m_up],dim=1)       # [B,2C,H,W]
            return out, p, m

# ----------------- V1 (binocular) -----------------
def disparity_energy(left, right, dmax=4, kappa=3.0):
    Ds=list(range(-dmax,dmax+1)); energies=[]
    for d in Ds:
        r_shift=torch.roll(right, shifts=d, dims=3)  # horizontal only
        e=(left*r_shift).abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
        energies.append(e)
    E=torch.cat(energies,dim=1)             # [B,D,H,W]
    w=F.softmax(kappa*E,dim=1)
    conf=E.max(dim=1,keepdim=True).values/(E.sum(dim=1,keepdim=True)+1e-6)
    return E,w,conf.clamp(0,1), torch.tensor(Ds, device=left.device)

def dfa_alpha_1d(x_np, min_win=4, max_win=None, n_scales=6):
    x=np.asarray(x_np,dtype=np.float64); x=x-x.mean(); y=np.cumsum(x); L=len(y)
    if max_win is None: max_win=L//2
    wins=np.unique(np.round(np.logspace(np.log10(min_win), np.log10(max_win), n_scales)).astype(int))
    Fs=[]; Ss=[]
    for s in wins:
        if s<2: continue
        nseg=L//s
        if nseg<2: continue
        rms=[]
        for k in range(nseg):
            seg=y[k*s:(k+1)*s]; t=np.arange(s)
            A=np.vstack([t, np.ones(s)]).T
            a,b=np.linalg.lstsq(A, seg, rcond=None)[0]
            detr=seg-(a*t+b); rms.append(np.sqrt((detr**2).mean()))
        if rms: Fs.append(np.mean(rms)); Ss.append(s)
    if len(Ss)<2: return np.nan
    return float(np.polyfit(np.log(Ss), np.log(Fs), 1)[0])

class V1Binocular(nn.Module):
    """
    Shared binocular V1: energy -> soft disparity -> rivalry oscillator.
    ONLY learning: tiny tilewise disparity prior (EMA logits, bounded). Oscillations protected via confidence clamp.
    """
    def __init__(self, dmax=4, steps=15, tile=4, prior_strength=0.6, ema=0.02, prior_cap=2.0):
        super().__init__()
        self.dmax=dmax; self.steps=steps; self.tile=tile
        self.ema=ema; self.prior_strength=prior_strength; self.prior_cap=prior_cap
        Ds=torch.arange(-dmax,dmax+1)
        self.register_buffer("Ds",Ds)
        self.register_buffer("priors", torch.zeros(1, len(Ds), 1, 1))
        self._init=False; self.learn_enabled=True
    def _maybe_init(self,H,W,device):
        gh,gw=max(1,H//self.tile), max(1,W//self.tile)
        if self._init and self.priors.shape[-2:]==(gh,gw): return
        with torch.no_grad(): self.priors=torch.zeros(1,len(self.Ds),gh,gw,device=device); self._init=True
    @torch.no_grad()
    def _upsample_prior(self,H,W): return F.interpolate(self.priors, size=(H,W), mode='nearest')
    @torch.no_grad()
    def _update_priors(self,w):
        if not self.learn_enabled: return
        B,D,H,W=w.shape; gh,gw=self.priors.shape[-2:]; th,tw=H//gh, W//gw
        wt=w.unfold(2,th,th).unfold(3,tw,tw)      # [B,D,gh,gw,th,tw]
        wmean=wt.mean(dim=(0,4,5))                # [D,gh,gw]
        self.priors.mul_(1.0-self.ema).add_(self.ema*wmean.unsqueeze(0))
        eps=1e-6
        logp=torch.log(self.priors.clamp_min(eps))
        logp=logp-logp.mean(dim=1,keepdim=True)
        logp=logp.clamp(-self.prior_cap,self.prior_cap)
        self.priors.copy_(logp)
    @torch.no_grad()
    def forward(self, FL, FR, learn=True):
        B,C,H,W=FL.shape; self._maybe_init(H,W,FL.device)
        E, w_data, conf_data, Ds = disparity_energy(FL, FR, dmax=self.dmax)
        # add small prior bias (upsampled)
        prior_up=self._upsample_prior(H,W)
        E_biased = E + self.prior_strength*prior_up
        w = F.softmax(3.0*E_biased, dim=1)
        if learn and self.learn_enabled: self._update_priors(w)
        # fuse right by soft disparity
        fused_R=0
        for i,d in enumerate(Ds):
            fused_R = fused_R + w[:,i:i+1]*torch.roll(FR, shifts=int(d.item()), dims=3)
        # rivalry oscillator (limit-cycle protected)
        SL=FL.pow(2).mean(dim=1,keepdim=True)
        SR=fused_R.pow(2).mean(dim=1,keepdim=True)
        beta=0.95; gamma=0.8; alpha=0.35; lam=0.995; rho=0.006; eta=0.15; sigma=0.012
        r=torch.zeros_like(SL); aL=torch.zeros_like(SL); aR=torch.zeros_like(SL)
        for _ in range(self.steps):
            conf_used=conf_data.clamp(max=0.80)
            conflict=1.0-conf_used
            aL=lam*aL + rho*(r>0).float()
            aR=lam*aR + rho*(r<0).float()
            Delta=(SL - alpha*aL) - (SR - alpha*aR)
            noise=sigma*torch.randn_like(r)
            r=beta*r + gamma*torch.tanh(Delta) - eta*conflict + noise
            r=r.clamp(-1,1)
        # final fused features (never fully gate to keep oscillations visible)
        conf_used=conf_data.clamp(max=0.80)
        gL=(1+r)/2; gR=(1-r)/2; gbin=0.85*conf_used
        wta=torch.where(SL>=SR, FL, fused_R)
        Fbin = gbin*(gL*FL + gR*fused_R) + (1-gbin)*wta
        return Fbin, r, conf_data, w, E

# ----------------- Full model -----------------
class TwoEyesV1Model(nn.Module):
    def __init__(self, in_ch=3, cell_grid=4, dmax=4, steps=15):
        super().__init__()
        self.cellL=CellPop(cell_grid); self.cellR=CellPop(cell_grid)
        self.kL=KEye(grid=cell_grid);  self.kR=KEye(grid=cell_grid)
        self.mpkL=MPKExact(in_ch=in_ch); self.mpkR=MPKExact(in_ch=in_ch)
        self.v1 = V1Binocular(dmax=dmax, steps=steps, tile=4, prior_strength=0.6, ema=0.02, prior_cap=2.0)
        for p in self.parameters(): p.requires_grad=False
    @torch.no_grad()
    def forward(self, xL, xR, learn=True):
        L=self.kL(self.cellL(xL))
        R=self.kR(self.cellR(xR))
        FL,_,_=self.mpkL(L)
        FR,_,_=self.mpkR(R)
        return self.v1(FL, FR, learn=learn)

# ----------------- data + natural stereo offsets -----------------
def make_transforms(): return T.Compose([T.ToTensor()])  # keep raw; per-image norms happen in modules

@torch.no_grad()
def stereo_offsets(x, max_disp=4):
    """
    From a batch of base images x [B,C,H,W] in [0,1], create a binocular pair by
    applying symmetric horizontal translations: Left = -d//2, Right = +d - d//2.
    Uses bilinear resampling with fill=0 (no wraparound).
    Returns xL, xR, d_true (np array of ints).
    """
    B,C,H,W = x.shape
    d = torch.randint(low=-max_disp, high=max_disp+1, size=(B,), device=x.device)
    xL_list=[]; xR_list=[]
    for i in range(B):
        di = int(d[i].item())
        dxL = - (di // 2)
        dxR = di - dxL
        xi = x[i]
        # affine on tensor
        xLi = TF.affine(xi, angle=0.0, translate=[dxL, 0], scale=1.0, shear=[0.0, 0.0],
                        interpolation=InterpolationMode.BILINEAR, fill=0.0)
        xRi = TF.affine(xi, angle=0.0, translate=[dxR, 0], scale=1.0, shear=[0.0, 0.0],
                        interpolation=InterpolationMode.BILINEAR, fill=0.0)
        xL_list.append(xLi); xR_list.append(xRi)
    xL = torch.stack(xL_list, dim=0)
    xR = torch.stack(xR_list, dim=0)
    return xL, xR, d

# ----------------- main -----------------
def main():
    import time, math, random, argparse
    from pathlib import Path
    import numpy as np

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    from torchvision.transforms import InterpolationMode
    from torchvision.datasets import CIFAR100

    from torch.utils.tensorboard import SummaryWriter
    ap=argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='./data')
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--max_disp', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--logdir', type=str, default='./runs/mpk_v1bin')
    args=ap.parse_args()

    set_seed(args.seed); ensure_dir(args.logdir)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer=SummaryWriter(log_dir=args.logdir)

    tr=make_transforms()
    train_set=CIFAR100(root=args.data, train=True, transform=tr, download=True)
    test_set =CIFAR100(root=args.data, train=False, transform=tr, download=True)
    import platform
    mac = (platform.system() == "Darwin")

    train_loader = DataLoader(
        train_set,
    batch_size=args.batch,
    shuffle=False,
    num_workers=0 if mac else args.workers,
    pin_memory=False,           # important on macOS
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0 if mac else args.workers,
        pin_memory=False,
    )

    model=TwoEyesV1Model(in_ch=3, cell_grid=4, dmax=args.max_disp, steps=15).to(device)
    model.eval()

    global_step=0

    @torch.no_grad()
    def run_split(name, loader):
        nonlocal global_step
        t0=time.time()
        imgs_total=0; conf_sum=0.0; rabs_sum=0.0
        top1_hits=0; mae_sum=0.0; n_disp=0

        for it,(imgs, labels) in enumerate(loader):
            imgs=imgs.to(device, non_blocking=True)
            xL, xR, d_true = stereo_offsets(imgs, args.max_disp)
            # forward (learn only on train)
            learn=(name=='train')
            Fbin, r, conf, w, E = model(xL, xR, learn=learn)
            B=imgs.size(0)

            # disparity metrics (global): argmax over spatial-mean weights
            w_mean = w.mean(dim=(2,3))        # [B,D]
            d_set  = model.v1.Ds.cpu().numpy()
            d_pred_idx = w_mean.argmax(dim=1).cpu().numpy()
            d_pred = d_set[d_pred_idx]
            d_true_cpu = d_true.cpu().numpy()

            top1_hits += int((d_pred == d_true_cpu).sum())
            mae_sum   += float(np.abs(d_pred - d_true_cpu).sum())
            n_disp    += B

            # scalars
            conf_img = conf.mean(dim=(1,2,3)).cpu().numpy()
            r_abs    = r.abs().mean(dim=(1,2,3)).cpu().numpy()

            conf_sum += conf_img.sum()
            rabs_sum += r_abs.sum()
            imgs_total += B

            # throughput
            if (it+1) % 25 == 0:
                dt = time.time() - t0
                ips = imgs_total / max(dt,1e-6)
                writer.add_scalar('perf/imgs_per_sec', ips, global_step)

            # small rivalry probe for DFA (center-pixel series)
            Tsteps=64
            SL=Fbin.pow(2).mean(dim=1,keepdim=True)
            SR=torch.roll(SL, shifts=1, dims=3)  # a different competitor proxy
            r_t=torch.zeros((B,1,1,1), device=device)
            aL=torch.zeros_like(r_t); aR=torch.zeros_like(r_t)
            beta=0.95; gamma=0.8; alpha=0.35; lam=0.995; rho=0.006; eta=0.15; sigma=0.012
            conf_used = torch.full((B,1,1,1), 0.5, device=device)
            r_series=[]
            for t in range(Tsteps):
                aL=lam*aL + rho*(r_t>0).float(); aR=lam*aR + rho*(r_t<0).float()
                Delta=(SL.mean(dim=(2,3),keepdim=True) - alpha*aL) - (SR.mean(dim=(2,3),keepdim=True) - alpha*aR)
                noise=sigma*torch.randn_like(r_t)
                r_t = beta*r_t + gamma*torch.tanh(Delta) - eta*(1.0-conf_used) + noise
                r_t = r_t.clamp(-1,1); r_series.append(r_t.squeeze().detach().cpu().numpy())
            r_series=np.stack(r_series,axis=0).mean(axis=1)  # [T]
            dfa = dfa_alpha_1d(r_series, min_win=4, max_win=Tsteps//2, n_scales=6)
            writer.add_scalar('fractal/dfa_alpha', 0.0 if np.isnan(dfa) else dfa, global_step)

            # log scalars
            writer.add_scalar('confidence/mean_conf', conf_img.mean(), global_step)
            writer.add_scalar('rivalry/mean_abs_r', r_abs.mean(), global_step)
            writer.add_scalar('disparity/acc_top1', top1_hits / max(n_disp,1), global_step)
            writer.add_scalar('disparity/mae_px',  mae_sum / max(n_disp,1),  global_step)

            global_step += 1

        dt=time.time()-t0
        print(f"{name}: {imgs_total} imgs in {dt/60:.2f} min | conf {conf_sum/imgs_total:.4f} | |r| {rabs_sum/imgs_total:.4f} | acc {top1_hits/max(n_disp,1):.3f} | mae {mae_sum/max(n_disp,1):.3f}")

    run_split('train', train_loader)
    run_split('test',  test_loader)
    writer.flush(); writer.close()

if __name__ == '__main__':
    import platform, torch, torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)  # avoid fork on macOS
    except RuntimeError:
        pass
    # be very conservative with intra-op threads on macOS
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    main()
