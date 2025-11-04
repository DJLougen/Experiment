#!/usr/bin/env python3
# dvx_text_bench.py
# Train DVX (MPK + Dorsal/Ventral) on enwik8 or text8 with BPC evaluation.

import argparse, math, time, random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

# -------------------- Utilities --------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def bpc_from_loss(loss: float) -> float:
    return loss / math.log(2.0)

def chunk_ids(buf: torch.Tensor, seq_len: int):
    L = (buf.numel() - 1) // seq_len
    buf = buf[: L * seq_len + 1]
    x = buf[:-1].view(L, seq_len)
    y = buf[1: ].view(L, seq_len)
    return x, y

def get_batch(stream_x, stream_y, batch_size, device):
    N = stream_x.size(0)
    idx = torch.randint(0, N, (batch_size,))
    return stream_x[idx].to(device), stream_y[idx].to(device)

# -------------------- Core layers --------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return self.scale * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

class CausalDWConv1d(nn.Module):
    def __init__(self, channels, kernel_size, dilation=1):
        super().__init__()
        self.ks = kernel_size
        self.dil = dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              dilation=dilation, groups=channels)
    def forward(self, x):
        x = x.transpose(1,2)
        x = F.pad(x, (self.dil*(self.ks-1),0))
        return self.conv(x).transpose(1,2)

class SEGate(nn.Module):
    def __init__(self, channels, hidden=None, num_mix_heads=2):
        super().__init__()
        if hidden is None: hidden = max(64, channels//8)
        self.fc1, self.fc2 = nn.Linear(channels, hidden), nn.Linear(hidden, channels)
        self.mix = nn.Linear(channels, num_mix_heads)
    def forward(self, x):
        g = torch.sigmoid(self.fc2(F.gelu(self.fc1(x.mean(1)))))
        logits = self.mix(x)
        return g.unsqueeze(1), F.softmax(logits, -1)

# -------------------- MPK + D/V --------------------
class LGN_MPK(nn.Module):
    def __init__(self, channels, k_long=9, k_short=3, dil_long=2, expand=2):
        super().__init__()
        self.norm = RMSNorm(channels)
        self.m_dw, self.m_pw1, self.m_pw2 = CausalDWConv1d(channels,k_long,dil_long), nn.Linear(channels,channels*expand), nn.Linear(channels*expand,channels)
        self.p_dw, self.p_pw1, self.p_pw2 = CausalDWConv1d(channels,k_short,1), nn.Linear(channels,channels*expand), nn.Linear(channels*expand,channels)
        self.k, self.out = SEGate(channels, num_mix_heads=2), nn.Linear(channels,channels)
    def forward(self, x):
        h = self.norm(x)
        m = self.m_pw2(F.gelu(self.m_pw1(self.m_dw(h))))
        p = self.p_pw2(F.gelu(self.p_pw1(self.p_dw(h))))
        ch_scale, mix = self.k(h)
        fused = mix[...,0:1]*m*ch_scale + mix[...,1:2]*p*ch_scale
        return self.out(fused)

class DorsalBlock(nn.Module):
    def __init__(self, channels, k_long=9, dil=2, expand=2):
        super().__init__()
        self.norm, self.dw = RMSNorm(channels), CausalDWConv1d(channels,k_long,dil)
        self.pw1, self.pw2, self.out = nn.Linear(channels,channels*expand), nn.Linear(channels*expand,channels), nn.Linear(channels,channels)
    def forward(self, x):
        y = self.pw2(F.gelu(self.pw1(self.dw(self.norm(x)))))
        return x + self.out(y)

class VentralBlock(nn.Module):
    def __init__(self, channels, k_short=3, expand=2):
        super().__init__()
        self.norm, self.dw = RMSNorm(channels), CausalDWConv1d(channels,k_short,1)
        self.pw1, self.pw2, self.out = nn.Linear(channels,channels*expand), nn.Linear(channels*expand,channels), nn.Linear(channels,channels)
    def forward(self, x):
        y = self.pw2(F.gelu(self.pw1(self.dw(self.norm(x)))))
        return x + self.out(y)

class DV_Fusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.k, self.out = SEGate(channels, num_mix_heads=2), nn.Linear(channels,channels)
    def forward(self, d,v):
        base=(d+v)/2
        ch_scale,mix=self.k(base)
        fused=mix[...,0:1]*d*ch_scale+mix[...,1:2]*v*ch_scale
        return base+self.out(fused)

class DVXTextMPK(nn.Module):
    def __init__(self, vocab_size, seq_len, channels=512, layers=6, k_long=9, k_short=3, dil_base=2, expand=2):
        super().__init__()
        self.seq_len=seq_len
        self.tok, self.pos = nn.Embedding(vocab_size,channels), nn.Embedding(seq_len,channels)
        self.lgn=nn.ModuleList([LGN_MPK(channels,k_long,k_short,dil_base+(i%3),expand) for i in range(layers)])
        self.dorsal=nn.ModuleList([DorsalBlock(channels,k_long,dil_base+(i%3),expand) for i in range(layers)])
        self.ventral=nn.ModuleList([VentralBlock(channels,k_short,expand) for _ in range(layers)])
        self.fuse=nn.ModuleList([DV_Fusion(channels) for _ in range(layers)])
        self.norm, self.head=RMSNorm(channels), nn.Linear(channels,vocab_size)
    def forward(self, idx):
        B,T=idx.shape
        pos=torch.arange(T,device=idx.device).unsqueeze(0)
        x=self.tok(idx)+self.pos(pos); d,v=x,x
        for l,dblk,vblk,fuse in zip(self.lgn,self.dorsal,self.ventral,self.fuse):
            lgn_out=l(x); d=dblk(d+lgn_out); v=vblk(v+lgn_out); x=fuse(d,v)
        return self.head(self.norm(x))

# -------------------- Dataset loaders --------------------
def load_enwik8():
    ds=load_dataset("enwik8"); raw=ds["train"][0]["text"]
    if isinstance(raw,str): raw=raw.encode("utf-8")
    data=torch.tensor(list(raw),dtype=torch.long)
    return data[:90_000_000], data[90_000_000:95_000_000], data[95_000_000:], 256

def load_text8():
    ds=load_dataset("text8"); raw=ds["train"][0]["text"]
    if isinstance(raw,bytes): raw=raw.decode("utf-8")
    charset=list("abcdefghijklmnopqrstuvwxyz "); stoi={c:i for i,c in enumerate(charset)}
    ids=torch.tensor([stoi.get(ch,26) for ch in raw],dtype=torch.long)
    return ids[:90_000_000], ids[90_000_000:95_000_000], ids[95_000_000:], 27

# -------------------- Eval --------------------
@torch.no_grad()
def eval_bpc(model,x,y,batch_size,device):
    model.eval(); total_loss,total_tok=0.0,0
    for i in range(0,x.size(0),batch_size):
        xb,yb=x[i:i+batch_size].to(device),y[i:i+batch_size].to(device)
        logits=model(xb)
        loss=F.cross_entropy(logits.view(-1,logits.size(-1)), yb.view(-1), reduction="sum")
        total_loss+=loss.item(); total_tok+=yb.numel()
    return bpc_from_loss(total_loss/total_tok)

# -------------------- Main --------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",choices=["enwik8","text8"],required=True)
    ap.add_argument("--seq_len",type=int,default=512)
    ap.add_argument("--batch_size",type=int,default=32)
    ap.add_argument("--channels",type=int,default=512)
    ap.add_argument("--layers",type=int,default=6)
    ap.add_argument("--steps",type=int,default=1000)
    args=ap.parse_args()

    device=get_device(); print("Device:",device)
    if args.dataset=="enwik8": train,valid,test,vocab=load_enwik8()
    else: train,valid,test,vocab=load_text8()

    tx,ty=chunk_ids(train,args.seq_len); vx,vy=chunk_ids(valid,args.seq_len); qx,qy=chunk_ids(test,args.seq_len)
    model=DVXTextMPK(vocab,args.seq_len,args.channels,args.layers).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4)

    for step in range(1,args.steps+1):
        xb,yb=get_batch(tx,ty,args.batch_size,device)
        loss=F.cross_entropy(model(xb).view(-1,vocab), yb.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step%100==0 or step==1:
            val_bpc=eval_bpc(model,vx,vy,min(8,args.batch_size),device)
            print(f"Step {step}/{args.steps} | Train CE {loss.item():.3f} | Val BPC {val_bpc:.3f}")

    print("Final val BPC:", eval_bpc(model,vx,vy,args.batch_size,device))
    print("Final test BPC:", eval_bpc(model,qx,qy,args.batch_size,device))

if __name__=="__main__":
    main()
