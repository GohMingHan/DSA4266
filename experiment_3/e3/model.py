"""One frozen-feature fusion head; no pretrained image/audio encoders in this graph.

cross: V queries A, A queries V, with simultaneous bidirectional updates.
self:  V queries V, A queries A; parameter-identical independent temporal control.
concat: the exact Experiment 2B mean-pooled MLP class, for a matched-budget baseline.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from e2b.train import ConcatClassifier


class ContinuousTimeEncoding(nn.Module):
    """Shared video-relative seconds, NOT independent per-modality sequence indices.

Fixed log-spaced periods cover 0.5 to 512 seconds. Audio uses source-window centers;
window support is still recorded in the cache, not asserted instantaneous.
"""
    def __init__(self,width):
        super().__init__()
        if width<4 or width%2:raise ValueError('Attention width must be even and >=4.')
        periods=torch.logspace(math.log10(.5),math.log10(512.),width//2)
        self.register_buffer('angular_frequency',2*math.pi/periods)
    def forward(self,time,padding):
        time=time.masked_fill(padding,0.)
        phase=time.unsqueeze(-1)*self.angular_frequency
        pe=torch.stack([phase.sin(),phase.cos()],dim=-1).flatten(-2)
        return pe.masked_fill(padding.unsqueeze(-1),0.)


class DirectionalBlock(nn.Module):
    def __init__(self,width,heads,dropout,ff_multiplier):
        super().__init__()
        self.query_norm=nn.LayerNorm(width)
        self.context_norm=nn.LayerNorm(width)
        self.attention=nn.MultiheadAttention(width,heads,dropout=dropout,batch_first=True)
        self.dropout=nn.Dropout(dropout)
        self.ff_norm=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,width*ff_multiplier),nn.GELU(),nn.Dropout(dropout),
                              nn.Linear(width*ff_multiplier,width),nn.Dropout(dropout))
    def forward(self,query,context,query_padding,context_padding):
        q=self.query_norm(query);kv=self.context_norm(context)
        z,_=self.attention(q,kv,kv,key_padding_mask=context_padding,need_weights=False)
        h=(query+self.dropout(z)).masked_fill(query_padding.unsqueeze(-1),0.)
        h=(h+self.ff(self.ff_norm(h))).masked_fill(query_padding.unsqueeze(-1),0.)
        return h


def masked_mean(values,padding):
    valid=(~padding).unsqueeze(-1)
    return values.masked_fill(~valid,0.).sum(1)/valid.sum(1).clamp_min(1)


class FusionModel(nn.Module):
    def __init__(self,dims,normalization,mode='cross',width=256,heads=4,layers=1,
                 ff_multiplier=2,head_width=128,dropout=.2,time_encoding=True):
        super().__init__()
        if mode not in ('cross','self','concat'):raise ValueError('mode must be cross, self or concat.')
        if width<4 or width%2 or heads<1 or width%heads:raise ValueError('width must be even and divisible by heads.')
        if layers<1 or ff_multiplier<1 or head_width<1 or not 0<=dropout<1:raise ValueError('Invalid model settings.')
        self.mode=mode;self.use_time=bool(time_encoding)
        self.dims=dict(dims)
        for m in ('visual','audio'):
            for key in ('mean','std'):
                value=torch.as_tensor(normalization[m][key],dtype=torch.float32).clone()
                if value.shape!=(dims[m],) or not torch.isfinite(value).all():raise ValueError('Bad feature normalizer.')
                if key=='std' and (value<=0).any():raise ValueError('Invalid feature standard deviation.')
                self.register_buffer(f'{m}_{key}',value)
        if mode=='concat':
            self.classifier=ConcatClassifier(dims['visual'],dims['audio'],'concat',head_width,dropout)
            return
        self.visual_projection=nn.Linear(dims['visual'],width)
        self.audio_projection=nn.Linear(dims['audio'],width)
        self.time_encoding=ContinuousTimeEncoding(width)
        self.input_dropout=nn.Dropout(dropout)
        self.visual_blocks=nn.ModuleList([DirectionalBlock(width,heads,dropout,ff_multiplier) for _ in range(layers)])
        self.audio_blocks=nn.ModuleList([DirectionalBlock(width,heads,dropout,ff_multiplier) for _ in range(layers)])
        self.visual_out=nn.LayerNorm(width);self.audio_out=nn.LayerNorm(width)
        self.classifier=nn.Sequential(nn.Linear(2*width,head_width),nn.ReLU(),nn.Dropout(dropout),nn.Linear(head_width,1))
    def standardize(self,x,m):return (x-getattr(self,m+'_mean'))/getattr(self,m+'_std')
    def forward(self,batch):
        if self.mode=='concat':
            return self.classifier(self.standardize(batch['visual_mean'],'visual'),
                                   self.standardize(batch['audio_mean'],'audio'))
        seq={};masks={}
        for m in ('visual','audio'):
            padding=batch[m+'_padding'];x=batch[m]
            if padding.dtype!=torch.bool or x.ndim!=3 or padding.shape!=x.shape[:2]:raise ValueError('Bad padding shapes/types.')
            if x.shape[-1]!=self.dims[m]:raise ValueError('Embedding dimension does not match checkpoint.')
            if torch.any(padding.all(dim=1)):raise ValueError('A modality has no valid tokens; missing modalities are not zero-filled.')
            # Zero masked inputs BEFORE transformations; their arbitrary padding content cannot leak.
            x=x.masked_fill(padding.unsqueeze(-1),0.)
            x=self.standardize(x,m).masked_fill(padding.unsqueeze(-1),0.)
            h=getattr(self,m+'_projection')(x)
            if self.use_time:h=h+self.time_encoding(batch[m+'_time'],padding)
            seq[m]=self.input_dropout(h).masked_fill(padding.unsqueeze(-1),0.);masks[m]=padding
        v,a=seq['visual'],seq['audio'];vp,ap=masks['visual'],masks['audio']
        for vb,ab in zip(self.visual_blocks,self.audio_blocks):
            old_v,old_a=v,a
            if self.mode=='cross':
                v=vb(old_v,old_a,vp,ap)
                a=ab(old_a,old_v,ap,vp)
            else:
                v=vb(old_v,old_v,vp,vp)
                a=ab(old_a,old_a,ap,ap)
        vector=torch.cat([masked_mean(self.visual_out(v),vp),masked_mean(self.audio_out(a),ap)],dim=-1)
        return self.classifier(vector).squeeze(-1)


def build_model(dims,norm,settings):
    return FusionModel(dims,norm,**{k:settings[k] for k in
        ('mode','width','heads','layers','ff_multiplier','head_width','dropout','time_encoding')})
