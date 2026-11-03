"""Groupmate's projected CLIP + normalized 512-vector + two-logit linear head.

The original output order is retained: index0 Fake, index1 Real. Dataset and
reported targets remain canonical Real=0/Fake=1. Imports never download weights.
"""
from dataclasses import asdict,dataclass
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .data import LazyDataset,stable_seed

MODEL_ID='clip_vit_b32_shared'
MODEL_NAME='openai/clip-vit-base-patch32'
REVISION='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
CLIP_SEED=4266


@dataclass(frozen=True)
class CLIPConfig:
    model:str=MODEL_ID
    seed:int=CLIP_SEED
    sampling_seed:int=CLIP_SEED
    batch_size:int=32
    epochs:int=15
    patience:int=5
    backbone_lr:float=3e-6
    head_lr:float=1e-4
    weight_decay:float=1e-4
    gradient_clip:float=1.
    flip_probability:float=.5
    train_per_class:int=4000
    workers:int=0
    prefetch_factor:int=2
    loader_timeout_seconds:float=120.
    amp:bool=True
    threshold:float=.5
    topk_fraction:float=.05
    checkpoint_selection:str='frame_roc_auc'
    train_progress_every:int=50
    dev_progress_every:int=200
    max_consecutive_gradient_skips:int=5
    max_gradient_skips_per_epoch:int=20
    pretrained_model:str=MODEL_NAME
    pretrained_revision:str=REVISION

    def as_dict(self):return asdict(self)

    def validate(self):
        fixed=CLIPConfig(amp=self.amp,workers=self.workers)
        if self!=fixed or type(self.amp) is not bool or self.workers<0:raise ValueError('CLIP recipe differs from the reviewed shared-data lower-encoder-LR experiment')
        return self


class CLIPBinaryClassifier(nn.Module):
    def __init__(self,weights=None,*,encoder=None):
        super().__init__()
        if encoder is None:
            from transformers import CLIPVisionModelWithProjection
            if weights is None:raise FileNotFoundError('Prepare pinned pretrained CLIP assets first')
            encoder=CLIPVisionModelWithProjection.from_pretrained(str(weights),local_files_only=True,use_safetensors=True,attn_implementation='eager')
        self.encoder=encoder
        self.classifier=nn.Linear(512,2)

    def forward(self,pixel_values):
        outputs=self.encoder(pixel_values=pixel_values)
        return self.classifier(F.normalize(outputs.image_embeds,p=2,dim=-1))


def fake_probability(logits):return logits.float().softmax(dim=1)[:,0]


def classification_loss(logits,canonical_targets):
    # Exact original two-class CE, remapping dataset Fake=1 -> original output0.
    return F.cross_entropy(logits,1-canonical_targets.long(),reduction='none')


def prepare_clip_assets(destination,*,download=False):
    """Explicit public pretrained download; snapshot is pinned, fresh head is local."""
    from pathlib import Path
    from .checkpoints import digest
    from .data import write_json
    import json
    destination=Path(destination)
    marker=destination/'clip_pretrained_manifest.json'
    if marker.is_file():
        record=json.loads(marker.read_text())
        if (record['model'],record['revision'])!=(MODEL_NAME,REVISION):raise ValueError('Pretrained CLIP identity differs')
        if not {'config.json','model.safetensors','preprocessor_config.json'}<=set(record['files']):raise ValueError('Incomplete CLIP pretrained snapshot')
        for name,sha in record['files'].items():
            if Path(name).name!=name or (destination/name).is_symlink():raise ValueError('Unsafe CLIP asset path')
            if digest(destination/name)!=sha:raise ValueError('Pretrained CLIP asset hash differs: '+name)
        return destination
    if not download:raise FileNotFoundError('Missing verified CLIP assets. Attach a saved snapshot or explicitly enable DOWNLOAD_CLIP_WEIGHTS.')
    if destination.exists() and any(destination.iterdir()):raise FileExistsError('Unverified CLIP directory preserved; use a new empty directory')
    from transformers import CLIPVisionModelWithProjection,CLIPImageProcessor
    # The original pinned repository has pytorch_model.bin, not safetensors.
    # Transformers/torch load its tensors with weights_only=True, then persist
    # only the required vision/projection weights locally as safetensors.
    encoder=CLIPVisionModelWithProjection.from_pretrained(MODEL_NAME,revision=REVISION,
        use_safetensors=False,weights_only=True,attn_implementation='eager')
    processor=CLIPImageProcessor.from_pretrained(MODEL_NAME,revision=REVISION)
    encoder.save_pretrained(destination,safe_serialization=True);processor.save_pretrained(destination)
    write_json(marker,{'model':MODEL_NAME,'revision':REVISION,'files':{p.name:digest(p) for p in destination.iterdir() if p.is_file()}})
    del encoder
    return destination


def load_processor(weights):
    from transformers import CLIPImageProcessor
    processor=CLIPImageProcessor.from_pretrained(str(weights),local_files_only=True)
    if processor.crop_size!={'height':224,'width':224} or processor.size!={'shortest_edge':224}:
        raise ValueError('Unexpected CLIP input geometry')
    return processor


def checkpoint_components(state,weights=None):
    """Restore fine-tuned encoder AND head offline; never fetch pretrained weights.

    Full checkpoint components make a fresh-session DEV/CAM run independent of
    the original pretrained snapshot. Legacy checkpoints need explicit assets.
    """
    if 'encoder_config' in state and 'processor_config' in state:
        from transformers import CLIPVisionConfig,CLIPVisionModelWithProjection,CLIPImageProcessor
        encoder=CLIPVisionModelWithProjection(CLIPVisionConfig.from_dict(state['encoder_config']))
        encoder.set_attn_implementation('eager')
        model=CLIPBinaryClassifier(encoder=encoder)
        processor=CLIPImageProcessor.from_dict(state['processor_config'])
    else:
        if weights is None:raise FileNotFoundError('Legacy CLIP checkpoint lacks embedded component configs; supply verified original assets')
        weights=prepare_clip_assets(weights)
        model=CLIPBinaryClassifier(weights);processor=load_processor(weights)
    if processor.crop_size!={'height':224,'width':224} or processor.size!={'shortest_edge':224}:raise ValueError('Checkpoint CLIP preprocessing geometry differs')
    model.load_state_dict(state['model'],strict=True)
    return model,processor


class CLIPDataset(LazyDataset):
    def __init__(self,records,data_root,processor,config,*,epoch=0,training=False,cache=None):
        super().__init__(records,data_root,config.sampling_seed,epoch,config.flip_probability if training else 0.,0)
        self.processor=processor
        if cache is not None:
            from .frame_cache import CachedReader
            self.reader=CachedReader(data_root,cache)
        if any(r['kind']!='frame' for r in records):raise ValueError('CLIP requires existing frame records')

    def __getitems__(self,indices):
        rows=[self.records[i] for i in indices];decoded=self.reader.read(rows)
        from PIL import Image
        images=[]
        for row,frames in zip(rows,decoded):
            pixels=frames[0]
            if stable_seed(self.seed,self.epoch,row['kind'],row['id'],'flip')/2**64<self.flip_probability:
                pixels=pixels[:,::-1,:].copy()
            images.append(Image.fromarray(pixels))
        values=self.processor(images=images,return_tensors='pt')['pixel_values']
        return [(value,torch.tensor(float(row['label'])),i) for value,row,i in zip(values,rows,indices)]


def make_clip_loader(records,data_root,processor,config,*,epoch=0,training=False,cache=None):
    dataset=CLIPDataset(records,data_root,processor,config,epoch=epoch,training=training,cache=cache)
    args={'num_workers':config.workers,'pin_memory':True,'generator':torch.Generator().manual_seed(stable_seed(config.sampling_seed,epoch,training)%2**63)}
    if config.workers:args.update(multiprocessing_context='spawn',prefetch_factor=config.prefetch_factor,timeout=config.loader_timeout_seconds)
    return torch.utils.data.DataLoader(dataset,batch_size=config.batch_size,shuffle=False,**args)


class CLIPViTGradCAM:
    """Adapted groupmate penultimate-layer token Grad-CAM; Fake logit index0."""
    def __init__(self,model):
        self.model=model
        self.target_layer=model.encoder.vision_model.encoder.layers[-2].layer_norm1

    def generate(self,pixel_values):
        captured=[];states=[(m,m.training) for m in self.model.modules()]
        hook=self.target_layer.register_forward_hook(lambda m,i,o:captured.append(o))
        try:
            self.model.eval()
            with torch.inference_mode(False),torch.enable_grad(),torch.autocast(pixel_values.device.type,enabled=False):
                logits=self.model(pixel_values.detach().clone().float().requires_grad_(True))
                if len(captured)!=1:raise RuntimeError('Expected one transformer activation')
                activation=captured[0];gradient,=torch.autograd.grad(logits[:,0].sum(),activation)
                if not all(bool(torch.isfinite(t).all()) for t in (logits,activation,gradient)):raise FloatingPointError('Non-finite CLIP attribution')
                a,g=activation[:,1:,:],gradient[:,1:,:];grid=math.isqrt(a.shape[1])
                if grid*grid!=a.shape[1]:raise ValueError('Patch tokens do not form a square')
                maps=(a*g.mean(dim=1,keepdim=True)).sum(dim=-1).relu().reshape(-1,1,grid,grid)
                maps=F.interpolate(maps,size=(224,224),mode='bilinear',align_corners=False)[:,0]
                maxima=maps.flatten(1).amax(1).view(-1,1,1)
                maps=torch.where(maxima>0,maps/maxima.clamp_min(1e-8),torch.zeros_like(maps))
                return maps.detach().cpu().numpy(),fake_probability(logits).detach().cpu().numpy()
        finally:
            hook.remove();captured.clear()
            for module,state in states:module.training=state
