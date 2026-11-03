"""Five finalized recipes, immutable defaults and ROC-AUC selection."""

from dataclasses import asdict, dataclass, replace
import math


@dataclass
class Config:
    model: str = 'resnet50'
    seed: int = 4266
    batch_size: int = 32
    epochs: int = 10
    warmup_epochs: int = 2
    patience: int = 3
    head_lr: float = 1e-4
    backbone_lr: float = 1.5e-5
    weight_decay: float = 1e-4
    dropout: float = .3
    flip_probability: float = .5
    finetune_final_stage: bool = True
    train_per_class: int = 4000
    clip_per_class: int = 2000
    workers: int = 2
    prefetch_factor: int = 2
    cache_mib: int = 0
    amp: bool = True
    threshold: float = .5
    topk_fraction: float = .05
    scheduler_patience: int = 1
    loader_timeout_seconds: float = 120.
    train_progress_every: int = 50
    dev_progress_every: int = 200
    heartbeat_seconds: float = 30.
    max_consecutive_gradient_skips: int = 5
    max_gradient_skips_per_epoch: int = 20
    checkpoint_selection: str = 'frame_roc_auc'
    training_pool: str = 'fixed20k'

    def validate(self):
        if self.training_pool != 'fixed20k':
            raise ValueError('Unknown CNN training-pool protocol')
        if self.training_pool=='fixed20k' and self.train_per_class!=4000:
            raise ValueError('Fixed-pool CNN epochs require 4,000 per class')
        if self.checkpoint_selection != 'frame_roc_auc':
            raise ValueError('The standalone finalized CNN policy requires DEV frame ROC-AUC')
        if self.model not in ('resnet50', 'efficientnet_b0'):
            raise ValueError('This stage trains CNNs only')
        if not 1 <= self.warmup_epochs < self.epochs:
            raise ValueError('Use at least one head epoch and one later epoch')
        if min(self.batch_size, self.patience, self.train_per_class,
               self.clip_per_class, self.prefetch_factor) < 1 or self.batch_size % 2:
            raise ValueError('Positive counts and an even batch size required')
        if self.workers < 0 or self.cache_mib < 0 or self.scheduler_patience < 0:
            raise ValueError('Invalid loader or scheduler configuration')
        if type(self.amp) is not bool or min(self.max_consecutive_gradient_skips, self.max_gradient_skips_per_epoch) < 1:
            raise ValueError('AMP must be boolean; gradient skip limits must be positive')
        if (not math.isfinite(self.loader_timeout_seconds) or self.loader_timeout_seconds < 0 or
            not math.isfinite(self.heartbeat_seconds) or self.heartbeat_seconds < 0 or
            min(self.train_progress_every, self.dev_progress_every) < 1):
            raise ValueError('Invalid progress interval or worker timeout')
        for x in (self.head_lr, self.backbone_lr):
            if not math.isfinite(x) or x <= 0:
                raise ValueError('Learning rates must be finite and positive')
        if self.backbone_lr > self.head_lr:
            raise ValueError('Backbone LR must not exceed head LR')
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('Invalid weight decay')
        if not 0 <= self.dropout < 1 or not 0 <= self.flip_probability <= 1:
            raise ValueError('Invalid dropout or augmentation probability')
        if not 0 <= self.threshold <= 1 or self.topk_fraction != .05:
            raise ValueError('This version fixes Top-5% aggregation')
        return self

    def as_dict(self):
        return asdict(self)


def selection_label(selection):
    return {'video_roc_auc': 'DEV video ROC-AUC, fixed Top-5%',
            'frame_pr_auc': 'DEV frame PR-AUC (trapezoidal)',
            'frame_roc_auc': 'DEV frame ROC-AUC',
            'clip_roc_auc': 'DEV clip ROC-AUC',
            'clip_pr_auc':'DEV clip PR-AUC (trapezoidal)'}[selection]


def checkpoint_score(report, selection):
    selection_label(selection)
    value = (report['video']['roc_auc'] if selection == 'video_roc_auc' else
             report['unit']['roc_auc'] if selection in ('frame_roc_auc', 'clip_roc_auc') else report['unit']['pr_auc'])
    if value is None or not math.isfinite(value):
        raise ValueError('Checkpoint-selection score is undefined/non-finite')
    return value


@dataclass
class TemporalConfig(Config):
    model: str='resnet50_lstm'
    training_pool: str='temporal_clips'
    checkpoint_selection: str='clip_roc_auc'
    batch_size: int=8
    clip_per_class: int=1500
    warmup_epochs: int=0
    hidden_size: int=128
    lstm_layers: int=1
    feature_microbatch: int=32
    workers: int=0
    finetune_final_stage: bool=False

    def validate(self):
        # Validate common numerical/loader settings through the unchanged CNN validator.
        from dataclasses import fields
        values={f.name:getattr(self,f.name) for f in fields(Config)}
        values.update(model='resnet50',training_pool='fixed20k',checkpoint_selection='frame_roc_auc',warmup_epochs=1)
        Config(**values).validate()
        if (self.model!='resnet50_lstm' or self.training_pool!='temporal_clips' or
            self.checkpoint_selection != 'clip_roc_auc' or self.warmup_epochs!=0 or self.finetune_final_stage or
            min(self.hidden_size,self.lstm_layers,self.feature_microbatch)<1):
            raise ValueError('Temporal protocol requires frozen ResNet50 and predeclared clip PR/ROC-AUC')
        return self


@dataclass(frozen=True)
class Recipe:
    label: str
    architecture: str
    input: str

RECIPES = {
    'resnet50_baseline': Recipe('ResNet50 baseline','resnet50','full_frame'),
    'resnet50_face_mask': Recipe('ResNet50 Face mask','resnet50','face_mask'),
    'resnet50_lstm': Recipe('ResNet50 LSTM','resnet50_lstm','full_frame_clips'),
    'efficientnet_b0_baseline': Recipe('EfficientNet-B0 baseline','efficientnet_b0','full_frame'),
    'efficientnet_b0_face_mask': Recipe('EfficientNet-B0 Face mask','efficientnet_b0','face_mask'),
}
ORDER=tuple(RECIPES)
INDEX_SHA256='ac3cd8c440f0bb1e53cf955bcc6a2596e9ee1bc40e508f254c82e10bde13cfcb'
INDEX_LOGICAL_SHA256='af2d57a66eeef994bd56161b415fcde4bc8a898a9d8cf286130e5467acf3f33e'
POOL_SHA256='8bcb5c9845f92b43e36213f22e410bcecedf983f7db87250f02a9968868eea92'

def configuration(model_id, *, amp=True, workers=None):
    r=RECIPES[model_id]
    c=TemporalConfig(amp=amp) if r.architecture=='resnet50_lstm' else Config(model=r.architecture,amp=amp)
    if workers is not None:c.workers=workers
    return c.validate()

def declared_configurations():
    return {k:{'label':r.label,'input':r.input,'config':configuration(k).as_dict(),
               'checkpoint_monitor':'DEV clip ROC-AUC' if k=='resnet50_lstm' else 'DEV frame ROC-AUC',
               'scheduler_monitor':'DEV clip ROC-AUC' if k=='resnet50_lstm' else 'DEV frame ROC-AUC',
               'comparison_metric':'DEV video trapezoidal PR-AUC','classification_threshold':.5}
            for k,r in RECIPES.items()}
