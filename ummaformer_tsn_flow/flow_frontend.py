"""Explicit dense-flow -> five-pair TSN input, with no labels or HSV visualization.

Default Farneback retains the existing motion estimator and 224-square analysis
geometry. TV-L1 is an optional separately recorded variant; it is not a fallback.
Both are fixed estimators. Only the downstream CNN has learned pretrained weights.
"""
from __future__ import annotations
import numpy as np
import cv2
from PIL import Image

FARNEBACK = dict(pyr_scale=.5, levels=3, winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0)


def analysis_gray(native_rgb):
    # Exactly the former RGBEncoder.crop's PIL geometry; no ResNet forward needed.
    from torchvision.transforms import functional as TF, InterpolationMode
    im = TF.resize(Image.fromarray(native_rgb), 256,
                   interpolation=InterpolationMode.BILINEAR, antialias=True)
    im = TF.center_crop(im, [224, 224])
    return cv2.cvtColor(np.array(im, dtype=np.uint8, copy=True), cv2.COLOR_RGB2GRAY)


class DenseFlow:
    def __init__(self, method='farneback'):
        if method not in ('farneback', 'tvl1'):
            raise ValueError('flow-method must be farneback or tvl1.')
        self.method, self.tvl1 = method, None
        if method == 'tvl1':
            module = getattr(cv2, 'optflow', None)
            factory = getattr(module, 'DualTVL1OpticalFlow_create', None)
            if factory is None:
                raise ImportError('TV-L1 requires an OpenCV contrib build. Keep the working environment '
                                  'unchanged; install it in a separate extraction environment, or explicitly '
                                  'run the Farneback variant. There is no automatic fallback.')
            self.tvl1 = factory()
            # Explicit common DualTVL1 settings, not hardware-specific denseflow parity.
            setters = {'setTau': .25, 'setLambda': .15, 'setTheta': .3,
                       'setScalesNumber': 5, 'setWarpingsNumber': 5,
                       'setEpsilon': .01, 'setInnerIterations': 30,
                       'setOuterIterations': 10, 'setScaleStep': .8,
                       'setGamma': 0., 'setMedianFiltering': 5,
                       'setUseInitialFlow': False}
            for name, value in setters.items():
                getattr(self.tvl1, name)(value)
        self.identity = {'method': method, 'analysis_geometry': 'former_RGB_crop_short256_center224',
                         'parameters': FARNEBACK if method == 'farneback' else setters,
                         'pair': 'native_decoded_frame[k-1] -> frame[k]',
                         'units': 'pixels per native pair, NOT normalized velocity',
                         'tvl1_implementation': 'OpenCV CPU contrib, not necessarily paper denseflow' if self.tvl1 else None}

    def __call__(self, before, after):
        if before.shape != (224, 224) or after.shape != before.shape or before.dtype != np.uint8 or after.dtype != np.uint8:
            raise ValueError('Expected matching 224x224 uint8 grayscale frames.')
        flow = (cv2.calcOpticalFlowFarneback(before, after, None, **FARNEBACK)
                if self.tvl1 is None else self.tvl1.calc(before, after, None))
        if flow.shape != (224, 224, 2) or not np.isfinite(flow).all():
            raise ValueError('Dense flow calculation failed.')
        return flow


def encode_flow_image(flow, bound=20.0):
    """Bound/quantize dx,dy as conventional flow images, resize, normalize later.

No dividing by dt, no per-video extrema, no HSV, and no extra JPEG round trip.
Resizing the ENCODED flow image does not rescale physical displacement values.
This follows the form of the original extractor's image frontend, not its full
rawflow/JPEG geometry pipeline. The fixed bound is part of this adaptation.
"""
    f = np.asarray(flow, np.float32)
    if f.ndim != 3 or f.shape[-1] != 2 or not np.isfinite(f).all() or bound <= 0:
        raise ValueError('Expected finite HxWx2 dense flow and positive bound.')
    q = np.rint((np.clip(f, -bound, bound) + bound) * (255.0 / (2 * bound))).astype(np.uint8)
    q = cv2.resize(q, (256, 256), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(q.transpose(2, 0, 1))  # 2,H,W


def stack_pair_indices(source_indices, native_count):
    """Five chronological native pairs around each RGB anchor; edge-repeat only.

Pair k means original frames (k-1,k), k >= 1. Context at the clip boundaries
repeats the nearest EXISTING pair; this does not create extra source timestamps.
"""
    indices = np.asarray(source_indices)
    if native_count < 2 or indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError('Need >=2 native frames and integer source indices.')
    if (indices < 0).any() or (indices >= native_count).any() or (np.diff(indices) <= 0).any():
        raise ValueError('Invalid source indices.')
    return np.clip(indices[:, None] + np.arange(-2, 3)[None, :], 1, native_count - 1).astype(np.int64)


def pack_stack(encoded_pairs):
    x = np.stack(encoded_pairs)
    if x.shape != (5, 2, 256, 256) or x.dtype != np.uint8:
        raise ValueError('Expected five encoded [2,256,256] flow pairs.')
    # Chronological interleaving: dx1,dy1,dx2,dy2,...,dx5,dy5.
    return (x.reshape(10, 256, 256).astype(np.float32) - 128.0) / 128.0
