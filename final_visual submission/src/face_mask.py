"""Pinned YuNet/MediaPipe geometry, cached face masking and full-frame fallback."""

from dataclasses import asdict, dataclass
from contextlib import closing
from fractions import Fraction
import hashlib
import json
import math
import os
import sqlite3
import numpy as np
import torch
import csv
import shutil
from pathlib import Path
from torch.utils.data import DataLoader
from .checkpoints import digest
from .data import FrameReader, LazyDataset, stable_seed
from collections import Counter


DETECTOR_SHA256 = '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'
DETECTOR_URL = ('https://media.githubusercontent.com/media/opencv/opencv_zoo/main/'
                'models/face_detection_yunet/face_detection_yunet_2023mar.onnx')
CROP_VERSION = 'yunet2023mar_primary_area_square_margin_total_v2'


@dataclass(frozen=True)
class CropConfig:
    margin: float = .30  # TOTAL expansion: 30% larger side, i.e. 15% padding per side
    score_threshold: float = .8
    nms_threshold: float = .3
    top_k: int = 5000
    fallback: str = 'full_frame'
    output_size: int = 224
    interpolation: str = 'linear'

    def validate(self):
        if not .25 <= self.margin <= .35:
            raise ValueError('Face margin must be 25%-35% total expansion (12.5%-17.5% per side)')
        if not 0 < self.score_threshold < 1 or not 0 < self.nms_threshold < 1 or self.top_k < 1:
            raise ValueError('Invalid detector thresholds')
        if (self.fallback, self.output_size, self.interpolation) != ('full_frame', 224, 'linear'):
            raise ValueError('This protocol fixes full-frame fallback, 224 RGB and bilinear resize')
        return self


def ensure_detector(path, *, download=False):
    """Explicit small public-weight download, pinned by SHA; never a training download."""
    path = Path(path)
    if path.is_file():
        if digest(path) != DETECTOR_SHA256:
            raise ValueError('YuNet checkpoint SHA256 differs; do not use an unreviewed detector')
        return path
    if not download:
        raise FileNotFoundError('Attach face_detection_yunet_2023mar.onnx or explicitly run ensure_detector(download=True) in setup')
    import urllib.request
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name+'.download')
    with urllib.request.urlopen(DETECTOR_URL, timeout=90) as source, partial.open('xb') as target:
        import shutil
        shutil.copyfileobj(source, target)
    if digest(partial) != DETECTOR_SHA256:
        raise ValueError('Downloaded detector failed SHA256; partial file preserved for diagnosis')
    partial.rename(path)
    return path


def crop_rectangle(box, width, height, margin=.3):
    """Square, shift inside image (no black padding); right/bottom exclusive."""
    x, y, w, h = map(float, box)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or min(w, h) <= 0:
        raise ValueError('Invalid face bounding box')
    side = min(min(width, height), max(1, math.ceil(max(w, h)*(1+margin))))
    left = min(max(0, math.floor(x+w/2-side/2)), width-side)
    top = min(max(0, math.floor(y+h/2-side/2)), height-side)
    return [int(left), int(top), int(left+side), int(top+side)]


def choose_primary(faces, width, height, config):
    """Largest valid face, then confidence, centre distance, coordinates. No labels."""
    candidates = []
    if faces is not None:
        for f in faces:
            f = np.asarray(f).tolist()
            if len(f) < 15 or not all(math.isfinite(float(v)) for v in f):
                continue
            x, y, w, h, score = *map(float, f[:4]), float(f[14])
            if score < config.score_threshold or min(w, h) <= 0 or x+w <= 0 or y+h <= 0 or x >= width or y >= height:
                continue
            area = max(0, min(width, x+w)-max(0, x))*max(0, min(height, y+h)-max(0, y))
            distance = (x+w/2-width/2)**2+(y+h/2-height/2)**2
            candidates.append(((-area, -score, distance, x, y, w, h), f))
    if not candidates:
        return {'status': 'fallback_no_face', 'confidence': None, 'bbox_xywh': None,
                'crop_xyxy': [0, 0, width, height], 'faces_detected': 0}
    _, face = min(candidates, key=lambda pair: pair[0])
    return {'status': 'detected', 'confidence': float(face[14]), 'bbox_xywh': list(map(float, face[:4])),
            'crop_xyxy': crop_rectangle(face[:4], width, height, config.margin), 'faces_detected': len(candidates)}


class YuNet:
    def __init__(self, weights, config):
        import cv2
        if cv2.__version__ != '4.13.0':
            raise RuntimeError('Use the pinned opencv-contrib-python==4.13.0.92 for reproducible face detection')
        ensure_detector(weights)
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        self.config = config.validate()
        self.detector = cv2.FaceDetectorYN.create(str(weights), '', (224, 224), config.score_threshold,
            config.nms_threshold, config.top_k, cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)

    def detect(self, rgb):
        import cv2
        h, w = rgb.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return choose_primary(faces, w, h, self.config)


def frame_key(record):
    if record['split'] not in ('train', 'dev') or record['kind'] != 'frame' or len(record['indices']) != 1:
        raise ValueError('Face experiments expose existing TRAIN/DEV CNN frames only')
    if not record['source_file'].startswith(record['split']+'/') or '..' in Path(record['source_file']).parts:
        raise ValueError('Invalid source identity')
    return json.dumps([record['source_file'], int(record['indices'][0]), str(Fraction(record['timestamps_exact'][0]))], separators=(',', ':'))


class BoxCache:
    """Process-local SQLite connections, transactional per-frame writes, WAL for spawn."""
    def __init__(self, path, policy):
        self.path, self.policy = Path(path), policy
        self.db = None
        self.pid = None

    def connection(self):
        if self.db is None or self.pid != os.getpid():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(self.path, timeout=90)
            self.db.execute('PRAGMA journal_mode=WAL')
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.execute('CREATE TABLE IF NOT EXISTS info (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            self.db.execute('CREATE TABLE IF NOT EXISTS boxes (frame_key TEXT PRIMARY KEY, split TEXT, source_file TEXT, source_index INTEGER, timestamp_exact TEXT, metadata TEXT NOT NULL)')
            value = json.dumps(self.policy, sort_keys=True)
            self.db.execute('INSERT OR IGNORE INTO info VALUES (?,?)', ('policy', value)); self.db.commit()
            if self.db.execute("SELECT value FROM info WHERE key='policy'").fetchone()[0] != value:
                self.close()
                raise ValueError('Face cache policy/detector/dataset changed; use a separate cache')
            self.pid = os.getpid()
        return self.db

    def get(self, record):
        row = self.connection().execute('SELECT metadata FROM boxes WHERE frame_key=?', (frame_key(record),)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, record, metadata):
        db = self.connection()
        db.execute('INSERT OR IGNORE INTO boxes VALUES (?,?,?,?,?,?)', (frame_key(record), record['split'],
            record['source_file'], record['indices'][0], str(Fraction(record['timestamps_exact'][0])), json.dumps(metadata, sort_keys=True)))
        db.commit()

    def snapshot(self, output):
        """SQLite backup API includes committed WAL rows; copying the live DB is unsafe."""
        output = Path(output); temporary = output.with_suffix('.snapshot.tmp')
        if output.resolve() == self.path.resolve():
            raise ValueError('Snapshot destination must differ from the live face-cache database')
        with closing(sqlite3.connect(temporary)) as target:
            self.connection().backup(target)
        temporary.replace(output)

    def summary(self):
        from collections import Counter
        counts, confidences, areas = Counter(), [], []
        for split, payload in self.connection().execute('SELECT split,metadata FROM boxes'):
            metadata = json.loads(payload); counts[(split, metadata['status'])] += 1
            x1, y1, x2, y2 = metadata['crop_xyxy']; areas.append((x2-x1)*(y2-y1)/(224*224))
            if metadata['confidence'] is not None:
                confidences.append(metadata['confidence'])
        return {'counts': {f'{s}_{status}': n for (s, status), n in sorted(counts.items())},
                'cached_unique_frames': sum(counts.values()), 'confidence_mean_detected': sum(confidences)/len(confidences) if confidences else None,
                'mean_retained_area_fraction': sum(areas)/len(areas) if areas else None,
                'full_frame_inputs_including_fallback': sum(a == 1 for a in areas),
                'dev_frames_excluded': 0, 'test_used': False}

    def close(self):
        if self.db is not None:
            self.db.close(); self.db = None; self.pid = None

    def __getstate__(self):
        return {'path': self.path, 'policy': self.policy, 'db': None, 'pid': None}

    def __del__(self):
        if getattr(self, 'db', None) is not None:
            self.close()


class FaceReader(FrameReader):
    def __init__(self, root, weights, cache_path, policy, config, cache_mib=0):
        super().__init__(root, cache_mib)
        self.weights, self.config = str(weights), config
        self.boxes = BoxCache(cache_path, policy)
        self.detector = None


    def original(self, records):
        return super().read(records)


class FaceDataset(LazyDataset):
    def __init__(self, records, data_root, pipeline, seed=4266, epoch=0, flip_probability=0., cache_mib=0):
        super().__init__(records, data_root, seed, epoch, flip_probability, cache_mib)
        self.reader = FaceReader(data_root, pipeline.weights, pipeline.cache_path,
                                 pipeline.identity, pipeline.config, cache_mib)


class FacePipeline:


    def make_loader(self, records, data_root, config, training=False, epoch=0):
        dataset = self.dataset(records, data_root, config, training, epoch)
        kwargs = {'num_workers': config.workers, 'pin_memory': True,
                  'generator': torch.Generator().manual_seed(stable_seed(config.seed, epoch, training) % 2**63)}
        if config.workers:
            kwargs.update(prefetch_factor=config.prefetch_factor, persistent_workers=False,
                          multiprocessing_context='spawn', timeout=config.loader_timeout_seconds)
        return DataLoader(dataset, batch_size=config.batch_size, shuffle=False, **kwargs)


PREPROCESSING_MODES = ('face_mask',)
MASK_VERSION = 'full_face_oval_outer_feather_v1'
LANDMARK_VERSION = '0.10.32'
LANDMARK_URL = ('https://storage.googleapis.com/mediapipe-models/face_landmarker/'
                'face_landmarker/float16/1/face_landmarker.task')
LANDMARK_SHA256 = '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'
LANDMARK_PACKAGES = ('mediapipe==0.10.32', 'absl-py==2.3.1', 'flatbuffers==25.9.23',
                     'sounddevice==0.5.3', 'cffi==1.17.1', 'pycparser==2.22')
OPENCV_PACKAGE = 'opencv-contrib-python'
OPENCV_VERSION = '4.13.0.92'
OPENCV_DISTRIBUTIONS = ('opencv-python', 'opencv-python-headless',
                        'opencv-contrib-python', 'opencv-contrib-python-headless')
# Ordered official MediaPipe FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL.
FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)
REGIONS = {
    'forehead': (10, 67, 109, 338, 297, 103, 332),
    'eyebrows': (70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 300, 293, 334, 296, 336, 285, 295, 282, 283, 276),
    'eyes': (33, 133, 159, 145, 263, 362, 386, 374),
    'nose': (1, 4, 5, 6, 98, 327), 'cheeks': (234, 93, 132, 454, 323, 361),
    'mouth': (61, 291, 0, 17, 13, 14, 78, 308),
    'jaw': (172, 136, 150, 176, 152, 400, 378, 365, 397), 'chin': (152, 148, 377)}


@dataclass(frozen=True)
class MaskConfig:
    margin_fraction: float = .06  # outward radius / largest oval dimension
    feather_pixels: int = 4       # OUTSIDE the fully preserved face + margin
    neutral_rgb: tuple = (128, 128, 128)
    min_detection_confidence: float = .5
    min_presence_confidence: float = .5
    max_faces: int = 5
    fallback: str = 'full_frame'

    def validate(self):
        if not 0 < self.margin_fraction <= .15 or not 2 <= self.feather_pixels <= 12:
            raise ValueError('Use a small outward margin and 2-12px feather')
        if len(self.neutral_rgb) != 3 or any(not isinstance(v, int) or not 0 <= v <= 255 for v in self.neutral_rgb):
            raise ValueError('neutral_rgb must contain three integer RGB values')
        if self.fallback != 'full_frame' or not 0 < self.min_detection_confidence < 1 or not 0 < self.min_presence_confidence < 1 or not 1 <= self.max_faces <= 10:
            raise ValueError('Invalid landmark policy; only explicit full-frame fallback is supported')
        return self


def ensure_landmarker(path, *, download=False):
    """Versioned public model, SHA-pinned; existing files are never overwritten."""
    path = Path(path)
    if path.is_file():
        if digest(path) != LANDMARK_SHA256:
            raise ValueError('Face landmarker SHA256 mismatch')
        return path
    if not download:
        raise FileNotFoundError('Attach face_landmarker_float16_v1.task or allow its explicit setup download')
    import urllib.request
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.task.download')
    with urllib.request.urlopen(LANDMARK_URL, timeout=90) as source, temporary.open('xb') as target:
        shutil.copyfileobj(source, target)
    if digest(temporary) != LANDMARK_SHA256:
        raise ValueError('Landmarker download hash mismatch; partial preserved')
    temporary.rename(path)
    return path


def _opencv_distributions():
    import importlib.metadata
    versions = {}
    for name in OPENCV_DISTRIBUTIONS:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    return versions


def ensure_opencv_runtime(*, install=False):
    """Use exactly one cv2 provider satisfying MediaPipe, at the audited version.

    Download a compatible wheel before removing overlapping distributions. Never
    replace native cv2 inside a process that has already imported it. NumPy,
    torch, torchvision, detector settings and mask policy are left unchanged.
    """
    import importlib
    import importlib.metadata
    import subprocess
    import sys
    import tempfile
    versions = _opencv_distributions()
    if versions != {OPENCV_PACKAGE: OPENCV_VERSION}:
        if 'cv2' in sys.modules:
            raise RuntimeError('OpenCV providers differ from the audited setup and cv2 is already loaded. '
                               'Restart the remote kernel and run setup first; no packages were changed.')
        if not install:
            raise RuntimeError('Use only opencv-contrib-python==4.13.0.92; run '
                               'ensure_opencv_runtime(install=True) in a fresh kernel before importing cv2.')
        if int(np.__version__.split('.')[0]) < 2:
            raise RuntimeError('OpenCV 4.13 requires NumPy >=2. Preserve the experiment environment; '
                               'review dependencies before setup. No packages were changed.')
        with tempfile.TemporaryDirectory(prefix='dsa4266_opencv_') as folder:
            subprocess.run([sys.executable, '-m', 'pip', 'download', '--no-deps',
                            '--only-binary=:all:', '--dest', folder,
                            OPENCV_PACKAGE+'=='+OPENCV_VERSION], check=True)
            wheels = list(Path(folder).glob('*.whl'))
            if len(wheels) != 1:
                raise RuntimeError('Expected one compatible OpenCV wheel; existing packages preserved')
            # Uninstall all existing providers first: they share the same cv2 files.
            if versions:
                subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', *versions], check=True)
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', str(wheels[0])], check=True)
        importlib.invalidate_caches()
        if _opencv_distributions() != {OPENCV_PACKAGE: OPENCV_VERSION}:
            raise RuntimeError('OpenCV provider verification failed; stop and repair setup before training')
    import cv2
    if cv2.__version__ != '4.13.0' or not hasattr(cv2, 'FaceDetectorYN'):
        raise RuntimeError('Pinned OpenCV/YuNet import check failed; restart and repair setup before training')
    return {OPENCV_PACKAGE: OPENCV_VERSION, 'cv2': cv2.__version__}


def ensure_landmark_runtime(*, install=False):
    """Add pinned landmarks without changing existing NumPy/PyTorch packages.

    The single contrib provider satisfies MediaPipe's declared OpenCV dependency.
    Supporting packages already installed are preserved; missing ones are explicit.
    """
    import importlib.metadata
    import subprocess
    import sys
    runtime = ensure_opencv_runtime(install=install)
    missing = []
    for spec in LANDMARK_PACKAGES:
        name, expected = spec.split('==')
        try: actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: actual = None
        if actual is None or name == 'mediapipe' and actual != expected: missing.append(spec)
    if missing:
        if not install: raise RuntimeError('Landmark packages unavailable; run ensure_landmark_runtime(install=True) in setup')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-cache-dir', *missing], check=True)
    if _opencv_distributions() != {OPENCV_PACKAGE: OPENCV_VERSION}:
        raise RuntimeError('OpenCV providers changed during landmark setup; stop before training')
    return dict(runtime, **{spec.split('==')[0]: importlib.metadata.version(spec.split('==')[0])
                            for spec in LANDMARK_PACKAGES})


class DenseLandmarker:
    """Process-local CPU IMAGE task: no tracking state or order-dependent smoothing."""
    def __init__(self, weights, config):
        import mediapipe as mp
        if mp.__version__ != LANDMARK_VERSION:
            raise RuntimeError('Use mediapipe=='+LANDMARK_VERSION+' with the existing pinned OpenCV (see setup)')
        ensure_landmarker(weights)
        self.mp = mp
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(weights), delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=config.max_faces,
            min_face_detection_confidence=config.min_detection_confidence,
            min_face_presence_confidence=config.min_presence_confidence,
            output_face_blendshapes=False, output_facial_transformation_matrixes=False)
        self.task = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect(self, image):
        result = self.task.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                                data=np.ascontiguousarray(image)))
        h, w = image.shape[:2]
        return [np.array([(p.x*w, p.y*h) for p in face], dtype=np.float32) for face in result.face_landmarks]

    def close(self):
        if self.task is not None:
            self.task.close(); self.task = None


def _rgb(image):
    if image.shape != (224, 224, 3) or image.dtype != np.uint8:
        raise ValueError('Masking requires the existing native 224x224 uint8 RGB frame; no resizing or warping is allowed')


def _valid_landmarks(points):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 468 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('Expected finite dense face landmarks in original-frame pixel coordinates')
    return points


def select_landmarks(faces, bbox):
    """Match dense landmarks to the unchanged YuNet primary face, never to a label."""
    if bbox is None:
        return None, 'no_face'
    x, y, w, h = bbox
    choices = []
    for face in faces:
        try:
            points = _valid_landmarks(face)
        except ValueError:
            continue
        lower, upper = points[list(FACE_OVAL)].min(0), points[list(FACE_OVAL)].max(0)
        intersection = np.maximum(0, np.minimum(upper, [x+w, y+h])-np.maximum(lower, [x, y])).prod()
        area = np.maximum(0, upper-lower).prod()
        union = area+w*h-intersection
        iou = float(intersection/union) if union > 0 else 0.
        distance = float(np.square((lower+upper)/2-[x+w/2, y+h/2]).sum())
        choices.append(((-iou, distance, tuple(lower), tuple(upper), tuple(points.ravel())), points))
    if not choices:
        return None, 'no_landmarks' if not faces else 'invalid_landmarks'
    score, points = min(choices, key=lambda pair: pair[0])
    return (points, None) if -score[0] >= .1 else (None, 'unmatched_landmarks')


def face_alpha(points, config=MaskConfig()):
    """Filled full oval + small margin. Feather extends OUTWARD; interior stays 1."""
    import cv2
    config.validate(); points = _valid_landmarks(points)
    # Include all surface landmarks as well as the full oval to protect eyebrows,
    # lips and nose in pose variations; never form a mouth-only/central mask.
    hull = cv2.convexHull(np.rint(points[:468]).astype(np.int32))
    span = np.ptp(points[list(FACE_OVAL)], axis=0)
    if min(span) < 8 or cv2.contourArea(hull) < 64:
        raise ValueError('Degenerate facial oval')
    margin = max(1, math.ceil(float(max(span))*config.margin_fraction))
    core = np.zeros((224, 224), np.uint8); cv2.fillConvexPoly(core, hull, 1)
    core = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*margin+1, 2*margin+1)))
    outside_distance = cv2.distanceTransform(1-core, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    t = np.clip(outside_distance/config.feather_pixels, 0, 1)
    alpha = ((1+np.cos(np.pi*t))*.5).astype(np.float32); alpha[core == 1] = 1.
    return alpha, margin


def clipping_diagnostics(points, alpha, crop_bounds):
    """Geometric landmark coverage only; cannot certify hidden/unobserved anatomy."""
    points = _valid_landmarks(points)
    x1, y1, x2, y2 = crop_bounds
    regions, crop_regions = {}, {}
    for name, ids in REGIONS.items():
        coordinates = points[list(ids)]
        inside = (coordinates[:, 0] >= 0) & (coordinates[:, 0] < 224) & (coordinates[:, 1] >= 0) & (coordinates[:, 1] < 224)
        pixels = np.clip(np.rint(coordinates).astype(int), 0, 223)
        regions[name] = bool(inside.all() and np.all(alpha[pixels[:, 1], pixels[:, 0]] == 1))
        crop_regions[name] = not bool(((coordinates[:, 0] >= x1) & (coordinates[:, 0] < x2) &
                                      (coordinates[:, 1] >= y1) & (coordinates[:, 1] < y2)).all())
    return {'regions_fully_preserved': regions, 'facial_regions_clipped': not all(regions.values()),
            'crop_regions_clipped': crop_regions, 'crop_facial_regions_clipped': any(crop_regions.values()),
            'face_touches_source_edge': bool(((points[:468] <= 1) | (points[:468] >= 223)).any()),
            'coverage_check': 'landmark geometry, not an anatomical guarantee or occlusion detector'}


def masked_frame(image, points, crop_metadata, config=MaskConfig(), *, failure=None):
    """RGB-only masking; exact original interior pixels, no resize/repositioning."""
    _rgb(image); config.validate()
    details = {'mode': 'face_mask', 'output_size': [224, 224], 'spatial_scale': 1.,
               'yunet_status': crop_metadata['status'], 'confidence': crop_metadata['confidence'],
               'crop_xyxy': crop_metadata['crop_xyxy'], 'landmarks_detected': points is not None,
               'facial_regions_clipped': None, 'crop_facial_regions_clipped': None}
    if points is not None:
        try:
            alpha, margin = face_alpha(points, config)
            details.update(clipping_diagnostics(points, alpha, crop_metadata['crop_xyxy']))
        except ValueError:
            failure = 'invalid_landmarks'
    else:
        failure = failure or 'no_landmarks'
    if failure:
        details.update(status='fallback_'+failure, fallback='full_frame', retained_area_fraction=1.,
                       fully_preserved_area_fraction=1., nonneutral_support_fraction=1., margin_pixels=None,
                       oval_points_px=None)
        return image.copy(), np.ones((224, 224), np.float32), details
    blend = image.astype(np.float32)*alpha[..., None]+np.array(config.neutral_rgb, np.float32)*(1-alpha[..., None])
    output = np.rint(blend).astype(np.uint8)
    details.update(status='masked', fallback=None, retained_area_fraction=float(alpha.mean()),
        fully_preserved_area_fraction=float((alpha == 1).mean()), nonneutral_support_fraction=float((alpha > 0).mean()),
        margin_pixels=margin, oval_points_px=np.asarray(points).tolist(),
        margin_or_feather_touches_edge=bool((alpha[0] > 0).any() or (alpha[-1] > 0).any() or (alpha[:, 0] > 0).any() or (alpha[:, -1] > 0).any()))
    return np.ascontiguousarray(output), alpha, details


class MaskReader(FaceReader):
    def __init__(self, root, pipeline, cache_mib=0):
        super().__init__(root, pipeline.weights, pipeline.cache_path, pipeline.identity, pipeline.config, cache_mib)
        self.landmark_weights, self.mask_config = str(pipeline.landmark_weights), pipeline.mask_config
        self.landmarker = None

    def transform(self, record, image):
        frame_key(record); _rgb(image)
        metadata = self.boxes.get(record)
        if metadata is None:
            if self.detector is None: self.detector = YuNet(self.weights, self.config)
            crop = self.detector.detect(image)
            points, reason = None, 'no_face'
            if crop['bbox_xywh'] is not None:
                if self.landmarker is None: self.landmarker = DenseLandmarker(self.landmark_weights, self.mask_config)
                points, reason = select_landmarks(self.landmarker.detect(image), crop['bbox_xywh'])
            output, alpha, metadata = masked_frame(image, points, crop, self.mask_config, failure=reason)
            self.boxes.put(record, metadata)  # fallback reason recorded per source frame, never dropped
        else:
            points = metadata['oval_points_px']
            crop = {'status': metadata['yunet_status'], 'confidence': metadata['confidence'], 'crop_xyxy': metadata['crop_xyxy']}
            reason = metadata['status'].removeprefix('fallback_') if metadata['fallback'] else None
            output, alpha, _ = masked_frame(image, points, crop, self.mask_config, failure=reason)
        return output, alpha, metadata

    def read(self, records):
        for record in records: frame_key(record)  # guard BEFORE any decode
        originals = self.original(records)
        return [[self.transform(record, image[0])[0]] for record, image in zip(records, originals)]

    def close(self):
        self.boxes.close()
        if self.landmarker is not None: self.landmarker.close(); self.landmarker = None

    def __getstate__(self):
        from collections import OrderedDict
        state = dict(self.__dict__)
        state.update(detector=None, landmarker=None, cache=OrderedDict(), bytes=0)
        return state


class MaskDataset(FaceDataset):
    def __init__(self, records, data_root, pipeline, seed=4266, epoch=0, flip_probability=0., cache_mib=0):
        super().__init__(records, data_root, pipeline, seed, epoch, flip_probability, cache_mib)
        self.reader = MaskReader(data_root, pipeline, cache_mib)


class FaceMaskPipeline(FacePipeline):
    def __init__(self, weights, landmark_weights, cache_path, index_sha256, pool_sha256, mask_config=MaskConfig()):
        import cv2
        self.weights, self.landmark_weights, self.cache_path = Path(weights), Path(landmark_weights), Path(cache_path)
        self.config, self.mask_config = CropConfig(margin=.30).validate(), mask_config.validate()
        ensure_detector(self.weights); ensure_landmarker(self.landmark_weights)
        self.identity = {'kind': 'face_mask', 'version': MASK_VERSION, 'mask': asdict(mask_config),
            'crop': asdict(self.config), 'detector_sha256': digest(self.weights), 'landmarker_sha256': LANDMARK_SHA256,
            'mediapipe': LANDMARK_VERSION, 'opencv': cv2.__version__,
            'index_sha256': index_sha256, 'pool_sha256': pool_sha256}
        # JSON-normalize tuples so checkpoint/JSON/cache identities compare identically.
        self.identity = json.loads(json.dumps(self.identity))
        cache = BoxCache(self.cache_path, self.identity); cache.connection(); cache.close()

    def dataset(self, records, data_root, config, training=False, epoch=0, cache_mib=None):
        return MaskDataset(records, data_root, self, config.seed, epoch,
            config.flip_probability if training else 0., config.cache_mib if cache_mib is None else cache_mib)

    def snapshot(self, out):
        """Stream diagnostic metadata; optional geometry snapshot, same mask pixels."""
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        cache=BoxCache(self.cache_path,self.identity)
        counts=Counter();splits={s:{'frames':0,'fallback_count':0,'clipping_warning_count':0} for s in ('train','dev')}
        area=0.;total=0
        try:
            for split,payload in cache.connection().execute('SELECT split,metadata FROM boxes'):
                row=json.loads(payload);total+=1;counts[row['status']]+=1;area+=row['retained_area_fraction']
                sub=splits[split];sub['frames']+=1;sub['fallback_count']+=int(row['fallback'] is not None)
                sub['clipping_warning_count']+=int(row.get('facial_regions_clipped') is True)
            if getattr(self,'include_geometry_cache',False):cache.snapshot(out/'face_boxes.sqlite')
        finally:cache.close()
        report={'mode':'face_mask','input_pipeline':self.identity,'cached_unique_frames':total,
                'status_counts':dict(counts),'by_split':splits,'dev_frames_excluded':0,
                'mean_retained_area_fraction':area/total if total else None,
                'geometry_cache_exported':bool(getattr(self,'include_geometry_cache',False)),
                'cache_note':'Optional cache; exact recorded geometry retained only when included. Same pinned assets/policy required to recompute.',
                'test_used':False}
        (out/'face_detection_report.json').write_text(json.dumps(report,indent=2))
        for source,name in [(self.weights,'face_detector.onnx'),(self.landmark_weights,'face_landmarker.task'),
                            (Path(__file__).with_name('YUNET_LICENSE.txt'),'face_detector_LICENSE.txt')]:
            if not (out/name).exists():shutil.copy2(source,out/name)
