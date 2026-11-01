import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = os.environ["LAVDF_METADATA"]
CHECKPOINT_PATH = os.environ.get("CLIP_CHECKPOINT", str(PROJECT_ROOT / "best.pt"))
OUTPUT_ROOT = PROJECT_ROOT / "experiment_2b_outputs/new5000_v2_dev"

# ============================================================
# Experiment 2B: CLIP Video Feature Extraction
# Local LAV-DF-5000 | CLIP ViT-B/32 Lower-LR Checkpoint
# ============================================================

from pathlib import Path
from collections import Counter
import hashlib
import json
import math

import av
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from transformers import (
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)
from tqdm.auto import tqdm

# ------------------------------------------------------------
# 1. Verify paths and load the local video metadata
# ------------------------------------------------------------

METADATA_PATH = Path(METADATA_PATH)
CHECKPOINT_PATH = Path(CHECKPOINT_PATH)
OUTPUT_ROOT = Path(OUTPUT_ROOT)

assert METADATA_PATH.is_file(), f"Missing metadata: {METADATA_PATH}"
assert CHECKPOINT_PATH.is_file(), f"Missing checkpoint: {CHECKPOINT_PATH}"

VIDEO_ROOT = METADATA_PATH.parent

with METADATA_PATH.open("r", encoding="utf-8") as f:
    video_metadata = json.load(f)

assert isinstance(video_metadata, list), (
    "Expected metadata.min.json to contain a list of video records."
)

required_fields = {"file", "split"}
for i, item in enumerate(video_metadata):
    missing = required_fields - item.keys()
    assert not missing, (
        f"Metadata record {i} is missing fields: {missing}"
    )

split_counts = Counter(item["split"] for item in video_metadata)

print("Video root:", VIDEO_ROOT)
print("Metadata records:", len(video_metadata))
print("Video splits:", dict(split_counts))

# Process all three splits; do not change their membership.
EXPECTED_SPLITS = {"train", "dev", "test"}
assert set(split_counts) == EXPECTED_SPLITS, (
    f"Expected {EXPECTED_SPLITS}; found {set(split_counts)}"
)

# ------------------------------------------------------------
# 2. Load the exact trained CLIP model
# ------------------------------------------------------------

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
)

assert checkpoint.get("protocol") == "clip_shared_fixed20k_pts10_v1"
assert checkpoint["config"]["pretrained_model"] == MODEL_NAME

state_dict = checkpoint["model"]

assert state_dict["classifier.weight"].shape == (2, 512)
assert state_dict["classifier.bias"].shape == (2,)

class CLIPBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        config = CLIPVisionConfig.from_dict(
            checkpoint["encoder_config"]
        )

        self.encoder = CLIPVisionModelWithProjection(config)
        self.encoder.set_attn_implementation("eager")

        self.classifier = nn.Linear(512, 2)

    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values=pixel_values)
        features = F.normalize(
            outputs.image_embeds, p=2, dim=-1
        )
        return self.classifier(features)


model = CLIPBinaryClassifier()
model.load_state_dict(state_dict, strict=True)
model = model.to(DEVICE).eval()

image_processor = CLIPImageProcessor.from_dict(
    checkpoint["processor_config"]
)

assert image_processor.crop_size == {
    "height": 224, "width": 224
}
assert image_processor.size == {
    "shortest_edge": 224
}

print("CLIP checkpoint loaded:", CHECKPOINT_PATH)
print("Best epoch:", checkpoint.get("best_epoch"))
print("Device:", DEVICE)

# ------------------------------------------------------------
# 3. Configure the embedding export
# ------------------------------------------------------------

TARGET_FPS = 10.0
BATCH_SIZE = 32

# Include checkpoint and metadata hashes to avoid mixing caches
# created from different checkpoints or dataset manifests.
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


checkpoint_sha256 = sha256_file(CHECKPOINT_PATH)
metadata_sha256 = sha256_file(METADATA_PATH)

EXPORT_ROOT = (
    OUTPUT_ROOT
    / "clip_visual_features"
    / f"vit_b32_shared_fixed20k_10fps_{checkpoint_sha256[:12]}_{metadata_sha256[:8]}"
)

FEATURE_ROOT = EXPORT_ROOT / "features"
FEATURE_ROOT.mkdir(parents=True, exist_ok=True)

INDEX_PATH = EXPORT_ROOT / "index.jsonl"
ERRORS_PATH = EXPORT_ROOT / "errors.json"

# Save export settings for provenance.
config = {
    "model_name": MODEL_NAME,
    "checkpoint_path": str(CHECKPOINT_PATH),
    "checkpoint_sha256": checkpoint_sha256,
    "metadata_path": str(METADATA_PATH),
    "metadata_sha256": metadata_sha256,
    "sampling_fps": TARGET_FPS,
    "feature_dimension": 512,
    "feature_layer": "normalized encoder.image_embeds",
    "timestamp_reference": "seconds relative to first decoded video frame",
    "splits": ["train", "dev", "test"],
}

with (EXPORT_ROOT / "export_config.json").open("w") as f:
    json.dump(config, f, indent=2)

# ------------------------------------------------------------
# 4. Encode batches of frames into normalized 512-D embeddings
# ------------------------------------------------------------

def encode_images(images):
    inputs = image_processor(
        images=images,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"].to(DEVICE)

    # The embeddings are taken BEFORE the fake/real classifier.
    with torch.inference_mode():
        image_embeds = model.encoder(
            pixel_values=pixel_values
        ).image_embeds

        image_embeds = F.normalize(
            image_embeds.float(),
            p=2,
            dim=-1,
        )

    result = image_embeds.cpu().numpy().astype(np.float32)

    assert result.ndim == 2 and result.shape[1] == 512
    assert np.isfinite(result).all()

    return result


# ------------------------------------------------------------
# 5. Decode one video and sample at approximately 10 FPS
# ------------------------------------------------------------

def extract_video_features(video_path):
    feature_chunks = []
    sampled_timestamps = []

    pending_images = []
    pending_timestamps = []

    def flush_batch():
        if not pending_images:
            return

        feature_chunks.append(encode_images(pending_images))
        sampled_timestamps.extend(pending_timestamps)

        pending_images.clear()
        pending_timestamps.clear()

    with av.open(str(video_path)) as container:
        video_streams = [
            stream for stream in container.streams
            if stream.type == "video"
        ]

        if not video_streams:
            raise RuntimeError("No video stream found.")

        stream = video_streams[0]

        first_pts_time = None
        next_sample_time = 0.0
        sample_interval = 1.0 / TARGET_FPS

        for frame in container.decode(stream):
            if frame.time is None:
                continue

            absolute_time = float(frame.time)

            if first_pts_time is None:
                first_pts_time = absolute_time

            relative_time = absolute_time - first_pts_time

            # Select the first decoded frame at or after each
            # scheduled sample time; do not use manipulation labels.
            if relative_time + 1e-9 < next_sample_time:
                continue

            rgb = frame.to_ndarray(format="rgb24")
            pending_images.append(Image.fromarray(rgb))
            pending_timestamps.append(relative_time)

            # Advance to the next sample tick without duplicating
            # a frame if the decoded timestamp is slightly late.
            next_sample_time = (
                math.floor(relative_time * TARGET_FPS + 1e-9) + 1
            ) * sample_interval

            if len(pending_images) >= BATCH_SIZE:
                flush_batch()

        flush_batch()

    if not feature_chunks:
        raise RuntimeError("No frames could be decoded and sampled.")

    features = np.concatenate(feature_chunks, axis=0)
    timestamps = np.asarray(sampled_timestamps, dtype=np.float32)

    assert features.shape == (len(timestamps), 512)
    assert np.isfinite(features).all()
    assert np.isfinite(timestamps).all()
    assert np.all(np.diff(timestamps) >= 0)

    return features, timestamps


# ------------------------------------------------------------
# 6. Extract all splits, with safe per-video resume
# ------------------------------------------------------------

all_records = sorted(
    [
        item for item in video_metadata
        if item["split"] == "dev"
    ],
    key=lambda item: item["file"],
)

assert len(all_records) == 500, (
    f"Expected 500 DEV videos, found {len(all_records)}"
)

print("Selected DEV videos:", len(all_records))

index_rows = []
errors = []

for item in tqdm(all_records, desc="Extracting CLIP features"):
    video_relpath = Path(item["file"])
    video_path = VIDEO_ROOT / video_relpath

    # Mirror split/subfolder structure; each video gets its own NPZ.
    feature_relpath = (
        Path("features")
        / video_relpath.with_suffix(".npz")
    )
    feature_path = EXPORT_ROOT / feature_relpath
    feature_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Reuse only a complete, valid cache from this export directory.
        if feature_path.is_file():
            try:
                with np.load(feature_path, allow_pickle=False) as saved:
                    cached_features = saved["features"]
                    cached_timestamps = saved["timestamp_s"]

                    valid_cache = (
                        cached_features.ndim == 2
                        and cached_features.shape[1] == 512
                        and len(cached_features) == len(cached_timestamps)
                        and len(cached_timestamps) > 0
                        and np.isfinite(cached_features).all()
                        and np.isfinite(cached_timestamps).all()
                    )

                if valid_cache:
                    index_rows.append({
                        "file": item["file"],
                        "split": item["split"],
                        "feature_path": feature_relpath.as_posix(),
                        "checkpoint_sha256": checkpoint_sha256,
                    })
                    continue
            except Exception:
                pass

        features, timestamps = extract_video_features(video_path)

        # Atomic save: a partially written NPZ will not appear complete.
        temporary_path = feature_path.with_name(
            feature_path.name + ".tmp"
        )

        with temporary_path.open("wb") as f:
            np.savez_compressed(
                f,
                features=features,
                timestamp_s=timestamps,
            )

        temporary_path.replace(feature_path)

        index_rows.append({
            "file": item["file"],
            "split": item["split"],
            "feature_path": feature_relpath.as_posix(),
            "checkpoint_sha256": checkpoint_sha256,
        })

    except Exception as exc:
        errors.append({
            "file": item["file"],
            "split": item["split"],
            "error": repr(exc),
        })

# Do not publish a complete index if any video failed.
if errors:
    with ERRORS_PATH.open("w") as f:
        json.dump(errors, f, indent=2)

    raise RuntimeError(
        f"{len(errors)} videos failed. See {ERRORS_PATH}. "
        "Fix the errors and rerun; completed valid videos can be reused."
    )

# Write the index only after every video has completed successfully.
with INDEX_PATH.open("w", encoding="utf-8") as f:
    for row in index_rows:
        f.write(json.dumps(row) + "\n")

print("\nCLIP feature extraction complete.")
print("Videos exported:", len(index_rows))
print("Split counts:", dict(Counter(row["split"] for row in index_rows)))
print("Feature root:", FEATURE_ROOT)
print("Index:", INDEX_PATH)
print("Export config:", EXPORT_ROOT / "export_config.json")