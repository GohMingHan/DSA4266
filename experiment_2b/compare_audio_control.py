import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/evan/dsa4266")
sys.path.insert(0, str(ROOT / "experiment_2b"))

from e2b.train import ConcatClassifier, normalize

EXPORTER = ROOT / "lavdf_new5000/clip_export_new_dev.py"

source = EXPORTER.read_text(encoding="utf-8")

# Execute only the model setup and extraction functions.
# Stop before the full video-processing loop.
marker = "all_records = sorted("
assert source.count(marker) == 1

namespace = {"__name__": "clip_control_exporter"}
exec(source.split(marker)[0], namespace)

extract_video_features = namespace["extract_video_features"]

original = Path(
    "/mnt/c/Users/wuche/Downloads/"
    "LAV-DF-5000-NEW/dev/004080.mp4"
)
modified = ROOT / "lavdf_new5000/controls/audio_changed.mp4"

assert original.is_file()
assert modified.is_file()

x0, t0 = extract_video_features(original)
x1, t1 = extract_video_features(modified)

print("Original feature shape:", x0.shape)
print("Modified feature shape:", x1.shape)
print("Timestamps identical:", np.array_equal(t0, t1))
print("Maximum embedding difference:", float(np.max(np.abs(x0 - x1))))

checkpoint = torch.load(
    ROOT / "experiment_2b/runs/visual_only_clip_any_s42/best.pt",
    map_location="cpu",
    weights_only=True,
)

s = checkpoint["settings"]

model = ConcatClassifier(
    checkpoint["dims"]["visual"],
    checkpoint["dims"]["audio"],
    "visual",
    s["hidden"],
    s["dropout"],
)

model.load_state_dict(checkpoint["model"], strict=True)
model.eval()

stats = checkpoint["normalization"]["visual"]

def predict(features):
    pooled = features.mean(axis=0).astype(np.float32)
    normalized = normalize(pooled[None, :], stats)

    with torch.inference_mode():
        return float(model(normalized, None).item())

z0 = predict(x0)
z1 = predict(x1)

print("Original logit:", z0)
print("Audio-modified logit:", z1)
print("Logit difference:", z1 - z0)
print("Prediction unchanged:", (z0 >= checkpoint["threshold_logit"]) == (z1 >= checkpoint["threshold_logit"]))
