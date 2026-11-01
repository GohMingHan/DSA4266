import csv
import json
from pathlib import Path

import av
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from tqdm import tqdm

ROOT = Path("/mnt/c/Users/wuche/Downloads/LAV-DF-5000-NEW")
OUT = Path.home() / "dsa4266/lavdf_new5000/frame_audit/alignment_results"
OUT.mkdir(parents=True, exist_ok=True)

WINDOW = 5
TIME_TOLERANCE = 0.055


def decode(path):
    frames, times = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        first = None
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            t = float(frame.time)
            if first is None:
                first = t
            frames.append(frame.to_ndarray(format="rgb24"))
            times.append(t - first)
    return np.asarray(frames), np.asarray(times)


def thumbnail(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)


def similarity(a, b):
    return float(structural_similarity(
        a, b, channel_axis=2, data_range=255
    ))


def region(t, periods):
    start = min(float(p[0]) for p in periods)
    end = max(float(p[1]) for p in periods)

    if t < start:
        return "before"
    if t >= end:
        return "after"
    if any(float(a) <= t < float(b) for a, b in periods):
        return "during"
    return "between"


def timestamp_matches(t0, t1):
    i = j = 0
    result = []
    while i < len(t0) and j < len(t1):
        d = float(t0[i] - t1[j])
        if abs(d) <= TIME_TOLERANCE:
            result.append((i, j))
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return result


def content_matches(f0, f1, t0, t1):
    thumbs0 = [thumbnail(x).astype(np.float32) for x in f0]
    thumbs1 = [thumbnail(x).astype(np.float32) for x in f1]

    # Monotonic, one-to-one local matching.
    # Search only near the timestamp-equivalent frame.
    result = []
    last_i = -1

    for j in range(len(f1)):
        center = int(np.argmin(np.abs(t0 - t1[j])))
        low = max(last_i + 1, center - WINDOW)
        high = min(len(f0), center + WINDOW + 1)

        if low >= high:
            continue

        candidates = range(low, high)
        i = min(
            candidates,
            key=lambda k: float(np.mean(
                (thumbs0[k] - thumbs1[j]) ** 2
            ))
        )
        result.append((i, j))
        last_i = i

    return result


def main():
    metadata = json.loads(
        (ROOT / "metadata.new5000.json").read_text()
    )
    pairs = sorted(
        (r["original"], r["file"], r["fake_periods"])
        for r in metadata
        if r["split"] == "dev"
        and r["modify_audio"] is True
        and r["modify_video"] is False
    )
    assert len(pairs) == 125

    rows = []
    coverage = []

    for original, fake, periods in tqdm(pairs, desc="Aligning pairs"):
        f0, t0 = decode(ROOT / original)
        f1, t1 = decode(ROOT / fake)

        assert len(f0) and len(f1)
        assert f0.shape[1:] == f1.shape[1:]

        methods = {
            "timestamp": timestamp_matches(t0, t1),
            "index": [(i, i) for i in range(min(len(f0), len(f1)))],
            "content": content_matches(f0, f1, t0, t1),
        }

        for method, matches in methods.items():
            coverage.append({
                "audio_only": fake,
                "method": method,
                "original_frames": len(f0),
                "fake_frames": len(f1),
                "matched_frames": len(matches),
                "coverage_original": len(matches) / len(f0),
                "coverage_fake": len(matches) / len(f1),
            })

            for i, j in matches:
                rows.append({
                    "audio_only": fake,
                    "original": original,
                    "method": method,
                    "region": region(float(t1[j]), periods),
                    "original_index": i,
                    "fake_index": j,
                    "time_offset_s": float(t0[i] - t1[j]),
                    "index_offset": i - j,
                    "ssim": similarity(f0[i], f1[j]),
                    "mae": float(np.mean(np.abs(
                        f0[i].astype(np.float32)
                        - f1[j].astype(np.float32)
                    ))),
                })

    for filename, data in (
        ("alignment_frames.csv", rows),
        ("alignment_coverage.csv", coverage),
    ):
        path = OUT / filename
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

    print("Completed pairs:", len(pairs))
    print("Frame comparisons:", len(rows))
    print("Output:", OUT)


if __name__ == "__main__":
    main()
