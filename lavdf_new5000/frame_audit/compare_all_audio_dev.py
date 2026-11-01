import csv
import json
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

DATA_ROOT = Path("/mnt/c/Users/wuche/Downloads/LAV-DF-5000-NEW")
METADATA = DATA_ROOT / "metadata.new5000.json"
OUTPUT = Path("/home/evan/dsa4266/lavdf_new5000/frame_audit/results_all_audio_dev")
OUTPUT.mkdir(parents=True, exist_ok=True)

PAIRS = None

TOLERANCE = 0.055


def decode_video(path):
    frames, timestamps = [], []

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        first_time = None

        for frame in container.decode(stream):
            if frame.time is None:
                continue

            t = float(frame.time)
            if first_time is None:
                first_time = t

            frames.append(frame.to_ndarray(format="rgb24"))
            timestamps.append(t - first_time)

    return np.asarray(frames), np.asarray(timestamps)


def match_frames(t0, t1):
    i = j = 0
    matches = []

    while i < len(t0) and j < len(t1):
        difference = t0[i] - t1[j]

        if abs(difference) <= TOLERANCE:
            matches.append((i, j))
            i += 1
            j += 1
        elif difference < 0:
            i += 1
        else:
            j += 1

    return matches


def get_region(t, periods):
    starts = [p[0] for p in periods]
    ends = [p[1] for p in periods]

    if t < min(starts):
        return "before"
    if t >= max(ends):
        return "after"

    for a, b in periods:
        if a <= t < b:
            return "during"

    return "between"


def metrics(a, b):
    assert a.shape == b.shape

    mae = np.mean(
        np.abs(a.astype(np.float32) - b.astype(np.float32))
    )

    ssim = structural_similarity(
        a, b,
        channel_axis=2,
        data_range=255,
    )

    psnr = peak_signal_noise_ratio(
        a, b,
        data_range=255,
    )

    return float(mae), float(ssim), float(psnr)


def comparison_image(a, b, label):
    difference = np.abs(
        a.astype(np.int16) - b.astype(np.int16)
    )

    amplified = np.clip(difference * 5, 0, 255).astype(np.uint8)

    images = [
        Image.fromarray(a),
        Image.fromarray(b),
        Image.fromarray(amplified),
    ]

    width, height = images[0].size
    canvas = Image.new("RGB", (width * 3, height + 35), "white")
    draw = ImageDraw.Draw(canvas)

    for i, image in enumerate(images):
        canvas.paste(image, (i * width, 35))

    for i, title in enumerate(
        ["Original", "Audio-only", "Difference x5"]
    ):
        draw.text((i * width + 8, 10), title, fill="black")

    return canvas


def main():
    metadata = json.loads(METADATA.read_text())
    lookup = {r["file"]: r for r in metadata}
    results = []

    pairs = sorted(
        [
            (r["original"], r["file"])
            for r in metadata
            if r["split"] == "dev"
            and r["modify_audio"] is True
            and r["modify_video"] is False
        ]
    )

    assert len(pairs) == 125
    assert len(set(pairs)) == 125

    for original, fake in pairs:
        assert fake in lookup
        assert lookup[fake]["original"] == original
        assert lookup[fake]["modify_audio"] is True
        assert lookup[fake]["modify_video"] is False

        periods = lookup[fake]["fake_periods"]

        f0, t0 = decode_video(DATA_ROOT / original)
        f1, t1 = decode_video(DATA_ROOT / fake)

        assert f0.shape[1:] == f1.shape[1:], (
            f"Different video dimensions: {original}, {fake}"
        )

        matches = match_frames(t0, t1)
        assert matches, f"No matched frames: {fake}"

        print(f"\n{original} vs {fake}")
        print("Decoded frames:", len(f0), len(f1))
        print("Matched frames:", len(matches))

        region_examples = {}

        for i, j in matches:
            region = get_region(t1[j], periods)

            mae, ssim, psnr = metrics(f0[i], f1[j])

            results.append({
                "original": original,
                "audio_only": fake,
                "region": region,
                "timestamp_original": float(t0[i]),
                "timestamp_fake": float(t1[j]),
                "mae": mae,
                "ssim": ssim,
                "psnr": psnr,
            })

            if region not in region_examples:
                region_examples[region] = (i, j)



    with (OUTPUT / "frame_metrics.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n=== REGION SUMMARY ===")

    for region in ("before", "during", "after"):
        subset = [
            r for r in results
            if r["region"] == region
        ]

        if not subset:
            continue

        print(f"\n{region.upper()}")
        print("Frames:", len(subset))
        print("Median MAE:", np.median([r["mae"] for r in subset]))
        print("Median SSIM:", np.median([r["ssim"] for r in subset]))
        print("Median PSNR:", np.median([r["psnr"] for r in subset]))

    print("\nResults saved to:", OUTPUT)


if __name__ == "__main__":
    main()
