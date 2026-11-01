import csv
import json
from pathlib import Path

import numpy as np
import torch

from e2b.train import ConcatClassifier, normalize
from e2b.io import code_hashes

ROOT = Path("/home/evan/dsa4266/experiment_2b")

FEATURE_ROOT = Path(
    "/home/evan/dsa4266/experiment_2b_outputs/new5000/"
    "clip_visual_features/"
    "vit_b32_lower_lr_10fps_a69a1ddcf56d_0576c2b0"
)

METADATA = Path(
    "/mnt/c/Users/wuche/Downloads/"
    "LAV-DF-5000-NEW/metadata.new5000.json"
)

CHECKPOINT = ROOT / "runs/visual_only_clip_any_s42/best.pt"
OUTPUT = ROOT / "audits/region_audit_new5000.csv"

TOLERANCE = 0.055
MIN_FRAMES = 3


def load_features(entry):
    with np.load(
        FEATURE_ROOT / entry["feature_path"],
        allow_pickle=False
    ) as z:
        x = np.asarray(z["features"], dtype=np.float32)
        t = np.asarray(z["timestamp_s"], dtype=np.float32)

    assert x.ndim == 2 and x.shape[1] == 512
    assert len(x) == len(t) and len(x) > 0
    assert np.isfinite(x).all() and np.isfinite(t).all()
    assert np.all(np.diff(t) >= 0)

    return x, t


def match_timestamps(a, b):
    i = j = 0
    ia, ib = [], []

    while i < len(a) and j < len(b):
        difference = float(a[i] - b[j])

        if abs(difference) <= TOLERANCE:
            ia.append(i)
            ib.append(j)
            i += 1
            j += 1
        elif difference < 0:
            i += 1
        else:
            j += 1

    return np.asarray(ia, dtype=int), np.asarray(ib, dtype=int)


def region_masks(t, periods):
    periods = sorted(
        (float(a), float(b)) for a, b in periods
    )
    assert periods and all(b > a for a, b in periods)

    start = min(a for a, _ in periods)
    end = max(b for _, b in periods)

    during = np.zeros(len(t), dtype=bool)

    for a, b in periods:
        during |= (t >= a) & (t < b)

    return {
        "before": t < start,
        "during": during,
        "after": t >= end,
    }


def cosine_rows(a, b):
    numerator = np.sum(a * b, axis=1)
    denominator = (
        np.linalg.norm(a, axis=1) *
        np.linalg.norm(b, axis=1)
    )
    return numerator / np.maximum(denominator, 1e-12)


def main():
    metadata = json.loads(METADATA.read_text())
    records = {
        r["file"]: r for r in metadata
        if r["split"] == "dev"
    }

    index = {}
    with (FEATURE_ROOT / "index.jsonl").open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            assert row["file"] not in index
            index[row["file"]] = row

    assert len(records) == len(index) == 500
    assert set(records) == set(index)

    ckpt = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    assert ckpt["settings"]["mode"] == "visual"
    assert ckpt["pooling"] == "mean"
    assert ckpt["code_hashes"] == code_hashes()

    expected_sha = ckpt["encoder_provenance"]["visual"][
        "engine"
    ]["checkpoint_sha256"]

    assert all(
        row["checkpoint_sha256"] == expected_sha
        for row in index.values()
    )

    s = ckpt["settings"]

    model = ConcatClassifier(
        ckpt["dims"]["visual"],
        ckpt["dims"]["audio"],
        "visual",
        s["hidden"],
        s["dropout"]
    )

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    stats = ckpt["normalization"]["visual"]

    def score(x):
        pooled = x.mean(axis=0).astype(np.float32)
        normalized = normalize(pooled[None, :], stats)

        with torch.inference_mode():
            return float(model(normalized, None).item())

    results = []

    for row in records.values():
        if not (
            row["modify_audio"] is True
            and row["modify_video"] is False
        ):
            continue

        original = row["original"]
        assert original in records

        source = records[original]
        assert not source["modify_audio"]
        assert not source["modify_video"]

        x0, t0 = load_features(index[original])
        x1, t1 = load_features(index[row["file"]])

        i0, i1 = match_timestamps(t0, t1)

        assert len(i0) > 0

        # Region membership uses the manipulated video's
        # annotated audio manipulation intervals.
        matched_times = t1[i1]
        masks = region_masks(
            matched_times,
            row["fake_periods"]
        )

        for region, mask in masks.items():
            a = x0[i0][mask]
            b = x1[i1][mask]

            result = {
                "original": original,
                "audio_only": row["file"],
                "region": region,
                "matched_frames": len(a),
                "usable": len(a) >= MIN_FRAMES,
                "mean_cosine": float("nan"),
                "median_cosine": float("nan"),
                "pooled_l2": float("nan"),
                "logit_delta": float("nan"),
            }

            if len(a) >= MIN_FRAMES:
                cosines = cosine_rows(a, b)

                result["mean_cosine"] = float(
                    np.mean(cosines)
                )
                result["median_cosine"] = float(
                    np.median(cosines)
                )
                result["pooled_l2"] = float(
                    np.linalg.norm(
                        b.mean(axis=0) - a.mean(axis=0)
                    )
                )
                result["logit_delta"] = (
                    score(b) - score(a)
                )

            results.append(result)

    assert len(results) == 125 * 3

    if OUTPUT.exists():
        raise FileExistsError(
            f"Output exists: {OUTPUT}. "
            "Do not overwrite an existing audit."
        )

    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print("Audio-only pairs:", len(results) // 3)

    for region in ("before", "during", "after"):
        subset = [
            r for r in results
            if r["region"] == region and r["usable"]
        ]

        print(f"\n=== {region.upper()} ===")
        print("Usable pairs:", len(subset))

        if not subset:
            continue

        cos = np.asarray([
            r["mean_cosine"] for r in subset
        ])
        delta = np.asarray([
            r["logit_delta"] for r in subset
        ])

        print("Median frame cosine:", float(np.median(cos)))
        print("Median logit delta:", float(np.median(delta)))
        print("Positive logit deltas:", int((delta > 0).sum()))

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
