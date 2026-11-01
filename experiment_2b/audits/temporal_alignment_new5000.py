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
OUTPUT = ROOT / "audits/temporal_alignment_new5000.csv"
TOLERANCE = 0.055


def load_features(entry):
    with np.load(
        FEATURE_ROOT / entry["feature_path"],
        allow_pickle=False,
    ) as data:
        x = data["features"].astype(np.float32)
        t = data["timestamp_s"].astype(np.float32)

    assert x.ndim == 2 and x.shape[1] == 512
    assert len(x) == len(t) and len(x) > 0
    assert np.isfinite(x).all() and np.isfinite(t).all()
    assert np.all(np.diff(t) >= 0)
    return x, t


def match_timestamps(a, b, tolerance):
    """Greedy one-to-one matching in timestamp order."""
    i = j = 0
    ia, ib = [], []

    while i < len(a) and j < len(b):
        difference = float(a[i] - b[j])

        if abs(difference) <= tolerance:
            ia.append(i)
            ib.append(j)
            i += 1
            j += 1
        elif difference < 0:
            i += 1
        else:
            j += 1

    return np.asarray(ia), np.asarray(ib)


def main():
    metadata = json.loads(METADATA.read_text())
    records = {
        r["file"]: r for r in metadata if r["split"] == "dev"
    }

    index = {}
    with (FEATURE_ROOT / "index.jsonl").open() as f:
        for line in f:
            entry = json.loads(line)
            assert entry["file"] not in index
            index[entry["file"]] = entry

    assert len(records) == len(index) == 500
    assert set(records) == set(index)

    ckpt = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=True
    )
    assert ckpt["settings"]["mode"] == "visual"
    assert ckpt["pooling"] == "mean"
    assert ckpt["code_hashes"] == code_hashes()

    settings = ckpt["settings"]
    model = ConcatClassifier(
        ckpt["dims"]["visual"],
        ckpt["dims"]["audio"],
        "visual",
        settings["hidden"],
        settings["dropout"],
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    stats = ckpt["normalization"]["visual"]

    def logit(x):
        pooled = x.mean(axis=0).astype(np.float32)
        v = normalize(pooled[None, :], stats)
        with torch.inference_mode():
            return float(model(v, None).item())

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

        full_delta = logit(x1) - logit(x0)

        # Control A: common timestamp range.
        end = min(float(t0[-1]), float(t1[-1]))
        c0 = x0[t0 <= end]
        c1 = x1[t1 <= end]
        common_delta = logit(c1) - logit(c0)

        # Control B: one-to-one timestamp matching.
        i0, i1 = match_timestamps(t0, t1, TOLERANCE)

        if len(i0) == 0:
            matched_delta = float("nan")
        else:
            matched_delta = logit(x1[i1]) - logit(x0[i0])

        results.append({
            "original": original,
            "audio_only": row["file"],
            "original_steps": len(t0),
            "fake_steps": len(t1),
            "matched_steps": len(i0),
            "matched_fraction_original": len(i0) / len(t0),
            "matched_fraction_fake": len(i1) / len(t1),
            "full_delta": full_delta,
            "common_delta": common_delta,
            "matched_delta": matched_delta,
        })

    assert len(results) == 125

    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} already exists. Preserve previous results."
        )

    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("Pairs:", len(results))

    for key in ("full_delta", "common_delta", "matched_delta"):
        values = np.asarray([r[key] for r in results])
        valid = values[np.isfinite(values)]

        print(f"\n{key}")
        print("Valid pairs:", len(valid))
        print("Positive deltas:", int((valid > 0).sum()))
        print("Median:", float(np.median(valid)))
        print("Mean:", float(np.mean(valid)))

    fractions = [
        r["matched_fraction_original"] for r in results
    ]
    print("\nMedian matched fraction:", np.median(fractions))
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
