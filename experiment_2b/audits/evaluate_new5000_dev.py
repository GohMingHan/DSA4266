import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path("/home/evan/dsa4266/experiment_2b")
sys.path.insert(0, str(ROOT))

from e2b.train import ConcatClassifier

CHECKPOINT = ROOT / "runs/visual_only_clip_any_s42/best.pt"

METADATA = Path(
    "/mnt/c/Users/wuche/Downloads/"
    "LAV-DF-5000-NEW/metadata.new5000.json"
)

FEATURE_ROOT = Path(
    "/home/evan/dsa4266/experiment_2b_outputs/new5000/"
    "clip_visual_features/"
    "vit_b32_lower_lr_10fps_a69a1ddcf56d_0576c2b0"
)

OUTPUT = ROOT / "evaluations/visual_only_new5000_dev"


def category(row):
    a = row["modify_audio"]
    v = row["modify_video"]

    if a and v:
        return "both"
    if a:
        return "audio_only"
    if v:
        return "video_only"
    return "genuine"


def main():
    if OUTPUT.exists():
        raise RuntimeError(
            f"Output already exists: {OUTPUT}. "
            "Do not overwrite previous evaluations."
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["settings"]["mode"] == "visual"
    assert checkpoint["pooling"] == "mean"

    metadata = json.loads(METADATA.read_text())
    records = {
        row["file"]: row
        for row in metadata
        if row["split"] == "dev"
    }

    index_rows = [
        json.loads(line)
        for line in (FEATURE_ROOT / "index.jsonl").read_text().splitlines()
        if line.strip()
    ]

    assert len(records) == 500
    assert len(index_rows) == 500

    # Verify the exact original CLIP checkpoint identity.
    expected_sha = (
        "a69a1ddcf56d6d85dfa1b3e2058e9778ebe7b305a4e6bfdb2dfa4071f38b52e7"
    )

    assert all(
        row["checkpoint_sha256"] == expected_sha
        for row in index_rows
    )

    features = []
    labels = []
    rows = []

    for entry in sorted(index_rows, key=lambda x: x["file"]):
        name = entry["file"]
        assert name in records

        row = records[name]

        with np.load(
            FEATURE_ROOT / entry["feature_path"],
            allow_pickle=False,
        ) as data:
            sequence = data["features"].astype(np.float32)

        assert sequence.ndim == 2
        assert sequence.shape[1] == 512
        assert np.isfinite(sequence).all()

        pooled = sequence.mean(axis=0)

        features.append(pooled)
        labels.append(
            int(row["modify_audio"] or row["modify_video"])
        )
        rows.append(row)

    V = np.stack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)

    # Apply saved TRAIN normalization, not new DEV statistics.
    norm = checkpoint["normalization"]["visual"]

    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)

    V = (V - mean) / std

    settings = checkpoint["settings"]

    model = ConcatClassifier(
        checkpoint["dims"]["visual"],
        checkpoint["dims"]["audio"],
        settings["mode"],
        settings["hidden"],
        settings["dropout"],
    )

    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    # Audio input is ignored in visual-only mode.
    dummy_audio = np.zeros(
        (len(V), checkpoint["dims"]["audio"]),
        dtype=np.float32,
    )

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)

    with torch.inference_mode():
        logits = model(
            torch.from_numpy(V).to(device),
            torch.from_numpy(dummy_audio).to(device),
        ).reshape(-1).cpu().numpy()

    threshold = float(checkpoint["threshold_logit"])
    predictions = (logits >= threshold).astype(int)

    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))

    metrics = {
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions)),
        "recall": float(recall_score(y, predictions)),
        "f1": float(f1_score(y, predictions)),
        "confusion_matrix": confusion_matrix(
            y, predictions, labels=[0, 1]
        ).tolist(),
        "threshold_logit": threshold,
    }

    print("\n=== NEW DEV RESULTS ===")
    print(json.dumps(metrics, indent=2))

    print("\n=== CATEGORY BREAKDOWN ===")

    for name in (
        "genuine",
        "audio_only",
        "video_only",
        "both",
    ):
        indices = [
            i for i, row in enumerate(rows)
            if category(row) == name
        ]

        positive_rate = float(predictions[indices].mean())

        print(
            f"{name:12s} "
            f"count={len(indices):3d} "
            f"predicted_fake={positive_rate:.4f}"
        )

    OUTPUT.mkdir(parents=True, exist_ok=False)

    (OUTPUT / "metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    with (OUTPUT / "predictions.jsonl").open("w") as f:
        for row, logit, score, pred in zip(
            rows, logits, scores, predictions
        ):
            f.write(json.dumps({
                "file": row["file"],
                "original": row.get("original"),
                "category": category(row),
                "label": int(
                    row["modify_audio"] or row["modify_video"]
                ),
                "logit": float(logit),
                "score": float(score),
                "prediction": int(pred),
            }) + "\n")

    # Compare each audio-only video with its linked original.
    by_file = {
        row["file"]: (row, float(logit))
        for row, logit in zip(rows, logits)
    }

    deltas = []

    for row, logit in zip(rows, logits):
        if category(row) != "audio_only":
            continue

        original = row.get("original")

        if original not in by_file:
            continue

        original_row, original_logit = by_file[original]

        assert category(original_row) == "genuine"

        deltas.append(float(logit - original_logit))

    print("\n=== AUDIO-ONLY PAIRED AUDIT ===")
    print("Pairs:", len(deltas))

    if deltas:
        print("Higher than original:", sum(d > 0 for d in deltas))
        print("Median logit difference:", float(np.median(deltas)))
        print("Mean logit difference:", float(np.mean(deltas)))


if __name__ == "__main__":
    main()