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

from e2b.train import ConcatClassifier, normalize

CHECKPOINT = ROOT / (
    "runs/visual_only_clip_modify_video_new5000_s42/best.pt"
)

METADATA = Path(
    "/mnt/c/Users/wuche/Downloads/"
    "LAV-DF-5000-NEW/metadata.new5000.json"
)

FEATURE_ROOT = Path(
    "/home/evan/dsa4266/experiment_2b_outputs/new5000/"
    "clip_visual_features/"
    "vit_b32_lower_lr_10fps_a69a1ddcf56d_0576c2b0"
)

OUTPUT = ROOT / "evaluations/visual_modify_video_new5000_dev"


def category(row):
    if row["modify_audio"] and row["modify_video"]:
        return "both"
    if row["modify_audio"]:
        return "audio_only"
    if row["modify_video"]:
        return "video_only"
    return "genuine"


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"Evaluation output already exists: {OUTPUT}"
        )

    ckpt = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=True
    )

    assert ckpt["target"] == "modify_video"
    assert ckpt["pooling"] == "mean"

    metadata = json.loads(METADATA.read_text())
    records = {
        r["file"]: r
        for r in metadata
        if r["split"] == "dev"
    }

    index = [
        json.loads(line)
        for line in (FEATURE_ROOT / "index.jsonl").read_text().splitlines()
        if line.strip()
    ]

    assert len(records) == len(index) == 500
    assert {r["file"] for r in index} == set(records)

    X, y, rows = [], [], []

    for entry in sorted(index, key=lambda r: r["file"]):
        row = records[entry["file"]]

        with np.load(
            FEATURE_ROOT / entry["feature_path"],
            allow_pickle=False,
        ) as data:
            features = data["features"].astype(np.float32)

        X.append(features.mean(axis=0))
        y.append(int(row["modify_video"]))
        rows.append(row)

    X = np.stack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.int64)

    s = ckpt["settings"]

    model = ConcatClassifier(
        ckpt["dims"]["visual"],
        ckpt["dims"]["audio"],
        "visual",
        s["hidden"],
        s["dropout"],
    )

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    V = normalize(X, ckpt["normalization"]["visual"])

    with torch.inference_mode():
        logits = model(V, None).cpu().numpy()

    scores = 1.0 / (
        1.0 + np.exp(-np.clip(logits, -80, 80))
    )

    predictions = (logits >= 0).astype(int)

    metrics = {
        "target": "modify_video",
        "threshold_logit": 0.0,
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(
            average_precision_score(y, scores)
        ),
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y, predictions, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y, predictions, labels=[0, 1]
        ).tolist(),
    }

    print("\n=== VISUAL MANIPULATION RESULTS ===")
    print(json.dumps(metrics, indent=2))

    print("\n=== CATEGORY BREAKDOWN ===")

    for name in ("genuine", "audio_only", "video_only", "both"):
        indices = [
            i for i, row in enumerate(rows)
            if category(row) == name
        ]

        print(
            f"{name:12s} "
            f"count={len(indices):3d} "
            f"predicted_visual_fake="
            f"{predictions[indices].mean():.4f}"
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
                "label": int(row["modify_video"]),
                "logit": float(logit),
                "score": float(score),
                "prediction": int(pred),
            }) + "\n")

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
