import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

ROOT = Path("/home/evan/dsa4266")

OUTPUT = (
    ROOT / "experiment_3/evaluations/"
    "category_comparison_new5000_dev"
)

SOURCES = {
    "visual_only": (
        ROOT / "experiment_2b/evaluations/"
        "visual_modify_video_new5000_dev",
        "predictions.jsonl",
    ),
    "concat": (
        ROOT / "experiment_3/evaluations/"
        "concat_clip_new5000_dev",
        "video_predictions.jsonl",
    ),
    "cross": (
        ROOT / "experiment_3/evaluations/"
        "cross_clip_new5000_dev",
        "video_predictions.jsonl",
    ),
}

EVALUATIONS = {
    "A_visual_only": ("real", "visual_only"),
    "B_audiovisual": ("real", "both"),
}

CATEGORY_MAP = {
    "genuine": "real",
    "real": "real",
    "audio_only": "audio_only",
    "video_only": "visual_only",
    "visual_only": "visual_only",
    "both": "both",
}


def load_model(folder, filename):
    metrics = json.loads(
        (folder / "metrics.json").read_text()
    )

    rows = [
        json.loads(line)
        for line in (folder / filename).read_text().splitlines()
        if line.strip()
    ]

    assert len(rows) == 500

    by_file = {}

    for row in rows:
        name = row["file"]
        assert name not in by_file

        row = dict(row)
        row["category"] = CATEGORY_MAP[row["category"]]

        by_file[name] = row

    threshold = float(metrics["threshold_logit"])

    # Verify that saved predictions correspond to
    # the saved evaluation threshold.
    for row in by_file.values():
        expected = int(float(row["logit"]) >= threshold)
        assert int(row["prediction"]) == expected, (
            f"Prediction/threshold mismatch: {row['file']}"
        )

    return by_file, threshold


def evaluate(rows, categories):
    selected = [
        row for row in rows.values()
        if row["category"] in categories
    ]

    assert len(selected) == 250

    # All models share these ground-truth labels
    # within the selected categories.
    y = np.array(
        [int(r["category"] != "real") for r in selected]
    )

    logits = np.array(
        [float(r["logit"]) for r in selected]
    )

    pred = np.array(
        [int(r["prediction"]) for r in selected]
    )

    assert np.bincount(y).tolist() == [125, 125]

    cm = confusion_matrix(y, pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    return {
        "videos": len(selected),
        "roc_auc": float(roc_auc_score(y, logits)),
        "average_precision": float(
            average_precision_score(y, logits)
        ),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(
            precision_score(y, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y, pred, zero_division=0)
        ),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, pred)
        ),
        "false_positive_rate": float(fp / (fp + tn)),
        "confusion_matrix": cm.tolist(),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    models = {}

    for name, (folder, filename) in SOURCES.items():
        models[name] = load_model(folder, filename)

    reference = set(models["visual_only"][0])

    for name, (rows, _) in models.items():
        assert set(rows) == reference, (
            f"Video IDs differ for {name}"
        )

    # Ensure category labels agree for every video.
    for video in reference:
        categories = {
            models[name][0][video]["category"]
            for name in models
        }
        assert len(categories) == 1, (
            f"Category mismatch: {video}"
        )

    report = {}

    for evaluation, categories in EVALUATIONS.items():
        report[evaluation] = {}

        print(f"\n=== {evaluation.upper()} ===")

        for model_name, (rows, threshold) in models.items():
            result = evaluate(rows, categories)
            result["threshold_logit"] = threshold

            report[evaluation][model_name] = result

            print(
                f"{model_name:12s} "
                f"AUC={result['roc_auc']:.4f} "
                f"F1={result['f1']:.4f} "
                f"Accuracy={result['accuracy']:.4f} "
                f"Recall={result['recall']:.4f} "
                f"FPR={result['false_positive_rate']:.4f}"
            )

    output = OUTPUT / "comparison.json"

    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {output}"
        )

    output.write_text(json.dumps(report, indent=2))

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
