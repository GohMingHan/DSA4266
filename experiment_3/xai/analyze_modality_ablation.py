import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("xai/results")
CONDITIONS = ("full", "no_audio", "no_visual", "neither")
CATEGORIES = ("real", "audio_only", "visual_only", "both")


def analyze(model):
    path = ROOT / f"{model}_v2_dev/predictions.jsonl"

    records = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]

    df = pd.DataFrame(records)

    assert len(df) == 500
    assert df["file"].is_unique
    assert set(df["category"]) == set(CATEGORIES)

    print(f"\n{'=' * 70}")
    print(f"{model.upper()} — CATEGORY-SPECIFIC MODALITY ABLATION")
    print(f"{'=' * 70}")

    summary = []

    for category in CATEGORIES:
        subset = df[df["category"] == category]

        assert len(subset) == 125

        for condition in CONDITIONS:
            predicted_fake = subset[
                f"{condition}_prediction"
            ].mean()

            logits = subset[
                f"{condition}_logit"
            ].to_numpy(dtype=float)

            summary.append({
                "category": category,
                "condition": condition,
                "predicted_fake_rate": predicted_fake,
                "median_logit": np.median(logits),
                "mean_logit": np.mean(logits),
            })

    result = pd.DataFrame(summary)

    print("\n=== PREDICTED FAKE RATE ===")
    print(
        result.pivot(
            index="category",
            columns="condition",
            values="predicted_fake_rate",
        ).reindex(
            index=CATEGORIES,
            columns=CONDITIONS,
        ).round(4).to_string()
    )

    print("\n=== MEDIAN FAKE LOGIT ===")
    print(
        result.pivot(
            index="category",
            columns="condition",
            values="median_logit",
        ).reindex(
            index=CATEGORIES,
            columns=CONDITIONS,
        ).round(3).to_string()
    )

    print("\n=== CONTENT CONTRIBUTION ===")

    for category in CATEGORIES:
        subset = df[df["category"] == category]

        audio_delta = subset[
            "audio_content_delta_logit"
        ].to_numpy(dtype=float)

        visual_delta = subset[
            "visual_content_delta_logit"
        ].to_numpy(dtype=float)

        print(
            f"{category:13s} "
            f"audio_median={np.median(audio_delta):+9.3f} "
            f"visual_median={np.median(visual_delta):+9.3f} "
            f"audio_positive={(audio_delta > 0).mean():.3f} "
            f"visual_positive={(visual_delta > 0).mean():.3f}"
        )

    print("\n=== PREDICTION FLIPS ===")

    for condition in CONDITIONS[1:]:
        print(f"\n{condition}")

        for category in CATEGORIES:
            subset = df[df["category"] == category]

            full = subset["full_prediction"].to_numpy()
            ablated = subset[
                f"{condition}_prediction"
            ].to_numpy()

            changed = np.mean(full != ablated)

            print(
                f"  {category:13s} "
                f"changed={changed:.3f}"
            )

    out = ROOT / f"{model}_v2_dev/category_analysis.csv"
    result.to_csv(out, index=False)

    print("\nSaved:", out)


if __name__ == "__main__":
    for model in ("concat", "cross"):
        analyze(model)
