import csv
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
ROOT = Path(
    "/mnt/c/Users/wuche/Downloads/LAV-DF-5000/LAV-DF-5000"
)
METADATA = ROOT / "metadata.min.json"
VISUAL_CONFIG = BASE / "models/visual/encoder.json"
PAIRS_CSV = BASE / "audits/paired_predictions_dev.csv"
OUTPUT = BASE / "audits/pair_confound_details.csv"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def get_index(index_path):
    index = {}
    with index_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row["file"] in index:
                    raise ValueError(f"Duplicate index entry: {row['file']}")
                index[row["file"]] = row
    return index


def resolve_feature(index_path, row):
    raw = Path(row["feature_path"]).expanduser()

    if raw.is_absolute():
        if raw.is_file():
            return raw.resolve()
        raise FileNotFoundError(raw)

    found = []
    for base in (index_path.parent, *list(index_path.parent.parents)[:5]):
        p = (base / raw).resolve()
        if p.is_file() and p not in found:
            found.append(p)

    if len(found) != 1:
        raise RuntimeError(
            f"Expected one feature file for {row['file']}; found: {found}"
        )
    return found[0]


def load_pooled(index_path, row):
    path = resolve_feature(index_path, row)
    with np.load(path, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float64)
        timestamps = np.asarray(data["timestamp_s"], dtype=np.float64)

    if features.ndim != 2 or len(features) == 0:
        raise ValueError(f"Bad feature shape for {row['file']}: {features.shape}")
    if len(timestamps) != len(features):
        raise ValueError(f"Timestamp mismatch for {row['file']}")

    return {
        "pooled": features.mean(axis=0),
        "steps": len(features),
        "first_ts": float(timestamps.min()),
        "last_ts": float(timestamps.max()),
        "frame_norm_mean": float(np.linalg.norm(features, axis=1).mean()),
        "frame_norm_std": float(np.linalg.norm(features, axis=1).std()),
    }


def spearman(x, y):
    """Rank correlation with average ranks for ties."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return float("nan")

    def ranks(a):
        order = np.argsort(a)
        out = np.empty(len(a), dtype=float)
        i = 0
        while i < len(a):
            j = i + 1
            while j < len(a) and a[order[j]] == a[order[i]]:
                j += 1
            out[order[i:j]] = (i + 1 + j) / 2
            i = j
        return out

    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def summary(name, values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        print(f"{name}: no valid values")
        return
    print(
        f"{name}: median={np.median(a):.6f}, "
        f"mean={np.mean(a):.6f}, "
        f"p10={np.percentile(a, 10):.6f}, "
        f"p90={np.percentile(a, 90):.6f}"
    )


def main():
    config = load_json(VISUAL_CONFIG)
    index_path = Path(config["index"]).expanduser().resolve()
    if not index_path.is_file():
        raise FileNotFoundError(index_path)

    metadata = load_json(METADATA)
    metadata_by_file = {r["file"]: r for r in metadata}
    index = get_index(index_path)

    with PAIRS_CSV.open(encoding="utf-8", newline="") as f:
        pairs = list(csv.DictReader(f))

    records = []
    cache = {}

    def feature(name):
        if name not in index:
            raise KeyError(f"Missing feature index entry: {name}")
        if name not in cache:
            cache[name] = load_pooled(index_path, index[name])
        return cache[name]

    for row in pairs:
        original = row["original"]
        fake = row["audio_only_video"]

        a = feature(original)
        b = feature(fake)

        if a["pooled"].shape != b["pooled"].shape:
            raise ValueError(f"Feature dimension mismatch: {original}, {fake}")

        meta_a = metadata_by_file[original]
        meta_b = metadata_by_file[fake]

        vec_a, vec_b = a["pooled"], b["pooled"]
        delta = vec_b - vec_a

        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        cosine = float(
            np.dot(vec_a, vec_b) /
            max(norm_a * norm_b, 1e-12)
        )

        dur_a = float(meta_a["duration"])
        dur_b = float(meta_b["duration"])

        records.append({
            "original": original,
            "audio_only_video": fake,
            "logit_difference": float(
                row["logit_difference_fake_minus_original"]
            ),
            "pooled_cosine": cosine,
            "raw_embedding_l2_difference": float(np.linalg.norm(delta)),
            "original_pooled_norm": norm_a,
            "audio_only_pooled_norm": norm_b,
            "pooled_norm_difference": norm_b - norm_a,
            "pooled_norm_ratio": norm_b / max(norm_a, 1e-12),
            "original_steps": a["steps"],
            "audio_only_steps": b["steps"],
            "step_difference": b["steps"] - a["steps"],
            "original_feature_span_s": a["last_ts"] - a["first_ts"],
            "audio_only_feature_span_s": b["last_ts"] - b["first_ts"],
            "feature_span_difference_s": (
                (b["last_ts"] - b["first_ts"])
                - (a["last_ts"] - a["first_ts"])
            ),
            "original_metadata_duration_s": dur_a,
            "audio_only_metadata_duration_s": dur_b,
            "metadata_duration_difference_s": dur_b - dur_a,
            "original_frame_norm_mean": a["frame_norm_mean"],
            "audio_only_frame_norm_mean": b["frame_norm_mean"],
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print("Pairs analysed:", len(records))
    print("\n=== Paired differences ===")
    for key in (
        "logit_difference",
        "pooled_cosine",
        "raw_embedding_l2_difference",
        "pooled_norm_difference",
        "pooled_norm_ratio",
        "step_difference",
        "metadata_duration_difference_s",
        "feature_span_difference_s",
    ):
        summary(key, [r[key] for r in records])

    print("\n=== Spearman correlation with logit difference ===")
    target = [r["logit_difference"] for r in records]
    for key in (
        "pooled_norm_difference",
        "pooled_norm_ratio",
        "step_difference",
        "metadata_duration_difference_s",
        "feature_span_difference_s",
        "raw_embedding_l2_difference",
        "pooled_cosine",
    ):
        print(
            f"{key}: rho="
            f"{spearman([r[key] for r in records], target):.4f}"
        )

    print("\nLargest logit differences:")
    for r in sorted(
        records, key=lambda x: x["logit_difference"], reverse=True
    )[:10]:
        print(
            f"{r['original']} -> {r['audio_only_video']}: "
            f"logit_delta={r['logit_difference']:.3f}, "
            f"steps={r['original_steps']}->{r['audio_only_steps']}, "
            f"duration_delta={r['metadata_duration_difference_s']:.3f}s, "
            f"norm_ratio={r['pooled_norm_ratio']:.4f}, "
            f"cosine={r['pooled_cosine']:.5f}"
        )

    print("\nCSV:", OUTPUT)


if __name__ == "__main__":
    main()
