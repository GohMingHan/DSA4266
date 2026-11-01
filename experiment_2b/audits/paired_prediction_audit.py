import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from e2b.train import ConcatClassifier, normalize
from e2b.prepare import pool
from e2b.io import code_hashes

METADATA = Path(
    "/mnt/c/Users/wuche/Downloads/LAV-DF-5000/"
    "LAV-DF-5000/metadata.min.json"
)
VISUAL_CONFIG = BASE / "models/visual/encoder.json"
CHECKPOINT = BASE / "runs/visual_only_clip_any_s42/best.pt"
OUTPUT = BASE / "audits/paired_predictions_dev.csv"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_index(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    result = {}
    for row in rows:
        name = row["file"]
        if name in result:
            raise ValueError(f"Duplicate feature index entry: {name}")
        result[name] = row
    return result


def feature_path(index_path, row):
    raw = Path(row["feature_path"]).expanduser()

    if raw.is_absolute():
        if raw.is_file():
            return raw.resolve()
        raise FileNotFoundError(raw)

    # Resolve relative paths against the index directory and its parents.
    candidates = []
    for base in (index_path.parent, *list(index_path.parent.parents)[:5]):
        candidate = (base / raw).resolve()
        if candidate.is_file() and candidate not in candidates:
            candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(
            f"Could not locate features for {row['file']}.\n"
            f"Relative path: {raw}\n"
            f"Index: {index_path}"
        )

    if len(candidates) > 1:
        raise RuntimeError(
            f"Ambiguous feature path for {row['file']}:\n"
            + "\n".join(map(str, candidates))
        )

    return candidates[0]


def load_sequence(index_path, row):
    path = feature_path(index_path, row)

    with np.load(path, allow_pickle=False) as z:
        if "features" not in z.files:
            raise ValueError(
                f"{path} has no 'features' array; found {z.files}"
            )
        features = np.asarray(z["features"], dtype=np.float32)

    if features.ndim != 2 or len(features) == 0:
        raise ValueError(
            f"Unexpected feature shape for {row['file']}: {features.shape}"
        )

    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite feature values in {path}")

    # Match Experiment 2B's configured mean pooling across frames.
    return pool(features, "mean")


def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def main():
    for path in (METADATA, VISUAL_CONFIG, CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    metadata_by_file = {r["file"]: r for r in metadata}

    visual_config = read_json(VISUAL_CONFIG)
    index_path = Path(visual_config["index"]).expanduser().resolve()
    if not index_path.is_file():
        raise FileNotFoundError(index_path)

    index = load_index(index_path)

    ckpt = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=True
    )

    if ckpt.get("format") != "experiment_2b_v1":
        raise ValueError("Unexpected checkpoint format.")
    if ckpt.get("settings", {}).get("mode") != "visual":
        raise ValueError("This is not a visual-only checkpoint.")
    if ckpt.get("pooling") != "mean":
        raise ValueError(
            f"Expected mean pooling; checkpoint says {ckpt.get('pooling')}"
        )
    if ckpt.get("code_hashes") != code_hashes():
        raise RuntimeError(
            "The current Experiment 2B source hashes differ from the "
            "checkpoint. Stop rather than risk reproducing inference "
            "with different code."
        )

    # Verify the feature index uses the selected CLIP checkpoint.
    visual_checkpoint = Path(visual_config["checkpoint"]).expanduser()
    if not visual_checkpoint.is_absolute():
        visual_checkpoint = (VISUAL_CONFIG.parent / visual_checkpoint).resolve()

    expected_sha = ckpt["encoder_provenance"]["visual"]["engine"][
        "checkpoint_sha256"
    ]

    if visual_checkpoint.is_file():
        actual_sha = sha256(visual_checkpoint)
        if actual_sha != expected_sha:
            raise RuntimeError(
                "Visual checkpoint SHA256 does not match the fusion "
                "checkpoint provenance."
            )

    # Select audio-only DEV videos and their linked originals.
    pairs = []
    skipped = []

    for manipulated in metadata:
        if (
            manipulated.get("split") != "dev"
            or manipulated.get("modify_audio") is not True
            or manipulated.get("modify_video") is not False
        ):
            continue

        original_name = manipulated.get("original")
        if not original_name:
            skipped.append((manipulated["file"], "No linked original"))
            continue

        original = metadata_by_file.get(original_name)
        if original is None:
            skipped.append((manipulated["file"], "Original absent from metadata"))
            continue

        if original.get("split") != "dev":
            skipped.append((manipulated["file"], "Original is not in DEV"))
            continue

        if manipulated["file"] not in index or original_name not in index:
            skipped.append((manipulated["file"], "Feature entry missing"))
            continue

        pairs.append((manipulated, original))

    if not pairs:
        raise RuntimeError("No eligible audio-only DEV pairs were found.")

    # Reconstruct the exact saved visual-only classifier.
    settings = ckpt["settings"]
    dims = ckpt["dims"]

    model = ConcatClassifier(
        dims["visual"],
        dims["audio"],
        mode="visual",
        hidden=settings["hidden"],
        dropout=settings["dropout"],
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    visual_stats = ckpt["normalization"]["visual"]
    threshold = float(ckpt["threshold_logit"])

    records = []
    pooled_cache = {}
    sequence_cache = {}

    def get_pooled(name):
        if name not in pooled_cache:
            pooled_cache[name] = load_sequence(index_path, index[name])
        return pooled_cache[name]

    for n, (manipulated, original) in enumerate(pairs, 1):
        fake_name = manipulated["file"]
        real_name = original["file"]

        fake_vec = get_pooled(fake_name)
        real_vec = get_pooled(real_name)

        if fake_vec.shape != real_vec.shape:
            raise ValueError(
                f"Embedding dimension mismatch: {fake_name} "
                f"{fake_vec.shape}, {real_name} {real_vec.shape}"
            )

        for name in (fake_name, real_name):
            row = index[name]
            if row.get("split") != "dev":
                raise ValueError(
                    f"Feature split mismatch for {name}: {row.get('split')}"
                )

        if fake_vec.shape[0] != dims["visual"]:
            raise ValueError(
                f"Expected {dims['visual']} visual dimensions, "
                f"got {fake_vec.shape[0]}"
            )

        # Match E2B evaluation: saved train-fitted normalization, then model.
        v = normalize(
            np.stack([real_vec, fake_vec]), visual_stats
        )
        dummy_audio = torch.zeros((2, dims["audio"]), dtype=torch.float32)

        with torch.no_grad():
            logits = model(v, dummy_audio).cpu().numpy()

        real_logit, fake_logit = map(float, logits)
        real_score = float(torch.sigmoid(torch.tensor(real_logit)))
        fake_score = float(torch.sigmoid(torch.tensor(fake_logit)))

        real_target = int(
            bool(original.get("modify_audio"))
            or bool(original.get("modify_video"))
        )

        records.append({
            "original": real_name,
            "audio_only_video": fake_name,
            "original_modify_audio": original.get("modify_audio"),
            "original_modify_video": original.get("modify_video"),
            "original_target_any": real_target,
            "audio_only_target_any": 1,
            "original_logit": real_logit,
            "audio_only_logit": fake_logit,
            "original_score_sigmoid": real_score,
            "audio_only_score_sigmoid": fake_score,
            "logit_difference_fake_minus_original": fake_logit - real_logit,
            "score_difference_fake_minus_original": fake_score - real_score,
            "original_prediction": int(real_logit >= threshold),
            "audio_only_prediction": int(fake_logit >= threshold),
            "decision_threshold_logit": threshold,
            "pooled_embedding_cosine": cosine_similarity(real_vec, fake_vec),
            "original_feature_steps": int(index[real_name].get("steps", -1)),
            "audio_only_feature_steps": int(index[fake_name].get("steps", -1)),
        })

        if n % 25 == 0:
            print(f"Scored {n}/{len(pairs)} pairs")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    differences = np.array([
        r["logit_difference_fake_minus_original"] for r in records
    ])
    cosine = np.array([
        r["pooled_embedding_cosine"] for r in records
    ])
    comparable = [
        r for r in records if r["original_target_any"] == 0
    ]

    print("\n=== PAIRED PREDICTION SUMMARY ===")
    print("Audio-only DEV pairs scored:", len(records))
    print("Pairs skipped:", len(skipped))
    print("Linked original is a clean target:", len(comparable))
    print("Threshold logit:", threshold)
    print("Median pooled embedding cosine:", float(np.nanmedian(cosine)))
    print("Pairs where audio-only score > original score:",
          float(np.mean(differences > 0)))
    print("Median logit difference (fake - original):",
          float(np.median(differences)))
    print("Mean logit difference (fake - original):",
          float(np.mean(differences)))

    if comparable:
        print(
            "Both classified positive:",
            sum(r["original_prediction"] == 1 and
                r["audio_only_prediction"] == 1 for r in comparable),
            "/", len(comparable)
        )
        print(
            "Original negative, audio-only positive:",
            sum(r["original_prediction"] == 0 and
                r["audio_only_prediction"] == 1 for r in comparable),
            "/", len(comparable)
        )
        print(
            "Same decision:",
            sum(r["original_prediction"] ==
                r["audio_only_prediction"] for r in comparable),
            "/", len(comparable)
        )

    if skipped:
        print("\nSkipped examples:")
        for name, reason in skipped[:10]:
            print(" ", name, "-", reason)

    print("\nCSV saved to:", OUTPUT.resolve())


if __name__ == "__main__":
    main()
