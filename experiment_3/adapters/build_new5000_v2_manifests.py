import json
import os
import hashlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiment_3"))

from e2b.io import digest, check_sequence, canonical_cohort

METADATA = Path(
    os.environ["LAVDF_METADATA"]
).expanduser().resolve()

CLIP_SUFFIX = (
    "clip_visual_features/"
    "vit_b32_shared_fixed20k_10fps_9f846830fa09_0576c2b0"
)

SOURCES = {
    "visual": {
        "train": ROOT / "experiment_2b_outputs/new5000_v2_train" / CLIP_SUFFIX,
        "dev": ROOT / "experiment_2b_outputs/new5000_v2_dev" / CLIP_SUFFIX,
    },
    "audio": {
        "train": ROOT / "experiment_2a_new5000_aasist_embeddings",
        "dev": ROOT / "experiment_2a_new5000_aasist_embeddings",
    },
}

OUTPUT = ROOT / "experiment_3/work/new5000_clip_v2_canonical_fixed"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_index(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def main():
    cohort = canonical_cohort(METADATA)

    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {OUTPUT}"
        )

    OUTPUT.mkdir(parents=True)

    for modality, split_roots in SOURCES.items():
        dest = OUTPUT / modality
        dest.mkdir()

        # Canonical features must remain under the cache root.
        # Copying the NPZ files preserves the originals.
        feature_dest = dest / "clips"
        feature_dest.mkdir()

        records_by_split = {}
        provenance = {
            "format": "experiment_2b_v1",
            "modality": modality,
            "time_origin": "first_decoded_video_frame",
            "test_only": False,
            "engine": {
                "kind": "preexported",
                "description": (
                    "Frozen CLIP embeddings"
                    if modality == "visual"
                    else "Frozen AASIST embeddings"
                ),
            },
        }

        if modality == "visual":
            train_config = json.loads(
                (split_roots["train"] / "export_config.json").read_text()
            )
            dev_config = json.loads(
                (split_roots["dev"] / "export_config.json").read_text()
            )

            expected_sha256 = (
                "9f846830fa093ceaed0ad272c5cb5ab9255ec301670eb542fbd2fb2825acada3"
            )

            for cfg in (train_config, dev_config):
                assert cfg["checkpoint_sha256"] == expected_sha256
                assert cfg["sampling_fps"] == 10.0
                assert cfg["feature_dimension"] == 512
                assert cfg["feature_layer"] == "normalized encoder.image_embeds"

            provenance["engine"]["checkpoint_sha256"] = expected_sha256
            provenance["engine"]["sampling_fps"] = 10.0
            provenance["engine"]["feature_dimension"] = 512
            provenance["engine"]["feature_layer"] = (
                "normalized encoder.image_embeds"
            )

        encoder_id = digest(provenance)
        provenance["encoder_id"] = encoder_id

        for split in ("train", "dev"):
            source = split_roots[split]
            index = [
                row
                for row in read_index(source / "index.jsonl")
                if row["split"] == split
            ]

            expected = {
                r["file"] for r in cohort
                if r["split"] == split
            }

            assert {r["file"] for r in index} == expected

            records = []

            for row in index:
                original_path = source / row["feature_path"]
                target = feature_dest / Path(row["file"]).with_suffix(".npz")
                target.parent.mkdir(parents=True, exist_ok=True)

                if target.exists():
                    raise FileExistsError(target)

                import shutil
                shutil.copy2(original_path, target)

                with np.load(target, allow_pickle=False) as data:
                    arrays = check_sequence(data, modality)
                    shape = list(arrays["features"].shape)

                records.append({
                    "file": row["file"],
                    "split": split,
                    "encoder_id": encoder_id,
                    "feature_path": target.relative_to(dest).as_posix(),
                    "sha256": sha256(target),
                    "shape": shape,
                    "status": "complete",
                })

            records.sort(key=lambda r: r["file"])

            records_by_split[split] = records

        (dest / "export.json").write_text(
            json.dumps(provenance, indent=2)
        )

        (dest / "splits").mkdir()

        for split, records in records_by_split.items():
            body = {
                "format": "experiment_2b_v1",
                "encoder_id": encoder_id,
                "modality": modality,
                "split": split,
                "partial_pilot": False,
                "test_only": False,
                "videos": records,
            }

            body["signature"] = digest(body)

            (dest / "splits" / f"{split}.json").write_text(
                json.dumps(body, indent=2)
            )

            print(
                modality,
                split,
                len(records),
                "videos"
            )

    print("\nCanonical caches created:", OUTPUT)


if __name__ == "__main__":
    main()
