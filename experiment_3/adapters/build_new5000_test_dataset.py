import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from e2b.io import canonical_cohort
from e2b.prepare import read_export_split, read_sequence, pool
from e3.data import identity_rows, split_id, load_index
from e3.common import read_json, write_json, write_jsonl, sha256

ROOT = Path("/home/evan/dsa4266/experiment_3")
SOURCE = ROOT / "work/sequences_clip_new5000_train_dev"
OUTPUT = ROOT / "work/sequences_clip_new5000_with_test"
CACHE = ROOT / "work/new5000_canonical_v2"

METADATA = Path(
    "/mnt/c/Users/wuche/Downloads/"
    "LAV-DF-5000-NEW/metadata.new5000.json"
)


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {OUTPUT}"
        )

    # Verify the original prepared dataset first.
    train_cfg, train_rows, _ = load_index(SOURCE, "train")
    dev_cfg, dev_rows, _ = load_index(SOURCE, "dev")

    assert train_cfg == dev_cfg
    assert len(train_rows) == 4000
    assert len(dev_rows) == 500

    config = read_json(SOURCE / "dataset.json")

    assert config["dims"] == {"visual": 512, "audio": 160}
    assert config["target"] == "modify_audio OR modify_video"
    assert not config["test_only"]

    cohort = canonical_cohort(METADATA)
    test = [r for r in cohort if r["split"] == "test"]

    assert len(test) == 500

    indexes = {}

    for modality in ("visual", "audio"):
        cache = CACHE / modality

        info, index = read_export_split(
            cache, "test", modality
        )

        assert info["encoder_id"] == config["encoder_ids"][modality]

        assert set(index) == {r["file"] for r in test}

        indexes[modality] = index

    items = []
    warnings = []

    for row in test:
        entry = dict(row)
        arrays = {}

        for modality in ("visual", "audio"):
            record = indexes[modality][row["file"]]
            cache = CACHE / modality

            assert record["split"] == "test"
            assert record["encoder_id"] == config["encoder_ids"][modality]

            sequence = read_sequence(
                cache, record, modality
            )

            arrays[modality] = sequence

            entry[modality] = {
                key: record[key]
                for key in ("feature_path", "sha256", "shape")
            }

        visual = arrays["visual"]
        audio = arrays["audio"]

        assert visual["features"].shape[1] == 512
        assert audio["features"].shape[1] == 160

        # Require overlapping video-relative timelines.
        if (
            audio["start_s"].min() >
            visual["timestamp_s"].max() + 1e-6
            or audio["end_s"].max() <
            visual["timestamp_s"].min() - 1e-6
        ):
            raise ValueError(
                f"No audio/visual timeline overlap: {row['file']}"
            )

        delta = max(
            abs(float(
                audio["start_s"].min()
                - visual["timestamp_s"].min()
            )),
            abs(float(
                audio["end_s"].max()
                - visual["timestamp_s"].max()
            )),
        )

        if delta > 0.5:
            warnings.append({
                "file": row["file"],
                "seconds": delta,
                "type": "audio_visual_extent_difference",
            })

        items.append(entry)

    OUTPUT.mkdir(parents=True)

    # Preserve original TRAIN/DEV metadata and normalization.
    for filename in (
        "train.jsonl",
        "dev.jsonl",
        "normalization.npz",
        "timing_warnings.jsonl",
    ):
        shutil.copy2(SOURCE / filename, OUTPUT / filename)

    write_jsonl(OUTPUT / "test.jsonl", items)

    config["splits"]["test"] = {
        "signature": split_id(items, config["encoder_ids"]),
        "rows_sha256": sha256(OUTPUT / "test.jsonl"),
        "videos": len(items),
    }

    config["counts"]["test"] = {
        "videos": len(items),
        "labels": dict(Counter(str(r["label"]) for r in items)),
        "categories": dict(Counter(r["category"] for r in items)),
        "source_families": len({
            r["source_family"] for r in items
        }),
        "sequence_lengths": {
            modality: {
                "min": min(r[modality]["shape"][0] for r in items),
                "max": max(r[modality]["shape"][0] for r in items),
                "mean": float(np.mean([
                    r[modality]["shape"][0] for r in items
                ])),
            }
            for modality in ("visual", "audio")
        },
    }

    config["timing_warnings"] += len(warnings)

    write_json(OUTPUT / "dataset.json", config)
    write_jsonl(
        OUTPUT / "test_timing_warnings.jsonl",
        warnings
    )

    # Verify all three splits through the original loader.
    for split, expected in (
        ("train", 4000),
        ("dev", 500),
        ("test", 500),
    ):
        _, rows, _ = load_index(OUTPUT, split)
        assert len(rows) == expected
        print(f"{split.upper()}: {len(rows)} videos verified")

    assert sha256(
        SOURCE / "normalization.npz"
    ) == sha256(
        OUTPUT / "normalization.npz"
    )

    print("\nPASS: TEST-capable Experiment 3 dataset ready.")
    print("Output:", OUTPUT)
    print("TEST timing warnings:", len(warnings))


if __name__ == "__main__":
    main()
