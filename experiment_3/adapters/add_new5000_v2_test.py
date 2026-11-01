import json
import os
import hashlib
import shutil
from pathlib import Path

import numpy as np

from e2b.io import (
    canonical_cohort,
    check_sequence,
    digest,
)
from e2b.prepare import read_export_split, read_sequence

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "experiment_3/work/new5000_clip_v2_canonical_fixed"

METADATA = Path(
    os.environ["LAVDF_METADATA"]
).expanduser().resolve()

CLIP_SUFFIX = (
    "clip_visual_features/"
    "vit_b32_shared_fixed20k_10fps_9f846830fa09_0576c2b0"
)

SOURCES = {
    "visual": (
        ROOT / "experiment_2b_outputs/new5000_v2_test"
        / CLIP_SUFFIX
    ),
    "audio": (
        ROOT / "experiment_2a_new5000_aasist_test_embeddings"
    ),
}

EXPECTED_ENCODERS = {
    "visual": "dc0e7b441834066fa0212acc6ea27a96f73a214b11fb3d6b12028ed9d81c2309",
    "audio": "314453656435b250a368d1adad0b7a7fb30dce5ef72576abe6aa599c1d406b90",
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    cohort = canonical_cohort(METADATA)
    expected = {
        r["file"] for r in cohort
        if r["split"] == "test"
    }

    assert len(expected) == 500

    for modality, source in SOURCES.items():
        dest = CACHE / modality
        export = json.loads(
            (dest / "export.json").read_text()
        )

        encoder_id = export["encoder_id"]
        assert encoder_id == EXPECTED_ENCODERS[modality]

        index_path = source / "index.jsonl"
        index = [
            json.loads(line)
            for line in index_path.read_text().splitlines()
            if line.strip()
        ]

        assert len(index) == 500
        assert {r["file"] for r in index} == expected
        assert all(r["split"] == "test" for r in index)

        records = []

        for row in index:
            name = row["file"]

            original = source / row["feature_path"]
            target = dest / "clips" / Path(name).with_suffix(".npz")

            assert original.is_file()

            target.parent.mkdir(parents=True, exist_ok=True)

            if target.exists():
                assert sha256(target) == sha256(original), (
                    f"Existing TEST feature differs: {target}"
                )
            else:
                shutil.copy2(original, target)

            with np.load(target, allow_pickle=False) as data:
                arrays = check_sequence(data, modality)
                shape = list(arrays["features"].shape)

            records.append({
                "file": name,
                "split": "test",
                "encoder_id": encoder_id,
                "feature_path": target.relative_to(dest).as_posix(),
                "sha256": sha256(target),
                "shape": shape,
                "status": "complete",
            })

        records.sort(key=lambda r: r["file"])

        body = {
            "format": "experiment_2b_v1",
            "encoder_id": encoder_id,
            "modality": modality,
            "split": "test",
            "partial_pilot": False,
            "test_only": False,
            "videos": records,
        }

        body["signature"] = digest(body)

        output = dest / "splits/test.json"

        if output.exists():
            existing = json.loads(output.read_text())

            assert existing == body, (
                f"Existing TEST manifest differs: {output}"
            )

            print(f"{modality}: Existing TEST manifest matches.")
        else:
            output.write_text(json.dumps(body, indent=2))

        info, verified_index = read_export_split(
            dest, "test", modality
        )

        assert len(verified_index) == 500

        for row in verified_index.values():
            read_sequence(dest, row, modality)

        print(
            f"{modality}: 500 TEST sequences verified"
        )

    print("\nPASS: TEST canonical manifests complete.")


if __name__ == "__main__":
    main()
