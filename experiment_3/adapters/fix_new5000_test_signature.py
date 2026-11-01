import json
from pathlib import Path

import numpy as np

from e2b.prepare import (
    read_sequence,
    pool,
    split_content_id,
)
from e3.data import identity_rows, load_index
from e3.common import write_json

ROOT = Path(
    "/home/evan/dsa4266/experiment_3"
)

DATA = ROOT / "work/sequences_clip_new5000_with_test"


def main():
    config_path = DATA / "dataset.json"
    config = json.loads(config_path.read_text())

    # Verify existing dataset before modifying metadata.
    _, rows, _ = load_index(DATA, "test")

    assert len(rows) == 500
    assert "e2b_mean_signature" not in config["splits"]["test"]

    visual = []
    audio = []

    for row in rows:
        v = read_sequence(
            config["embedding_roots"]["visual"],
            row["visual"],
            "visual",
        )

        a = read_sequence(
            config["embedding_roots"]["audio"],
            row["audio"],
            "audio",
        )

        visual.append(pool(v["features"], "mean"))
        audio.append(pool(a["features"], "mean"))

    V = np.stack(visual)
    A = np.stack(audio)

    assert V.shape == (500, 512)
    assert A.shape == (500, 160)

    signature = split_content_id(
        identity_rows(rows),
        V,
        A,
        config["encoder_ids"],
        "mean",
    )

    print("Calculated TEST E2B signature:", signature)

    # Preserve all existing dataset fields and fingerprints.
    config["splits"]["test"]["e2b_mean_signature"] = signature

    write_json(config_path, config)

    # Verify all three splits after repair.
    for split, expected in (
        ("train", 4000),
        ("dev", 500),
        ("test", 500),
    ):
        _, checked, _ = load_index(DATA, split)
        assert len(checked) == expected
        print(f"{split.upper()}: {len(checked)} verified")

    print("\nPASS: TEST E2B signature repaired.")


if __name__ == "__main__":
    main()
