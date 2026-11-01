import hashlib
import os
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from models import build_model
from audio_data import make_dataset
from common import loader
from experiment import source_compatible

DATA_ROOT = PROJECT_ROOT / "experiment_2a_new5000"

CHECKPOINT = Path(
    os.environ.get(
        "AASIST_CHECKPOINT",
        str(PROJECT_ROOT / "experiment_2b/models/audio/best.pt")
    )
).expanduser().resolve()

AASIST_DIR = ROOT / "upstream_aasist"
ORIGINAL_CONFIG = PROJECT_ROOT / "config.json"
OUTPUT = PROJECT_ROOT / "experiment_2a_new5000_aasist_test_embeddings"

EXPECTED_SHA = (
    "29b87952d352c4d69c0316d34effdba4babc0e8c156c18fd1993b39061153efc"
)

BATCH_SIZE = 16


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    assert sha256(CHECKPOINT) == EXPECTED_SHA

    old_config = load_json(ORIGINAL_CONFIG)
    new_config = load_json(DATA_ROOT / "config.json")

    assert old_config["config_id"] == (
        "8811df7fbbae7faac6a954822b8740767b99e2768b9a95e75b13187164a3de91"
    )

    # Verify identical numerical preprocessing parameters.
    assert old_config["audio"] == new_config["audio"], (
        "Audio preprocessing parameters differ."
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["model_name"] == "aasist"
    assert checkpoint["config_id"] == old_config["config_id"]

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    model = build_model(
        "aasist",
        "scratch",
        aasist_dir=AASIST_DIR,
    )

    assert source_compatible(
        checkpoint["model_provenance"],
        model.provenance,
    )

    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device).eval()
    model.requires_grad_(False)

    assert model.embedding_dim == 160

    records = [
        json.loads(line)
        for line in (DATA_ROOT / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]

    records = [
        r for r in records
        if r["split"] == "test"
    ]

    assert len(records) == 500

    counts = Counter(r["split"] for r in records)
    assert counts == {"test": 500}

    provenance = {
        "checkpoint_sha256": EXPECTED_SHA,
        "original_config_id": old_config["config_id"],
        "new_config_id": new_config["config_id"],
        "audio_parameters": new_config["audio"],
        "model_name": "aasist",
        "embedding_dim": 160,
        "representation": "forward_features()[0]",
        "batch_size": BATCH_SIZE,
        "environment": {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
        },
    }

    if OUTPUT.exists() and (OUTPUT / "provenance.json").exists():
        assert load_json(OUTPUT / "provenance.json") == provenance, (
            "Export configuration changed."
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )

    index = []

    for row in tqdm(records, desc="AASIST frozen embeddings"):
        name = row["file"]
        target = OUTPUT / "features" / Path(name).with_suffix(".npz")
        target.parent.mkdir(parents=True, exist_ok=True)

        if not target.exists():
            dataset = make_dataset(
                "aasist",
                DATA_ROOT,
                [row],
                row["split"],
                checkpoint.get("stats"),
                new_config["audio"],
            )

            batches = loader(
                dataset,
                BATCH_SIZE,
                0,
            )

            embeddings = np.empty(
                (len(dataset), 160),
                dtype=np.float32,
            )

            with torch.inference_mode():
                for x, _, indices in batches:
                    h, _ = model.forward_features(x.to(device))
                    embeddings[indices.numpy()] = (
                        h.cpu().numpy().astype(np.float32)
                    )

            assert embeddings.shape == (len(row["windows"]), 160)
            assert np.isfinite(embeddings).all()

            windows = row["windows"]

            temporary = target.with_name(target.name + ".tmp")

            with temporary.open("wb") as f:
                np.savez_compressed(
                    f,
                    features=embeddings,
                    start_s=np.array(
                        [w["start_s"] for w in windows],
                        dtype=np.float32,
                    ),
                    end_s=np.array(
                        [w["end_s"] for w in windows],
                        dtype=np.float32,
                    ),
                    center_s=np.array(
                        [w["center_s"] for w in windows],
                        dtype=np.float32,
                    ),
                    padding_fraction=np.array(
                        [w["padding_fraction"] for w in windows],
                        dtype=np.float32,
                    ),
                )

            temporary.replace(target)

        index.append({
            "file": name,
            "split": row["split"],
            "feature_path": target.relative_to(OUTPUT).as_posix(),
            "checkpoint_sha256": EXPECTED_SHA,
        })

    with (OUTPUT / "index.jsonl").open("w") as f:
        for row in index:
            f.write(json.dumps(row) + "\n")

    print("\nExtraction complete.")
    print("Videos:", len(index))
    print("Split counts:", dict(counts))
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()
