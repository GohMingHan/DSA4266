import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

ROOT = Path("/home/evan/dsa4266/experiment_2b")
sys.path.insert(0, str(ROOT))

from e2b.train import ConcatClassifier, normalizer, normalize

DATA_ROOT = Path("/mnt/c/Users/wuche/Downloads/LAV-DF-5000-NEW")
METADATA = DATA_ROOT / "metadata.new5000.json"

FEATURE_BASE = Path(
    "/home/evan/dsa4266/experiment_2b_outputs"
)
SUFFIX = (
    "clip_visual_features/"
    "vit_b32_lower_lr_10fps_a69a1ddcf56d_0576c2b0"
)

FEATURE_ROOTS = {
    "train": FEATURE_BASE / "new5000_train" / SUFFIX,
    "dev": FEATURE_BASE / "new5000" / SUFFIX,
}

OUTPUT = ROOT / "runs/visual_only_clip_modify_video_new5000_s42"

SEED = 42
EPOCHS = 50
PATIENCE = 8
BATCH_SIZE = 64
LR = 0.001
WEIGHT_DECAY = 0.0001
HIDDEN = 128
DROPOUT = 0.2


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_split(split, metadata):
    root = FEATURE_ROOTS[split]
    index = [
        json.loads(line)
        for line in (root / "index.jsonl").read_text().splitlines()
        if line.strip()
    ]

    records = {
        r["file"]: r
        for r in metadata
        if r["split"] == split
    }

    assert len(index) == len(records)
    assert {r["file"] for r in index} == set(records)

    X, y, rows = [], [], []

    for entry in sorted(index, key=lambda r: r["file"]):
        name = entry["file"]
        row = records[name]

        with np.load(
            root / entry["feature_path"],
            allow_pickle=False,
        ) as data:
            features = data["features"].astype(np.float32)

        assert features.ndim == 2
        assert features.shape[1] == 512
        assert len(features) > 0
        assert np.isfinite(features).all()

        X.append(features.mean(axis=0))
        y.append(int(row["modify_video"]))
        rows.append(row)

    return (
        np.stack(X).astype(np.float32),
        np.asarray(y, dtype=np.int64),
        rows,
    )


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT}"
        )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    metadata = json.loads(METADATA.read_text())

    X_train, y_train, train_rows = load_split("train", metadata)
    X_dev, y_dev, dev_rows = load_split("dev", metadata)

    assert len(X_train) == 4000
    assert len(X_dev) == 500
    assert np.bincount(y_train).tolist() == [2000, 2000]
    assert np.bincount(y_dev).tolist() == [250, 250]

    stats = normalizer(X_train)

    train_v = normalize(X_train, stats)
    dev_v = normalize(X_dev, stats)

    model = ConcatClassifier(
        dv=512,
        da=160,
        mode="visual",
        hidden=HIDDEN,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = torch.nn.BCEWithLogitsLoss()

    dataset = TensorDataset(
        train_v,
        torch.zeros(len(train_v), 160),
        torch.tensor(y_train, dtype=torch.float32),
    )

    counts = np.bincount(y_train)
    weights = torch.tensor(
        1.0 / counts[y_train],
        dtype=torch.double,
    )

    OUTPUT.mkdir(parents=True, exist_ok=False)

    best_auc = -float("inf")
    stale = 0
    history = []

    for epoch in range(EPOCHS):
        generator = torch.Generator().manual_seed(SEED + epoch)

        sampler = WeightedRandomSampler(
            weights,
            len(y_train),
            replacement=True,
            generator=generator,
        )

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=0,
        )

        model.train()
        total_loss = 0.0

        for v, a, y in loader:
            v, a, y = v.to(device), a.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(v, a)
            loss = criterion(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float(loss.item()) * len(y)

        model.eval()

        with torch.inference_mode():
            dev_logits = model(
                dev_v.to(device),
                torch.zeros(len(dev_v), 160, device=device),
            ).cpu().numpy()

        auc = float(roc_auc_score(y_dev, dev_logits))
        ap = float(average_precision_score(y_dev, dev_logits))

        record = {
            "epoch": epoch + 1,
            "train_loss": total_loss / len(y_train),
            "dev_auc": auc,
            "dev_ap": ap,
        }

        history.append(record)
        print(json.dumps(record), flush=True)

        if auc > best_auc:
            best_auc = auc
            stale = 0

            torch.save({
                "model": model.state_dict(),
                "normalization": {"visual": stats},
                "settings": {
                    "mode": "visual",
                    "hidden": HIDDEN,
                    "dropout": DROPOUT,
                    "seed": SEED,
                },
                "dims": {"visual": 512, "audio": 160},
                "pooling": "mean",
                "target": "modify_video",
                "epoch": epoch + 1,
                "dev_auc": auc,
                "dev_ap": ap,
                "metadata_sha256": sha256(METADATA),
            }, OUTPUT / "best.pt")

        else:
            stale += 1

        if stale >= PATIENCE:
            print("Early stopping triggered.")
            break

    (OUTPUT / "history.json").write_text(
        json.dumps(history, indent=2)
    )

    print("Best DEV ROC-AUC:", best_auc)
    print("Checkpoint:", OUTPUT / "best.pt")


if __name__ == "__main__":
    main()
