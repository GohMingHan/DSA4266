import csv
import json
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


ROOT = Path("/mnt/c/Users/wuche/Downloads/LAV-DF-5000/LAV-DF-5000")
METADATA = ROOT / "metadata.min.json"
VISUAL_CONFIG = Path("experiment_2b/models/visual/encoder.json")
OUTPUT = Path("experiment_2b/audits/audio_only_dev_pairs.csv")

# At approximately 10 FPS, this allows small timestamp differences
# while avoiding matches to adjacent frames in normal cases.
TIME_TOLERANCE_S = 0.055
MAX_IMAGE_COMPARISONS = 8


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_feature(index_path, row):
    """Resolve the feature path without changing any source files."""
    raw = Path(row["feature_path"])

    if raw.is_absolute() and raw.is_file():
        return raw

    candidates = [
        index_path.parent / raw,
        index_path.parent.parent / raw,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Cannot locate feature for {row.get('file')}: "
        f"{row['feature_path']}\n"
        f"Tried:\n" + "\n".join(str(p) for p in candidates)
    )


def load_feature(index_path, index_row):
    path = resolve_feature(index_path, index_row)

    with np.load(path, allow_pickle=False) as data:
        required = {"features", "timestamp_s"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f"{path} missing arrays {sorted(missing)}; "
                f"available arrays: {data.files}"
            )

        features = np.asarray(data["features"], dtype=np.float32)
        times = np.asarray(data["timestamp_s"], dtype=np.float64)

    if features.ndim != 2 or len(features) != len(times):
        raise ValueError(
            f"Unexpected shapes for {path}: "
            f"features={features.shape}, timestamps={times.shape}"
        )

    return features, times


def align_timestamps(times_a, times_b, tolerance):
    """Monotonic one-to-one matching of timestamps."""
    pairs = []
    i = j = 0

    while i < len(times_a) and j < len(times_b):
        delta = times_a[i] - times_b[j]

        if abs(delta) <= tolerance:
            pairs.append((i, j))
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1

    return pairs


def cosine_by_frame(features_a, features_b, pairs):
    a = np.stack([features_a[i] for i, _ in pairs])
    b = np.stack([features_b[j] for _, j in pairs])

    a_norm = a / np.maximum(
        np.linalg.norm(a, axis=1, keepdims=True), 1e-12
    )
    b_norm = b / np.maximum(
        np.linalg.norm(b, axis=1, keepdims=True), 1e-12
    )

    return np.sum(a_norm * b_norm, axis=1)


def read_gray_frame(cap, timestamp_s):
    # OpenCV seeks to the requested video time, then decodes a frame.
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_s) * 1000.0)
    ok, frame = cap.read()

    if not ok:
        return None

    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def ssim_gray(a, b):
    """SSIM-like structural similarity calculation on grayscale images."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)

    var_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    var_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    cov_ab = (
        cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    )

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    score = (
        (2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)
    ) / (
        (mu_a * mu_a + mu_b * mu_b + c1)
        * (var_a + var_b + c2)
    )

    return float(np.mean(score))


def compare_images(file_a, file_b, times_a, times_b, pairs):
    if cv2 is None or not pairs:
        return {
            "image_frames_compared": 0,
            "pixel_exact_fraction": None,
            "image_mae_mean": None,
            "image_ssim_mean": None,
        }

    # Select up to eight timestamps spread across the matched sequence.
    selected = np.unique(
        np.linspace(
            0, len(pairs) - 1,
            num=min(MAX_IMAGE_COMPARISONS, len(pairs)),
            dtype=int,
        )
    )

    cap_a = cv2.VideoCapture(str(file_a))
    cap_b = cv2.VideoCapture(str(file_b))

    exact = []
    maes = []
    ssims = []

    try:
        for k in selected:
            i, j = pairs[k]
            image_a = read_gray_frame(cap_a, times_a[i])
            image_b = read_gray_frame(cap_b, times_b[j])

            if image_a is None or image_b is None:
                continue

            exact.append(float(np.array_equal(image_a, image_b)))
            maes.append(float(np.mean(
                np.abs(image_a.astype(np.float32)
                       - image_b.astype(np.float32))
            )))
            ssims.append(ssim_gray(image_a, image_b))
    finally:
        cap_a.release()
        cap_b.release()

    return {
        "image_frames_compared": len(ssims),
        "pixel_exact_fraction": (
            float(np.mean(exact)) if exact else None
        ),
        "image_mae_mean": float(np.mean(maes)) if maes else None,
        "image_ssim_mean": float(np.mean(ssims)) if ssims else None,
    }


def main():
    if not METADATA.is_file():
        raise FileNotFoundError(METADATA)
    if not VISUAL_CONFIG.is_file():
        raise FileNotFoundError(VISUAL_CONFIG)

    config = load_json(VISUAL_CONFIG)
    index_path = Path(config["index"]).expanduser().resolve()

    if not index_path.is_file():
        raise FileNotFoundError(index_path)

    with METADATA.open(encoding="utf-8") as f:
        metadata = json.load(f)

    metadata_by_file = {r["file"]: r for r in metadata}

    index_rows = []
    with index_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                index_rows.append(json.loads(line))

    feature_by_file = {r["file"]: r for r in index_rows}

    # Audit DEV only. An audio-only example has audio manipulation
    # but no visual manipulation.
    candidates = [
        r for r in metadata
        if r["split"] == "dev"
        and r["modify_audio"] is True
        and r["modify_video"] is False
        and r.get("original")
    ]

    print("Metadata records:", len(metadata))
    print("Visual index:", index_path)
    print("DEV audio-only pairs to check:", len(candidates))
    print("Raw video root:", ROOT)
    print("OpenCV available:", cv2 is not None)

    rows = []
    skipped = []

    for n, manipulated in enumerate(candidates, 1):
        file_a = manipulated["file"]
        original = manipulated["original"]

        if original not in metadata_by_file:
            skipped.append((file_a, "original missing from metadata"))
            continue

        original_meta = metadata_by_file[original]

        # This audit is for pairs whose linked original is in DEV too.
        if original_meta["split"] != "dev":
            skipped.append((file_a, "original is not in DEV"))
            continue

        if file_a not in feature_by_file or original not in feature_by_file:
            skipped.append((file_a, "feature index row missing"))
            continue

        raw_a = ROOT / file_a
        raw_b = ROOT / original

        if not raw_a.is_file() or not raw_b.is_file():
            skipped.append((file_a, "raw video missing"))
            continue

        fa, ta = load_feature(index_path, feature_by_file[file_a])
        fb, tb = load_feature(index_path, feature_by_file[original])

        aligned = align_timestamps(ta, tb, TIME_TOLERANCE_S)

        result = {
            "file": file_a,
            "original": original,
            "modified_frames": len(ta),
            "original_frames": len(tb),
            "matched_timestamps": len(aligned),
            "feature_match_coverage": (
                len(aligned) / max(1, min(len(ta), len(tb)))
            ),
            "feature_cosine_mean": None,
            "feature_cosine_median": None,
            "feature_cosine_p10": None,
            "feature_fraction_cosine_ge_0_99": None,
            "feature_fraction_cosine_ge_0_999": None,
        }

        if aligned:
            cosine = cosine_by_frame(fa, fb, aligned)
            result.update({
                "feature_cosine_mean": float(np.mean(cosine)),
                "feature_cosine_median": float(np.median(cosine)),
                "feature_cosine_p10": float(np.percentile(cosine, 10)),
                "feature_fraction_cosine_ge_0_99": float(
                    np.mean(cosine >= 0.99)
                ),
                "feature_fraction_cosine_ge_0_999": float(
                    np.mean(cosine >= 0.999)
                ),
            })

            result.update(compare_images(
                raw_a, raw_b, ta, tb, aligned
            ))
        else:
            result.update({
                "image_frames_compared": 0,
                "pixel_exact_fraction": None,
                "image_mae_mean": None,
                "image_ssim_mean": None,
            })

        rows.append(result)

        if n % 25 == 0:
            print(f"Checked {n}/{len(candidates)} audio-only DEV videos")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        with OUTPUT.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    good = [r for r in rows if r["matched_timestamps"] > 0]

    print("\n=== AUDIT SUMMARY ===")
    print("Pairs checked:", len(rows))
    print("Pairs skipped:", len(skipped))
    print("Pairs with matched timestamps:", len(good))

    if good:
        for key in (
            "feature_cosine_mean",
            "feature_cosine_median",
            "feature_fraction_cosine_ge_0_99",
            "image_ssim_mean",
            "image_mae_mean",
            "pixel_exact_fraction",
        ):
            values = [
                r[key] for r in good
                if r.get(key) is not None
            ]
            if values:
                print(
                    f"{key}: median across pairs="
                    f"{np.median(values):.6f}; "
                    f"mean across pairs={np.mean(values):.6f}"
                )

    if skipped:
        print("\nSkipped examples:")
        for file_name, reason in skipped[:15]:
            print(f"  {file_name}: {reason}")

    print("\nCSV report:", OUTPUT.resolve())


if __name__ == "__main__":
    main()
