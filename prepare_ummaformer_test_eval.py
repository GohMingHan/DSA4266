#!/usr/bin/env python3
"""Extend the existing LAV-DF 4,500 train/dev cohort with its fixed 500 test IDs,
resume RGB+Farneback and TSN-flow feature extraction, and evaluate UMMAFormer.

This deliberately does not resplit the dataset, retrain the model, tune a threshold,
or overwrite the existing train/dev cohort or its checkpoints.
Run with the user's existing `ummaformer` environment and trusted project scripts.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any


def relpath(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected nonempty relative file path, got {value!r}")
    p = PurePosixPath(value.replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or ":" in str(p) or "\x00" in str(p):
        raise ValueError(f"Unsafe relative file path: {value!r}")
    return p.as_posix()


def merge_periods(value: Any) -> list[list[float]]:
    if not isinstance(value, list):
        raise ValueError("fake_periods must be a list")
    pairs: list[tuple[float, float]] = []
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"Invalid fake interval: {pair!r}")
        a, b = float(pair[0]), float(pair[1])
        if not (a >= 0 and b > a and a < float("inf") and b < float("inf")):
            raise ValueError(f"Invalid fake interval: {pair!r}")
        pairs.append((a, b))
    out: list[list[float]] = []
    for a, b in sorted(pairs):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path}, line {line_no}: {e}") from e
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object in {path}, line {line_no}")
            result.append(item)
    return result


def load_metadata(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(obj, list) or not all(isinstance(x, dict) for x in obj):
        raise ValueError("Expected reduced LAV-DF metadata as a JSON list of objects")
    return obj


def source_group(row: dict[str, Any]) -> str:
    return relpath(row.get("original") or row.get("source_group") or row["file"])


def bool_value(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    raise ValueError(f"Expected boolean {field}, got {value!r}")


def build_full_cohort(metadata_path: Path, old_cohort_path: Path, out_path: Path) -> dict[str, Any]:
    metadata = load_metadata(metadata_path)
    old_rows = read_jsonl(old_cohort_path)
    meta_by_file: dict[str, dict[str, Any]] = {}
    for r in metadata:
        name = relpath(r.get("file"))
        if name in meta_by_file:
            raise ValueError(f"Duplicate file in metadata: {name}")
        meta_by_file[name] = r
        if r.get("split") not in ("train", "dev", "test"):
            raise ValueError(f"Unknown split for {name}: {r.get('split')!r}")

    metadata_counts = Counter(str(r["split"]) for r in metadata)
    expected_meta = Counter({"train": 4000, "dev": 500, "test": 500})
    if len(metadata) != 5000 or metadata_counts != expected_meta:
        raise ValueError(f"Metadata does not match expected 5,000-video allocation: {dict(metadata_counts)}")

    old_by_file: dict[str, dict[str, Any]] = {}
    for r in old_rows:
        name = relpath(r.get("file"))
        if name in old_by_file:
            raise ValueError(f"Duplicate video in current cohort: {name}")
        old_by_file[name] = r

    old_counts = Counter(str(r.get("split")) for r in old_rows)
    if len(old_rows) != 4500 or old_counts != Counter({"train": 4000, "dev": 500}):
        raise ValueError(f"Expected current cohort to contain 4,000 train + 500 dev only; got {len(old_rows)} {dict(old_counts)}")

    # Validate the existing 4,500 IDs and labels against the source metadata. Keep
    # their rows unchanged so cached train/dev features retain the same identities.
    for name, existing in old_by_file.items():
        meta = meta_by_file.get(name)
        if meta is None:
            raise ValueError(f"Existing cohort video is absent from metadata: {name}")
        if str(existing.get("split")) != str(meta.get("split")):
            raise ValueError(f"Split mismatch for {name}")
        if bool_value(existing.get("modify_audio"), "modify_audio") != bool_value(meta.get("modify_audio"), "modify_audio"):
            raise ValueError(f"modify_audio mismatch for {name}")
        if bool_value(existing.get("modify_video"), "modify_video") != bool_value(meta.get("modify_video"), "modify_video"):
            raise ValueError(f"modify_video mismatch for {name}")
        if source_group(existing) != source_group(meta):
            raise ValueError(f"source_group/original mismatch for {name}: {source_group(existing)} vs {source_group(meta)}")
        if merge_periods(existing.get("fake_periods", [])) != merge_periods(meta.get("fake_periods", [])):
            raise ValueError(f"fake_periods mismatch for {name}")

    test_rows_meta = sorted((r for r in metadata if r["split"] == "test"), key=lambda r: relpath(r["file"]))
    if any(relpath(r["file"]) in old_by_file for r in test_rows_meta):
        raise ValueError("Some test IDs are already present in the train/dev manifest; refusing duplication")

    category_counts = Counter(
        (bool_value(r.get("modify_audio"), "modify_audio"), bool_value(r.get("modify_video"), "modify_video"))
        for r in test_rows_meta
    )
    expected_categories = {
        (False, False): 125,
        (True, False): 125,
        (False, True): 125,
        (True, True): 125,
    }
    if dict(category_counts) != expected_categories:
        raise ValueError(f"Test category allocation is not the expected four groups of 125: {dict(category_counts)}")

    full_rows = list(old_rows)  # preserve all 4,500 existing train/dev lines semantically
    for r in test_rows_meta:
        name = relpath(r["file"])
        ma = bool_value(r.get("modify_audio"), "modify_audio")
        mv = bool_value(r.get("modify_video"), "modify_video")
        periods = r.get("fake_periods", [])
        if bool(ma or mv) != bool(merge_periods(periods)):
            raise ValueError(f"Modality flags and intervals disagree for {name}")
        full_rows.append({
            "file": name,
            "split": "test",
            "source_group": source_group(r),
            "modify_audio": ma,
            "modify_video": mv,
            "fake_periods": periods,
            "metadata_duration_s": float(r.get("duration", 0)),
        })

    full_counts = Counter(str(r["split"]) for r in full_rows)
    if len(full_rows) != 5000 or full_counts != expected_meta:
        raise AssertionError(f"Internal final cohort validation failed: {len(full_rows)}, {dict(full_counts)}")

    groups: dict[str, str] = {}
    for r in full_rows:
        name, split = relpath(r["file"]), r["split"]
        if PurePosixPath(name).parts[0] != split:
            raise ValueError(f"Split/file directory mismatch: {name}")
        group = source_group(r)
        if group in groups and groups[group] != split:
            raise ValueError(f"Source group crosses splits: {group}")
        groups[group] = split

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in full_rows)
    if out_path.exists():
        if out_path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"Refusing to overwrite a different cohort file: {out_path}. Choose another --full-cohort path.")
    else:
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, out_path)

    return {
        "metadata": str(metadata_path),
        "original_cohort": str(old_cohort_path),
        "full_cohort": str(out_path),
        "metadata_counts": dict(metadata_counts),
        "original_counts": dict(old_counts),
        "full_counts": dict(full_counts),
        "test_category_counts_audio_video": {f"audio={int(a)},video={int(v)}": n for (a, v), n in sorted(category_counts.items())},
        "validated_old_train_dev_rows": len(old_rows),
        "added_test_rows": len(test_rows_meta),
    }


def locate_data_root(given: Path | None, metadata: list[dict[str, Any]]) -> Path:
    test_paths = [relpath(r["file"]) for r in metadata if r.get("split") == "test"]
    if given is not None:
        root = given.expanduser().resolve()
        absent = [s for s in test_paths if not (root / s).is_file()]
        if absent:
            raise FileNotFoundError(f"--data-root misses {len(absent)} test videos, e.g. {absent[:3]}: {root}")
        return root
    candidates = [
        Path.home() / "datasets/LAV-DF-5000",
        Path("/mnt/c/Users/wuche/Downloads/LAV-DF-5000/LAV-DF-5000"),
        Path("/mnt/c/Users/wuche/Downloads/LAV-DF-5000"),
        Path("/mnt/c/Users/wuche/Downloads/LAV-DF"),
    ]
    for root in candidates:
        root = root.expanduser().resolve()
        if root.is_dir() and all((root / s).is_file() for s in test_paths):
            return root
    sample = test_paths[:5]
    raise FileNotFoundError(
        "Could not find a dataset root containing every test MP4. Pass --data-root explicitly. "
        f"Tried: {[str(x) for x in candidates]}; sample relative paths: {sample}"
    )


def backup_manifest(cache: Path, backup_dir: Path, label: str) -> Path:
    manifest = cache / "manifest.json"
    config = cache / "config.json"
    if not config.is_file() or not manifest.is_file():
        raise FileNotFoundError(f"Expected existing completed feature cache with config.json and manifest.json: {cache}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    dst = backup_dir / f"{label}_manifest_before_test_expansion.json"
    shutil.copy2(manifest, dst)
    return dst


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> int:
    home = Path.home()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, default=home / "datasets/LAV-DF-5000/metadata.json")
    p.add_argument("--cohort-videos", type=Path, default=home / "lavdf5000_visual_manifest/videos.jsonl")
    p.add_argument("--full-cohort", type=Path, default=home / "lavdf5000_visual_manifest_test/videos.jsonl")
    p.add_argument("--data-root", type=Path, help="Folder whose train/, dev/, test/ directories contain the MP4s; auto-detected if omitted")
    p.add_argument("--rgb-project", type=Path, default=home / "dsa4266/ummaformer_opencv_flow")
    p.add_argument("--tsn-project", type=Path, default=home / "dsa4266/ummaformer_tsn_flow")
    p.add_argument("--rgb-cache", type=Path, help="Defaults to <rgb-project>/features/rgb_flow_train_dev")
    p.add_argument("--tsn-cache", type=Path, help="Defaults to <tsn-project>/features/tsn_flow_train_dev_explicit_ffmpeg")
    p.add_argument("--flow-weights", type=Path, help="Defaults to <tsn-project>/weights/tsn_flow_activitynet.pth")
    p.add_argument("--upstream", type=Path, help="Defaults to <rgb-project>/upstream")
    p.add_argument("--checkpoint", type=Path, help="Final UMMAFormer TSN-flow best_video_auc.pt checkpoint (required unless --prepare-only)")
    p.add_argument("--eval-out", type=Path, help="New empty evaluation folder; defaults to a timestamped folder under <tsn-project>/evaluations")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sample-fps", type=float, default=10.0)
    p.add_argument("--rgb-batch-size", type=int, default=32)
    p.add_argument("--tsn-batch-size", type=int, default=8)
    p.add_argument("--prepare-only", action="store_true", help="Only validate/build the 5,000-row cohort JSONL; do not extract or evaluate")
    p.add_argument("--skip-rgb-extraction", action="store_true", help="Use only if RGB cache already has a validated train/dev/test manifest")
    p.add_argument("--skip-tsn-extraction", action="store_true", help="Use only if TSN cache already has a validated train/dev/test manifest")
    a = p.parse_args()

    metadata_path = a.metadata.expanduser().resolve()
    old_cohort = a.cohort_videos.expanduser().resolve()
    full_cohort = a.full_cohort.expanduser().resolve()
    rgb_project = a.rgb_project.expanduser().resolve()
    tsn_project = a.tsn_project.expanduser().resolve()
    rgb_cache = (a.rgb_cache or (rgb_project / "features/rgb_flow_train_dev")).expanduser().resolve()
    tsn_cache = (a.tsn_cache or (tsn_project / "features/tsn_flow_train_dev_explicit_ffmpeg")).expanduser().resolve()
    flow_weights = (a.flow_weights or (tsn_project / "weights/tsn_flow_activitynet.pth")).expanduser().resolve()
    upstream = (a.upstream or (rgb_project / "upstream")).expanduser().resolve()
    checkpoint = a.checkpoint.expanduser().resolve() if a.checkpoint else None

    for path in (metadata_path, old_cohort):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
    if not a.prepare_only and (checkpoint is None or not checkpoint.is_file()):
        raise FileNotFoundError(f"Provide an existing --checkpoint; not found: {checkpoint}")
    metadata_rows = load_metadata(metadata_path)

    report = build_full_cohort(metadata_path, old_cohort, full_cohort)
    print("\n=== COHORT VALIDATION ===")
    print(json.dumps(report, indent=2))

    if a.prepare_only:
        print("\nPrepared the fixed train/dev/test cohort only. No media paths were checked, no features were extracted, and no evaluation was run.")
        return 0

    data_root = locate_data_root(a.data_root, metadata_rows)
    print("Detected/selected data root:", data_root)

    rgb_script = rgb_project / "extract_rgb_flow.py"
    tsn_script = tsn_project / "extract_tsn_flow.py"
    eval_script = tsn_project / "run_experiment.py"
    for f in (rgb_script, tsn_script, eval_script, flow_weights):
        if not f.exists():
            raise FileNotFoundError(f"Required project file/weight missing: {f}")
    if not upstream.is_dir():
        raise FileNotFoundError(f"UMMAFormer upstream directory missing: {upstream}")
    if a.sample_fps < 0 or a.rgb_batch_size < 1 or a.tsn_batch_size < 1:
        raise ValueError("Invalid sample rate or extraction batch sizes")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = home / "dsa4266" / "test_eval_manifest_backups" / stamp
    old_rgb_backup = None
    old_tsn_backup = None

    if not a.skip_rgb_extraction:
        old_rgb_backup = backup_manifest(rgb_cache, backup_dir, "rgb_flow")
        rgb_cmd = [
            sys.executable, str(rgb_script),
            "--data-root", str(data_root),
            "--cohort-videos", str(full_cohort),
            "--metadata", str(metadata_path),
            "--out", str(rgb_cache),
            "--splits", "train", "dev", "test",
            "--sample-fps", str(a.sample_fps),
            "--device", a.device,
            "--encode-batch-size", str(a.rgb_batch_size),
        ]
        try:
            run_cmd(rgb_cmd, rgb_project)
        except BaseException:
            # The extractor removes the old manifest before publishing a new one.
            # Restore the previous DEV/TRAIN-only manifest if expansion fails.
            if old_rgb_backup and old_rgb_backup.is_file():
                shutil.copy2(old_rgb_backup, rgb_cache / "manifest.json")
                print(f"Restored original RGB manifest from {old_rgb_backup}", file=sys.stderr)
            raise

    rgb_manifest = rgb_cache / "manifest.json"
    if not rgb_manifest.is_file():
        raise FileNotFoundError(f"Train/dev/test RGB manifest missing: {rgb_manifest}")
    rgb_manifest_data = json.loads(rgb_manifest.read_text(encoding="utf-8"))
    rgb_counts = Counter(str(r["split"]) for r in rgb_manifest_data.get("videos", []))
    if rgb_counts != Counter({"train": 4000, "dev": 500, "test": 500}):
        raise ValueError(f"RGB manifest does not have expected 5k split counts: {dict(rgb_counts)}")

    if not a.skip_tsn_extraction:
        old_tsn_backup = backup_manifest(tsn_cache, backup_dir, "tsn_flow")
        tsn_cmd = [
            sys.executable, str(tsn_script),
            "--reference-manifest", str(rgb_manifest),
            "--data-root", str(data_root),
            "--out", str(tsn_cache),
            "--flow-weights", str(flow_weights),
            "--flow-method", "farneback",
            "--splits", "train", "dev", "test",
            "--device", a.device,
            "--encode-batch-size", str(a.tsn_batch_size),
        ]
        try:
            run_cmd(tsn_cmd, tsn_project)
        except BaseException:
            if old_tsn_backup and old_tsn_backup.is_file():
                shutil.copy2(old_tsn_backup, tsn_cache / "manifest.json")
                print(f"Restored original TSN manifest from {old_tsn_backup}", file=sys.stderr)
            raise

    tsn_manifest = tsn_cache / "manifest.json"
    if not tsn_manifest.is_file():
        raise FileNotFoundError(f"Train/dev/test TSN manifest missing: {tsn_manifest}")
    tsn_manifest_data = json.loads(tsn_manifest.read_text(encoding="utf-8"))
    tsn_counts = Counter(str(r["split"]) for r in tsn_manifest_data.get("videos", []))
    if tsn_counts != Counter({"train": 4000, "dev": 500, "test": 500}):
        raise ValueError(f"TSN manifest does not have expected 5k split counts: {dict(tsn_counts)}")

    eval_out = (a.eval_out or (tsn_project / "evaluations" / f"tsn_flow_test_{stamp}")).expanduser().resolve()
    if eval_out.exists() and any(eval_out.iterdir()):
        raise FileExistsError(f"Evaluation output must be new/empty: {eval_out}")
    eval_cmd = [
        sys.executable, str(eval_script), "evaluate",
        "--upstream", str(upstream),
        "--manifest", str(tsn_manifest),
        "--checkpoint", str(checkpoint),
        "--split", "test",
        "--out", str(eval_out),
        "--device", a.device,
    ]
    run_cmd(eval_cmd, tsn_project)

    print("\n=== TEST EVALUATION COMPLETE ===")
    print("Test manifest:", tsn_manifest)
    print("Checkpoint:", checkpoint)
    print("Evaluation outputs:", eval_out)
    print("Manifest backups:", backup_dir)
    print("The evaluator reuses the checkpoint's DEV-selected threshold; it does not tune on TEST.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
