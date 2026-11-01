#!/usr/bin/env python3
"""Create a self-contained, smaller LAV-DF directory without altering the source.

Default: 2,000 train + 400 dev + 400 test, with equal modality categories.
One genuine clip and one audio-only, visual-only and joint fake are chosen per
source family. This keeps all nonempty `original` links inside the 2,800 records.
Files are copied byte-for-byte, NOT re-encoded. Python 3.10+; standard library only.

This defines a custom source-paired subset, not the full benchmark. If enough
complete source families are unavailable, the script stops before copying.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import socket
import sys
from typing import Any, Iterator

VERSION = 1
SPLITS = ("train", "dev", "test")
CATEGORIES = ("real", "audio_only", "visual_only", "both")
STATE = ".lavdf_subset_state"
CHUNK = 4 * 1024 * 1024


def json_bytes(value: Any, compact: bool = False) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True if compact else False,
                      indent=None if compact else 2,
                      separators=(",", ":") if compact else None).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(json_bytes(value, compact=True)).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"Refusing to overwrite a symlink: {path}")
    tmp = path.with_name(path.name + ".tmp")
    if tmp.is_symlink():
        raise ValueError(f"Unsafe temporary path: {tmp}")
    with tmp.open("wb") as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, json_bytes(value))


def relative_name(value: Any, field: str = "file") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty relative path.")
    value = value.replace("\\", "/")
    p = PurePosixPath(value)
    if (p.is_absolute() or not p.parts or ".." in p.parts or ":" in value
            or "\x00" in value):
        raise ValueError(f"Unsafe {field}: {value!r}")
    if p.parts[0] == STATE:
        raise ValueError(f"{field} collides with the builder's state directory.")
    return p.as_posix()


def safe_path(root: Path, relative: str) -> Path:
    """Require physical paths within root; do not follow symlinks in subset data."""
    root = root.resolve()
    parts = PurePosixPath(relative_name(relative)).parts
    p = root
    for part in parts:
        p = p / part
        if p.is_symlink():
            raise ValueError(f"Symlink not allowed in dataset/copy path: {p}")
    if not p.resolve().is_relative_to(root):
        raise ValueError(f"Path escapes root: {p}")
    return p


def boolean(value: Any, field: str) -> bool:
    if isinstance(value, bool) or type(value) is int and value in (0, 1):
        return bool(value)
    raise ValueError(f"{field} must be boolean or integer 0/1.")


def category(row: dict) -> str:
    ma, mv = bool(row["modify_audio"]), bool(row["modify_video"])
    return "both" if ma and mv else "audio_only" if ma else "visual_only" if mv else "real"


def validate_metadata(raw: Any) -> dict[str, dict]:
    """Validate structurally; preserve ALL original keys and values unchanged."""
    if not isinstance(raw, list) or not raw:
        raise ValueError("Expected a nonempty JSON list of LAV-DF metadata records.")
    by_name, folded = {}, set()
    for index, row in enumerate(raw):
        try:
            if not isinstance(row, dict):
                raise ValueError("Record is not an object.")
            name = relative_name(row["file"])
            if name.casefold() in folded:
                raise ValueError(f"Duplicate/case-colliding file path: {name}")
            folded.add(name.casefold())
            if row["split"] not in SPLITS:
                raise ValueError("Expected the original train/dev/test split names.")
            ma = boolean(row["modify_audio"], "modify_audio")
            mv = boolean(row["modify_video"], "modify_video")
            duration = float(row["duration"])
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("duration must be positive and finite.")
            periods = row["fake_periods"]
            if not isinstance(periods, list) or bool(periods) != bool(ma or mv):
                raise ValueError("fake_periods and modality flags disagree.")
            for pair in periods:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError(f"Invalid temporal interval: {pair!r}")
                a, b = map(float, pair)
                if not math.isfinite(a) or not math.isfinite(b) or a < 0 or b <= a:
                    raise ValueError(f"Invalid temporal interval: {pair!r}")
                # No duration clamp: availability is audited at extraction time.
            if row.get("original") not in (None, ""):
                relative_name(row["original"], "original")
            # Ensures unknown extra metadata also round-trips as JSON.
            json_bytes(row)
            by_name[name] = row
        except (KeyError, TypeError, ValueError) as exc:
            filename = row.get("file") if isinstance(row, dict) else None
            raise ValueError(f"Metadata row {index}, file={filename!r}: {exc}") from exc
    return by_name


def source_families(by_name: dict[str, dict]) -> dict[str, str]:
    """Connected file-original families; these are not person identities."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        top = x
        while parent[top] != top:
            top = parent[top]
        while parent[x] != x:
            nxt = parent[x]
            parent[x] = top
            x = nxt
        return top

    for name, row in by_name.items():
        original = relative_name(row["original"], "original") if row.get("original") else name
        a, b = find(name), find(original)
        if a != b:
            parent[max(a, b)] = min(a, b)
    families = {name: find(name) for name in by_name}
    seen = {}
    for name, row in by_name.items():
        family = families[name]
        if family in seen and seen[family] != row["split"]:
            raise ValueError(f"Source family crosses official splits: {family}. No automatic split repair.")
        seen[family] = row["split"]
    return families


def rank(seed: int, *parts: str) -> str:
    return digest([seed, *parts])


def select_subset(raw: Any, sizes: dict[str, int], seed: int) -> tuple[list[dict], dict]:
    """Exact four-category budgets with the corresponding originals included.

    Eligible quartet: one real with no external original dependency, plus a
    directly linked derivative in EACH of the three manipulated categories.
    This deliberately differs from the earlier metadata-only helper, which
    selected the categories independently and could leave original links dangling.
    """
    if not any(sizes.values()) or any(type(sizes.get(s)) is not int or sizes[s] < 0 or sizes[s] % 4 for s in SPLITS):
        raise ValueError("Each split size must be zero or a positive multiple of four.")
    by_name = validate_metadata(raw)
    families = source_families(by_name)
    real_sources = {}
    for name, row in by_name.items():
        origin = relative_name(row["original"], "original") if row.get("original") else name
        if category(row) == "real" and origin == name:
            real_sources[name] = row
    derivatives: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    ineligible = Counter()
    for name, row in by_name.items():
        cat = category(row)
        if cat == "real":
            continue
        if not row.get("original"):
            ineligible["fake_missing_original_field"] += 1
            continue
        origin = relative_name(row["original"], "original")
        if origin not in real_sources:
            ineligible["fake_not_directly_linked_to_self_contained_real"] += 1
            continue
        derivatives[origin][cat].append(name)
    eligible = {s: [] for s in SPLITS}
    for name, row in real_sources.items():
        if all(derivatives[name].get(cat) for cat in CATEGORIES[1:]):
            eligible[row["split"]].append(name)
    available = {s: len(eligible[s]) for s in SPLITS}
    for split in SPLITS:
        need = sizes[split] // 4
        if need > available[split]:
            raise ValueError(
                f"{split}: need {need} complete source families for {sizes[split]} videos, "
                f"but found {available[split]}. Available complete families: {available}. "
                f"Use --{split} <= {4*available[split]} (multiple of 4), or retain the full "
                "source dataset and use the older metadata-only subset. No files were copied. "
                f"Unsupported-link counts: {dict(ineligible)}")
    selected, selected_originals = [], []
    for split in SPLITS:
        sources = sorted(eligible[split], key=lambda name: (rank(seed, split, name), name))[:sizes[split] // 4]
        for origin in sources:
            selected.append(by_name[origin])
            selected_originals.append(origin)
            for cat in CATEGORIES[1:]:
                chosen = min(derivatives[origin][cat], key=lambda name: (rank(seed, split, cat, name), name))
                selected.append(by_name[chosen])
    selected.sort(key=lambda r: (SPLITS.index(r["split"]), relative_name(r["file"])))
    names = {relative_name(r["file"]) for r in selected}
    for row in selected:
        if row.get("original") and relative_name(row["original"], "original") not in names:
            raise AssertionError("Internal error: dangling original link.")
    counts = {s: {c: sum(r["split"] == s and category(r) == c for r in selected) for c in CATEGORIES} for s in SPLITS}
    report = {
        "protocol": "source_complete_four_category_LAVDF_subset_v1", "seed": seed,
        "split_counts": sizes, "category_counts": counts, "selected_videos": len(selected),
        "selected_source_families": len({families[n] for n in names}),
        "eligible_complete_families": available, "selected_real_sources": selected_originals,
        "ineligible_link_counts": dict(ineligible), "all_original_links_inside_subset": True,
        "any_fake_prevalence": 0.75,
        "annotation_end_exceeds_metadata_duration": sum(any(float(b) > float(r["duration"]) + .1 for a, b in r["fake_periods"]) for r in selected),
        "notes": [
            "This is a custom source-paired benchmark, not the full official distribution.",
            "Four equal modality categories produce 75% fake / 25% real for the overall target.",
            "Source families are not identities; no new random train/dev/test assignment was made.",
            "Selecting only families with all three fake categories changes the sampled population.",
            "Metadata rows and fake intervals are preserved, not corrected or clipped.",
            "Copy verification checks bytes, not video decoding or temporal annotation accuracy.",
            "Unknown extra metadata fields are preserved but not interpreted as extra asset dependencies.",
        ],
    }
    return selected, report


def find_metadata(root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
    else:
        path = root / "metadata.min.json"
        if not path.is_file():
            path = root / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Metadata not found: {path}. Pass --metadata /actual/path.json")
    return path


def roots_valid(root: Path, out: Path) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"Source directory not found: {root}")
    if out == root or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError("Output must be separate from the source: not identical, inside it, or its ancestor.")
    if out.exists() and not out.is_dir():
        raise ValueError("Output path is not a directory.")


def stat_signature(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Selected file missing: {path}. No silent replacement.")
    s = path.stat()
    if s.st_size <= 0:
        raise ValueError(f"Selected source file is empty: {path}")
    return {"bytes": s.st_size, "mtime_ns": s.st_mtime_ns}


def plan_copy(root: Path, metadata: Path, sizes: dict[str, int], seed: int) -> tuple[dict, list[dict]]:
    source_bytes = metadata.read_bytes()
    raw = json.loads(source_bytes.decode("utf-8-sig"))
    selected, selection = select_subset(raw, sizes, seed)
    entries = []
    for row in selected:
        name = relative_name(row["file"])
        entries.append({"file": name, "kind": "video", **stat_signature(safe_path(root, name))})
    # Carry root-level provenance/licensing text. Do not copy full-dataset lists,
    # archives, caches, or other media. Preserve the dataset's applicable terms.
    prefixes = ("readme", "license", "licence", "notice", "terms_and_conditions", "terms-and-conditions")
    for p in sorted(root.iterdir()):
        if p.is_file() and p.name.lower().startswith(prefixes) and p.suffix.lower() in ("", ".md", ".txt", ".rst"):
            entries.append({"file": p.name, "kind": "documentation", **stat_signature(safe_path(root, p.name))})
    if metadata.name in {"subset_report.json", "plan.json", "selected_records.json"}:
        raise ValueError("Input metadata basename is reserved by the subset builder; rename that source metadata copy first.")
    metadata_names = sorted({"metadata.min.json", "metadata.json", metadata.name})
    reserved = set(metadata_names) | {"subset_report.json", "selected_videos.csv"}
    if any(e["file"] in reserved for e in entries):
        raise ValueError("A source file collides with the subset metadata/report filenames.")
    if len({e["file"].casefold() for e in entries}) != len(entries):
        raise ValueError("Output path collision.")
    plan = {
        "version": VERSION, "source_root": str(root), "source_metadata": str(metadata),
        "source_metadata_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "selection": selection, "metadata_names": metadata_names,
        "records_sha256": digest(selected), "entries": entries,
        "selected_video_bytes": sum(e["bytes"] for e in entries if e["kind"] == "video"),
        "copy_bytes": sum(e["bytes"] for e in entries),
    }
    plan["plan_id"] = digest(plan)
    return plan, selected


def nearest_existing(path: Path) -> Path:
    while not path.exists():
        path = path.parent
    return path


def brief(plan: dict, out: Path) -> dict:
    return {
        "output": str(out), "videos": plan["selection"]["selected_videos"],
        "split_counts": plan["selection"]["split_counts"],
        "category_counts": plan["selection"]["category_counts"],
        "source_families": plan["selection"]["selected_source_families"],
        "original_links_self_contained": True,
        "video_bytes_to_copy": plan["selected_video_bytes"],
        "video_gib_to_copy": plan["selected_video_bytes"] / 1024**3,
        "documentation_bytes": plan["copy_bytes"] - plan["selected_video_bytes"],
        "destination_filesystem_free_gib": shutil.disk_usage(nearest_existing(out)).free / 1024**3,
        "note": "The earlier 11.1 GiB figure estimates extracted PNGs, NOT these original video files.",
    }


@contextmanager
def lock_output(out: Path) -> Iterator[None]:
    state = out / STATE
    if state.is_symlink():
        raise ValueError("State directory cannot be a symlink.")
    state.mkdir(parents=True, exist_ok=True)
    lock = state / "copy.lock"
    try:
        with lock.open("x", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "host": socket.gethostname()}, f)
    except FileExistsError:
        raise RuntimeError(f"Output is locked: {lock}. Verify no other builder is running before removing a stale lock.") from None
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def receipt_path(out: Path, relative: str) -> Path:
    return out / STATE / "receipts" / (hashlib.sha256(relative.encode()).hexdigest() + ".json")


def check_source(source: Path, entry: dict) -> None:
    sig = stat_signature(source)
    if sig != {k: entry[k] for k in ("bytes", "mtime_ns")}:
        raise ValueError(f"Source file changed since selection: {source}. Use a new output.")


def copy_one(root: Path, out: Path, entry: dict) -> str:
    rel = entry["file"]
    source, target = safe_path(root, rel), safe_path(out, rel)
    check_source(source, entry)
    receipt_file = receipt_path(out, rel)
    if receipt_file.is_symlink():
        raise ValueError("Receipt must not be a symlink.")
    receipt = json.loads(receipt_file.read_text(encoding="utf-8")) if receipt_file.is_file() else None
    if target.exists():
        if not target.is_file() or target.stat().st_size != entry["bytes"]:
            raise ValueError(f"Existing destination has the wrong size: {target}. It was NOT overwritten.")
        # Receipts carry a source-content hash established when copying. Rehash
        # both sides on resume rather than trusting an existing filename/size.
        src_hash, dst_hash = sha256_file(source), sha256_file(target)
        check_source(source, entry)
        if src_hash != dst_hash or receipt and receipt.get("sha256") != dst_hash:
            raise ValueError(f"Checksum mismatch for existing copy: {target}. It was NOT overwritten.")
        atomic_json(receipt_file, {"file": rel, "bytes": entry["bytes"], "sha256": dst_hash})
        return "reused"
    partial = out / STATE / "partials" / (hashlib.sha256(rel.encode()).hexdigest() + ".part")
    partial.parent.mkdir(parents=True, exist_ok=True)
    if partial.is_symlink():
        raise ValueError("Partial-file path cannot be a symlink.")
    h = hashlib.sha256()
    copied = 0
    with source.open("rb") as src, partial.open("wb") as dst:
        for block in iter(lambda: src.read(CHUNK), b""):
            dst.write(block)
            h.update(block)
            copied += len(block)
        dst.flush()
        os.fsync(dst.fileno())
    check_source(source, entry)
    if copied != entry["bytes"] or sha256_file(partial) != h.hexdigest():
        raise ValueError(f"Copy verification failed: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError(f"Destination appeared during copying: {target}")
    partial.replace(target)
    # Preserve mtime as convenience, not as a substitute for checksum validation.
    os.utime(target, ns=(entry["mtime_ns"], entry["mtime_ns"]))
    atomic_json(receipt_file, {"file": rel, "bytes": copied, "sha256": h.hexdigest()})
    return "copied"


def existing_plan_matches(out: Path, plan: dict) -> None:
    plan_path = out / STATE / "plan.json"
    if plan_path.is_file():
        old = json.loads(plan_path.read_text(encoding="utf-8"))
        if old != plan:
            raise ValueError("Destination belongs to a different source/selection/version. Use a new --out.")
    elif out.exists() and any(out.iterdir()):
        raise ValueError("Destination is nonempty and not a resumable subset. Use a new/empty --out.")


def perform_copy(root: Path, out: Path, plan: dict, selected: list[dict]) -> dict:
    roots_valid(root, out)
    existing_plan_matches(out, plan)
    expected = {e["file"] for e in plan["entries"]} | set(plan["metadata_names"]) | {"subset_report.json"}
    # A normal interruption can leave an atomic-publication temporary file.
    publication_temps = {name + ".tmp" for name in plan["metadata_names"] + ["subset_report.json"]}
    # Do not silently include unrelated files or accidentally create >2,800 videos.
    if out.exists():
        for p in out.rglob("*"):
            rel = p.relative_to(out).as_posix()
            if rel == STATE or rel.startswith(STATE + "/"):
                continue
            if p.is_symlink() or p.is_file() and rel not in expected | publication_temps:
                raise ValueError(f"Unexpected destination file/symlink: {p}. Use a dedicated output.")
    remaining = sum(e["bytes"] for e in plan["entries"] if not safe_path(out, e["file"]).is_file())
    free = shutil.disk_usage(nearest_existing(out)).free
    margin = max(64 * 1024**2, math.ceil(remaining * .05))
    if free < remaining + margin:
        raise OSError(f"Insufficient free space: need about {(remaining+margin)/1024**3:.2f} GiB including margin; have {free/1024**3:.2f} GiB.")
    outcomes = Counter()
    with lock_output(out):
        for name in publication_temps:
            safe_path(out, name).unlink(missing_ok=True)
        # Plan and selected rows survive Ctrl+C; final dataset metadata is only
        # published once ALL copies have been checked.
        atomic_json(out / STATE / "plan.json", plan)
        atomic_json(out / STATE / "selected_records.json", selected)
        try:
            for index, entry in enumerate(plan["entries"], 1):
                outcomes[copy_one(root, out, entry)] += 1
                if index == 1 or index % 25 == 0 or index == len(plan["entries"]):
                    print(f"Files {index}/{len(plan['entries'])}: copied={outcomes['copied']}, reused={outcomes['reused']}", flush=True)
            for name in plan["metadata_names"]:
                compact = name.endswith(".min.json")
                atomic_bytes(safe_path(out, name), json_bytes(selected, compact=compact))
            report = verify_subset(out, quiet=True)
            report.update({"source_root": str(root), "source_metadata_sha256": plan["source_metadata_sha256"],
                           "copy_outcomes_this_run": dict(outcomes), "selection": plan["selection"]})
            atomic_json(out / "subset_report.json", report)
            return report
        except KeyboardInterrupt:
            print("\nInterrupted. Completed files are retained. Rerun the identical command to resume; the current file restarts.", file=sys.stderr, flush=True)
            raise


def verify_subset(out: Path, quiet: bool = False) -> dict:
    """Verify independently of the old source folder (so the copy is portable)."""
    out = out.expanduser().resolve()
    plan = json.loads((out / STATE / "plan.json").read_text(encoding="utf-8"))
    selected = json.loads((out / STATE / "selected_records.json").read_text(encoding="utf-8"))
    without_id = {k: v for k, v in plan.items() if k != "plan_id"}
    if digest(without_id) != plan["plan_id"] or digest(selected) != plan["records_sha256"]:
        raise ValueError("Plan/selection checksum mismatch.")
    by_name = validate_metadata(selected)
    source_families(by_name)
    if len(by_name) != plan["selection"]["selected_videos"]:
        raise ValueError("Wrong selected video count.")
    video_names = {e["file"] for e in plan["entries"] if e["kind"] == "video"}
    if video_names != set(by_name):
        raise ValueError("Metadata and planned media disagree.")
    for row in selected:
        if row.get("original") and relative_name(row["original"], "original") not in by_name:
            raise ValueError(f"Dangling original reference in {row['file']}.")
    for name in plan["metadata_names"]:
        parsed = json.loads(safe_path(out, name).read_text(encoding="utf-8-sig"))
        if parsed != selected:
            raise ValueError(f"Published metadata does not match the subset: {name}")
    total = 0
    for index, entry in enumerate(plan["entries"], 1):
        target = safe_path(out, entry["file"])
        receipt = json.loads(receipt_path(out, entry["file"]).read_text(encoding="utf-8"))
        if (not target.is_file() or target.stat().st_size != entry["bytes"]
                or receipt.get("file") != entry["file"] or receipt.get("bytes") != entry["bytes"]
                or sha256_file(target) != receipt.get("sha256")):
            raise ValueError(f"Copy is missing/corrupt: {target}")
        total += entry["bytes"]
        if not quiet and (index % 100 == 0 or index == len(plan["entries"])):
            print(f"Verified {index}/{len(plan['entries'])} files", flush=True)
    allowed = {e["file"] for e in plan["entries"]} | set(plan["metadata_names"]) | {"subset_report.json"}
    for p in out.rglob("*"):
        rel = p.relative_to(out).as_posix()
        if rel == STATE or rel.startswith(STATE + "/"):
            continue
        if p.is_symlink() or p.is_file() and rel not in allowed:
            raise ValueError(f"Unexpected file in dataset folder: {p}")
    actual_counts = {s: sum(r["split"] == s for r in selected) for s in SPLITS}
    actual_categories = {s: {c: sum(r["split"] == s and category(r) == c for r in selected) for c in CATEGORIES} for s in SPLITS}
    if actual_counts != plan["selection"]["split_counts"] or actual_categories != plan["selection"]["category_counts"]:
        raise ValueError("Split/category counts do not match the plan.")
    return {"status": "complete_verified", "verified_utc": datetime.now(timezone.utc).isoformat(),
            "videos": len(by_name), "split_counts": actual_counts, "category_counts": actual_categories,
            "all_original_links_inside_subset": True, "metadata_fields_preserved": True,
            "verification": "SHA256 of destination vs source-established receipts; no video decode",
            "copied_bytes": total, "video_gib": plan["selected_video_bytes"] / 1024**3,
            "plan_id": plan["plan_id"], "output": str(out)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, help="Original full LAV-DF folder (not required for --verify-only).")
    p.add_argument("--metadata", type=Path, help="Default: metadata.min.json, then metadata.json in source.")
    p.add_argument("--out", type=Path, required=True, help="New independent dataset folder, not a cache folder.")
    p.add_argument("--train", type=int, default=2000)
    p.add_argument("--dev", type=int, default=400)
    p.add_argument("--test", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    actions = p.add_mutually_exclusive_group()
    actions.add_argument("--dry-run", action="store_true", help="Check selection and sizes; write/copy nothing.")
    actions.add_argument("--verify-only", action="store_true", help="Check an existing copy without accessing original videos.")
    args = p.parse_args()
    try:
        out = args.out.expanduser().resolve()
        if args.verify_only:
            print(json.dumps(verify_subset(out), indent=2))
            return
        if args.data_root is None:
            p.error("--data-root is required unless --verify-only is used.")
        root = args.data_root.expanduser().resolve()
        roots_valid(root, out)
        metadata = find_metadata(root, args.metadata)
        plan, selected = plan_copy(root, metadata, {s: getattr(args, s) for s in SPLITS}, args.seed)
        existing_plan_matches(out, plan)
        print(json.dumps(brief(plan, out), indent=2), flush=True)
        if args.dry_run:
            print("DRY RUN: no directory, metadata, or media was created or changed.")
            return
        result = perform_copy(root, out, plan, selected)
        print(json.dumps({k: v for k, v in result.items() if k != "selection"}, indent=2))
        print(f"\nReady: {out}\nPoint --data-root here. Source files were not changed or deleted.")
    except KeyboardInterrupt:
        p.exit(130, "Stopped. Rerun with the same arguments to resume.\n")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        p.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
