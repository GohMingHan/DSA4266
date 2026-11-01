"""Experiment 2A: LAV-DF -> matched raw waveform and log-mel windows.

This is an independent research baseline, not an official LAV-DF reproduction.
Audio labels use modify_audio AND interval overlap; visual-only edits are negatives.
All timing in manifests is relative to the first decoded video frame.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import shutil
import subprocess
import tempfile
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

PIPELINE_VERSION = 2


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    window_seconds: float = 4.0375
    stride_seconds: float = 0.5
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 128
    fmin: float = 0.0
    fmax: float = 8000.0
    floor_db: float = -100.0
    storage_dtype: str = "float32"
    padding: str = "repeat"

    def validate(self) -> None:
        if self.padding != "repeat":
            raise ValueError("Experiment 2A uses repeat padding for both representations.")
        if self.sample_rate < 1000 or not 0 < self.stride_seconds <= self.window_seconds:
            raise ValueError("Require positive stride <= window length and sample_rate >= 1000.")
        if not 0 < self.win_length <= self.n_fft or self.hop_length <= 0 or self.n_mels < 16:
            raise ValueError("Invalid FFT, window, hop or mel settings.")
        if not 0 <= self.fmin < self.fmax <= self.sample_rate / 2:
            raise ValueError("Require 0 <= fmin < fmax <= sample_rate / 2.")
        if self.window_samples < self.n_fft:
            raise ValueError("The audio window must be at least n_fft samples long.")
        if self.floor_db >= 0 or self.storage_dtype not in ("float16", "float32"):
            raise ValueError("Invalid floor_db or storage_dtype.")
        if self.stride_samples < 1:
            raise ValueError("Stride is shorter than one sample.")

    @property
    def window_samples(self) -> int:
        return round(self.window_seconds * self.sample_rate)

    @property
    def stride_samples(self) -> int:
        return round(self.stride_seconds * self.sample_rate)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, allow_nan=False) + "\n")
    tmp.replace(path)


def strict_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{name} must be a JSON boolean or 0/1, not {value!r}.")


def merge_periods(periods: list, duration: float) -> list[list[float]]:
    """Validate and merge fake intervals without clipping to metadata duration.

    Some LAV-DF records contain fake-period endpoints beyond the metadata
    ``duration`` value.  The metadata duration is therefore retained for
    auditing, but it is not used as a hard boundary here.  Actual media
    availability is checked later against the decoded audio timeline.
    """
    result: list[list[float]] = []
    for interval in sorted(periods):
        if len(interval) != 2:
            raise ValueError(f"Invalid fake interval: {interval}")
        a, b = map(float, interval)
        if not math.isfinite(a) or not math.isfinite(b) or a < 0 or b <= a:
            raise ValueError(
                f"Invalid fake interval {interval} (metadata duration={duration})."
            )
        if result and a <= result[-1][1]:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


def normalize_metadata(raw: dict) -> dict:
    rel = str(raw["file"]).replace("\\", "/")
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts or ":" in rel:
        raise ValueError("Metadata file paths must be relative to --data-root.")
    split = str(raw["split"])
    if split not in {"train", "dev", "test"}:
        raise ValueError(f"Unknown split {split!r}; preserve LAV-DF train/dev/test.")
    duration = float(raw["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Duration must be positive and finite.")
    ma = strict_bool(raw["modify_audio"], "modify_audio")
    mv = strict_bool(raw["modify_video"], "modify_video")
    periods = merge_periods(raw["fake_periods"], duration)
    if (ma or mv) and not periods:
        raise ValueError("A manipulated item must have temporal annotations.")
    if not (ma or mv) and periods:
        raise ValueError("An unmodified item cannot have nonempty fake_periods.")
    original = raw.get("original")
    return {"file": p.as_posix(), "split": split, "duration": duration,
            "original": str(original).replace("\\", "/") if original else None,
            "modify_audio": ma, "modify_video": mv, "fake_periods": periods}


def load_metadata(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Expected official LAV-DF metadata: a JSON list of records.")
    rows = [normalize_metadata(x) for x in raw]
    names = [r["file"] for r in rows]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate file entries in metadata.")
    # Literal source-group check. This does not infer subject identity from names.
    groups: dict[str, str] = {}
    for r in rows:
        group = r["original"] or r["file"]
        if group in groups and groups[group] != r["split"]:
            raise ValueError(f"Original/source group crosses splits: {group}")
        groups[group] = r["split"]
    return rows


def run_checked(command: list[str], timeout: float = 180) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                               encoding="utf-8", errors="replace", check=False)
    if completed.returncode:
        raise RuntimeError(f"{command[0]} failed: {completed.stderr[-2500:]}")
    return completed.stdout


def probe_timing(path: Path, ffprobe: str) -> dict:
    """Check decoded audio continuity and relate its first sample to video time zero.

    We do not silently insert/drop samples for timestamp gaps. Such clips are flagged.
    """
    base = [ffprobe, "-v", "error"]
    streams = json.loads(run_checked(base + ["-show_streams", "-of", "json", str(path)]))["streams"]
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if audio is None or video is None:
        raise ValueError("Expected both an audio and a video stream.")
    native_sr = int(audio["sample_rate"])
    frames = json.loads(run_checked(base + ["-select_streams", "a:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time,pts_time,nb_samples",
        "-of", "json", str(path)]))["frames"]
    if not frames:
        raise ValueError("No decoded audio frames.")
    first_audio = None
    expected = None
    total_native_samples = 0
    tolerance = max(2 / native_sr, 0.001)
    for frame in frames:
        stamp = frame.get("best_effort_timestamp_time", frame.get("pts_time"))
        n = int(frame.get("nb_samples", 0))
        if stamp is None or n <= 0:
            raise ValueError("Missing audio frame timestamps/sample counts.")
        t = float(stamp)
        if expected is not None and abs(t - expected) > tolerance:
            raise ValueError(f"Discontinuous audio timeline at {t:.6f}s (expected {expected:.6f}s).")
        if first_audio is None:
            first_audio = t
        expected = t + n / native_sr
        total_native_samples += n
    # Read enough initial packets to allow B-frame decoding, then select first PTS.
    vframes = json.loads(run_checked(base + ["-select_streams", "v:0", "-read_intervals", "%+#64",
        "-show_frames", "-show_entries", "frame=best_effort_timestamp_time,pts_time",
        "-of", "json", str(path)]))["frames"]
    if not vframes:
        raise ValueError("Cannot determine first decoded video timestamp.")
    stamps = [f.get("best_effort_timestamp_time", f.get("pts_time")) for f in vframes]
    if any(t is None for t in stamps):
        raise ValueError("Missing initial video timestamps.")
    first_video = min(map(float, stamps))
    return {"first_audio_pts_s": first_audio, "first_video_pts_s": first_video,
            "audio_offset_s": first_audio - first_video,
            "native_sample_rate": native_sr,
            "decoded_audio_duration_s": total_native_samples / native_sr,
            "native_audio_channels": int(audio.get("channels", 0))}


def extract_wav(video: Path, wav: Path, sample_rate: int, ffmpeg: str = "ffmpeg") -> np.ndarray:
    wav.parent.mkdir(parents=True, exist_ok=True)
    run_checked([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video), "-map", "0:a:0", "-vn", "-sn", "-dn",
        "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(wav)])
    y, sr = sf.read(wav, dtype="float32", always_2d=False)
    if sr != sample_rate or y.ndim != 1 or y.size == 0 or not np.isfinite(y).all():
        raise ValueError("Extracted waveform is empty, invalid, non-mono, or at the wrong sample rate.")
    return y


def window_starts(n_samples: int, length: int, stride: int) -> list[int]:
    """Fixed stride, plus one tail-aligned window; no label-dependent sampling."""
    if n_samples <= 0 or not 0 < stride <= length:
        raise ValueError("Invalid length/stride or empty waveform.")
    if n_samples <= length:
        return [0]
    starts = list(range(0, n_samples - length + 1, stride))
    if starts[-1] != n_samples - length:
        starts.append(n_samples - length)
    return starts


def overlap_seconds(start: float, end: float, periods: list[list[float]]) -> float:
    # periods must have been merged by merge_periods.
    return sum(max(0.0, min(end, b) - max(start, a)) for a, b in periods)


def log_mel(window: np.ndarray, config: AudioConfig) -> np.ndarray:
    """Reference=1, no per-window max normalization or dynamic top-dB clipping."""
    if window.ndim != 1 or window.size != config.window_samples:
        raise ValueError("log_mel expects one already padded, fixed-length waveform.")
    power = librosa.feature.melspectrogram(
        y=window.astype(np.float32, copy=False), sr=config.sample_rate,
        n_fft=config.n_fft, win_length=config.win_length, hop_length=config.hop_length,
        n_mels=config.n_mels, fmin=config.fmin, fmax=config.fmax,
        window="hann", center=True, pad_mode="constant", power=2.0,
        htk=False, norm="slaney")
    db = librosa.power_to_db(power, ref=1.0, amin=10 ** (config.floor_db / 10), top_db=None)
    if not np.isfinite(db).all():
        raise ValueError("Nonfinite spectrogram.")
    return db.astype(np.float32)


def repeat_pad(segment: np.ndarray, length: int) -> np.ndarray:
    """Match the official AASIST short-clip repetition, not time stretching."""
    if segment.ndim != 1 or len(segment) < 1 or length < 1:
        raise ValueError("Expected a nonempty mono segment and positive length.")
    return np.tile(segment, math.ceil(length / len(segment)))[:length].astype(np.float32)


def make_windows(y: np.ndarray, meta: dict, timing: dict, config: AudioConfig) -> tuple[np.ndarray, list[dict]]:
    sr, n = config.sample_rate, config.window_samples
    offset = float(timing["audio_offset_s"])
    periods = meta["fake_periods"] if meta["modify_audio"] else []
    # Do not truncate at metadata ``duration``.  Some official LAV-DF records
    # contain manipulation intervals beyond that value.  Use the decoded audio
    # extent as the availability boundary and retain metadata duration only as
    # an audit field.
    first = max(0, math.ceil((-offset) * sr - 1e-7))
    last = len(y)
    if last <= first:
        raise ValueError("Audio and annotated video have no temporal overlap.")

    available_a, available_b = offset + first / sr, offset + last / sr
    annotated_end = max((b for _, b in meta["fake_periods"]), default=0.0)
    if annotated_end > meta["duration"] + 0.1:
        print(
            f"WARNING: {meta['file']}: fake annotation ends at "
            f"{annotated_end:.3f}s, metadata duration is {meta['duration']:.3f}s, "
            f"decoded audio ends at {available_b:.3f}s."
        )

    if meta["modify_audio"]:
        if any(a < available_a - 0.02 or b > available_b + 0.02 for a, b in periods):
            raise ValueError(
                "Annotated audio manipulation lies outside available decoded audio "
                f"(available {available_a:.3f}-{available_b:.3f}s; periods={periods})."
            )
    specs, rows = [], []
    for local_start in window_starts(last - first, n, config.stride_samples):
        start = first + local_start
        end = min(start + n, last)
        segment = y[start:end]
        # Both branches receive this EXACT repeated waveform for short clips.
        padded = repeat_pad(segment, n)
        spec = log_mel(padded, config)
        ts, te = offset + start / sr, offset + end / sr
        overlap = overlap_seconds(ts, te, periods)
        rows.append({"window_index": len(rows), "start_sample": start, "end_sample": end,
                     "start_s": ts, "end_s": te, "center_s": (ts + te) / 2,
                     "valid_samples": end - start, "padding_fraction": 1 - (end-start) / n,
                     "audio_fake_seconds": overlap, "audio_label": int(overlap > 1e-9),
                     "input_samples": n, "padding_mode": config.padding})
        specs.append(spec)
    return np.stack(specs).astype(config.storage_dtype), rows


def process_clip(meta: dict, data_root: Path, out: Path, config: AudioConfig,
                 config_id: str, ffmpeg: str, ffprobe: str, keep_wav: bool) -> tuple[dict, bool]:
    data_root = data_root.resolve()
    source = (data_root / meta["file"]).resolve()
    if not source.is_relative_to(data_root):
        raise ValueError("Input path escapes the dataset root.")
    stat = source.stat()
    clip_id = hashlib.sha256(meta["file"].encode()).hexdigest()[:20]
    folder = out / "clips" / clip_id[:2]
    sidecar, target = folder / f"{clip_id}.json", folder / f"{clip_id}.npz"
    signature = digest({"source_path": str(source), "metadata": meta, "size": stat.st_size,
                    "config_id": config_id, "keep_wav": keep_wav})
    if sidecar.exists() and target.exists():
        previous = json.loads(sidecar.read_text(encoding="utf-8"))
        if previous.get("source_signature") == signature:
            if not keep_wav or (out / previous["wav_path"]).is_file():
                return previous, True
    # Remove stale record first, so failed replacement cannot leave it in the manifest.
    sidecar.unlink(missing_ok=True)
    timing = probe_timing(source, ffprobe)
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lavdf_audio_") as tmp:
        wav = Path(tmp) / "audio.wav"
        y = extract_wav(source, wav, config.sample_rate, ffmpeg)
        if abs(len(y) / config.sample_rate - timing["decoded_audio_duration_s"]) > 0.03:
            raise ValueError("Decoded sample count differs from probed audio timing by >30ms.")
        specs, windows = make_windows(y, meta, timing, config)
        tmp_npz = target.with_name(target.name + ".tmp")
        with tmp_npz.open("wb") as f:
            np.savez_compressed(f, log_mel=specs)
        tmp_npz.replace(target)
        wav_rel = None
        if keep_wav:
            wav_dest = folder / f"{clip_id}.wav"
            shutil.copyfile(wav, wav_dest)
            wav_rel = wav_dest.relative_to(out).as_posix()
    row = {**meta, **timing, "clip_id": clip_id, "source_group": meta["original"] or meta["file"],
           "source_signature": signature, "config_id": config_id,
           "feature_path": target.relative_to(out).as_posix(), "wav_path": wav_rel,
           "shape": list(specs.shape), "windows": windows,
           "audio_video_label": int(meta["modify_audio"]),
           "overall_video_label": int(meta["modify_audio"] or meta["modify_video"]),
           "feature_bytes": target.stat().st_size,
           "wav_bytes": (out / wav_rel).stat().st_size if wav_rel else 0}
    write_json(sidecar, row)
    return row, False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--metadata", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--splits", nargs="+", choices=["train", "dev", "test"], default=["train", "dev"])
    p.add_argument("--limit-per-split", type=int, default=0, help="0 means all; shuffled deterministically within each split.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window-seconds", type=float, default=4.0375)
    p.add_argument("--stride-seconds", type=float, default=0.5)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--n-fft", type=int, default=512)
    p.add_argument("--win-length", type=int, default=400)
    p.add_argument("--hop-length", type=int, default=160)
    p.add_argument("--fmax", type=float)
    p.add_argument("--storage-dtype", choices=["float16", "float32"], default="float32")
    p.add_argument("--keep-wav", action="store_true", default=True,
                   help="Always enabled in Experiment 2A; the AASIST branch needs the waveform.")
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--ffprobe", default="ffprobe")
    args = p.parse_args()
    if args.limit_per_split < 0:
        p.error("--limit-per-split must be >= 0")
    for executable in [args.ffmpeg, args.ffprobe]:
        if shutil.which(executable) is None:
            p.error(f"Cannot find {executable}. Install FFmpeg/ffprobe or provide their full executable paths.")
    root, out = args.data_root.resolve(), args.out.resolve()
    metadata = args.metadata or root / "metadata.min.json"
    if args.metadata is None and not metadata.exists():
        metadata = root / "metadata.json"
    rows = load_metadata(metadata)
    config = AudioConfig(sample_rate=args.sample_rate, window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds, n_mels=args.n_mels, n_fft=args.n_fft,
        win_length=args.win_length, hop_length=args.hop_length,
        fmax=args.fmax if args.fmax is not None else args.sample_rate / 2,
        storage_dtype=args.storage_dtype)
    config.validate()
    if config.sample_rate != 16000 or config.window_samples != 64600:
        p.error("Matched Experiment 2A requires 16,000 Hz and 64,600 samples (4.0375 seconds).")
    provenance = {"pipeline_version": PIPELINE_VERSION, "audio": asdict(config),
                  "librosa_version": librosa.__version__, "numpy_version": np.__version__,
                  "ffmpeg_version": run_checked([args.ffmpeg, "-version"]).splitlines()[0],
                  "label_policy": "modify_audio AND any strictly positive interval overlap"}
    config_id = digest(provenance)
    cfg_path = out / "config.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text(encoding="utf-8"))["config_id"] != config_id:
        p.error("Output directory uses a different preprocessing configuration. Choose a new --out.")
    write_json(cfg_path, {**provenance, "config_id": config_id})
    selection = []
    for split in args.splits:
        pool = sorted([r for r in rows if r["split"] == split], key=lambda r: r["file"])
        random.Random(args.seed).shuffle(pool)
        selection.extend(pool[:args.limit_per_split] if args.limit_per_split else pool)
    failures, resumed = [], 0
    for meta in tqdm(selection, desc="Audio -> matched waveform/log-mel", unit="video"):
        try:
            _, skipped = process_clip(meta, root, out, config, config_id,
                                      args.ffmpeg, args.ffprobe, args.keep_wav)
            resumed += int(skipped)
        except Exception as exc:
            failed_id = hashlib.sha256(meta["file"].encode()).hexdigest()[:20]
            (out / "clips" / failed_id[:2] / f"{failed_id}.json").unlink(missing_ok=True)
            failures.append({"file": meta["file"], "split": meta["split"],
                             "error_type": type(exc).__name__, "error": str(exc)})
            tqdm.write(f"FLAGGED {meta['file']}: {exc}")
    current_meta = {r["file"]: r for r in rows}
    records = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((out / "clips").glob("*/*.json"))]
    records = [r for r in records if r["config_id"] == config_id and r["file"] in current_meta
               and all(r[k] == v for k, v in current_meta[r["file"]].items())]
    records.sort(key=lambda r: (r["split"], r["file"]))
    write_jsonl(out / "manifest.jsonl", records)
    write_jsonl(out / "errors_latest_run.jsonl", failures)
    counts = Counter()
    for r in records:
        counts[f"{r['split']}_videos"] += 1
        counts[f"{r['split']}_audio_fake_videos"] += r["audio_video_label"]
        for w in r["windows"]:
            counts[f"{r['split']}_windows_label_{w['audio_label']}"] += 1
    summary = {"counts": dict(counts), "selected_this_run": len(selection),
               "resumed_this_run": resumed, "failures_this_run": len(failures),
               "feature_gib": sum(r["feature_bytes"] for r in records) / 1024**3,
               "wav_gib": sum(r.get("wav_bytes", 0) for r in records) / 1024**3,
               "metadata_videos": len(rows), "config_id": config_id}
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"Manifest: {out / 'manifest.jsonl'}")
    if failures:
        print("Inspect errors_latest_run.jsonl and report coverage; failed clips are NOT relabelled as genuine.")
    if not records:
        raise SystemExit("No successful clips were processed.")


if __name__ == "__main__":
    main()
