"""Read the actual CLIP videos.jsonl; join to official metadata without resampling IDs."""
from __future__ import annotations
import json
import math
from pathlib import Path, PurePosixPath
import random
from common import json_read, relative_file, strict_bool, SPLITS


def merge_periods(value):
    if not isinstance(value, list):
        raise ValueError('fake_periods must be a list.')
    pairs = []
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f'Invalid interval: {pair!r}')
        a, b = map(float, pair)
        if not math.isfinite(a + b) or a < 0 or b <= a:
            raise ValueError(f'Invalid interval: {pair!r}')
        pairs.append([a, b])
    merged = []
    for a, b in sorted(pairs):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def normalize(row):
    file = relative_file(row['file'])
    split = row['split']
    if split not in SPLITS or PurePosixPath(file).parts[0] != split:
        raise ValueError(f'Inconsistent split/path: {file!r}, {split!r}')
    ma, mv = strict_bool(row['modify_audio']), strict_bool(row['modify_video'])
    periods = merge_periods(row['fake_periods'])
    if bool(ma or mv) != bool(periods):
        raise ValueError(f'Modality flags and intervals disagree: {file}')
    duration = float(row.get('duration', row.get('metadata_duration_s', 0)))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f'Bad metadata duration: {file}')
    group = str(row.get('original') or row.get('source_group') or file).replace('\\', '/')
    return dict(file=file, split=split, modify_audio=ma, modify_video=mv,
                source_group=group, metadata_duration_s=duration,
                fake_periods=periods, raw_fake_periods=row['fake_periods'],
                visual_periods=periods if mv else [])


def select_cohort(metadata: Path, videos_jsonl: Path, splits, limit_per_split=0, seed=42):
    """All requested JSONL video IDs must exist; no full-dataset fallback."""
    requested = {}
    with Path(videos_jsonl).open(encoding='utf-8-sig') as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict) or 'file' not in item:
                raise ValueError(f'Expected a video object with file at JSONL line {line_number}.')
            key = relative_file(item['file'])
            if key in requested:
                raise ValueError(f'Duplicate video in reference cohort: {key}')
            requested[key] = item
    if not requested:
        raise ValueError('Reference videos.jsonl is empty.')
    raw = json_read(metadata)
    if not isinstance(raw, list):
        raise ValueError('Expected official LAV-DF metadata as a JSON list.')
    found = {}
    # Normalize only referenced rows; unrelated full-dataset metadata is not the cohort.
    for item in raw:
        if not isinstance(item, dict) or 'file' not in item:
            continue
        key = relative_file(item['file'])
        if key not in requested:
            continue
        if key in found:
            raise ValueError(f'Duplicate referenced ID in source metadata: {key}')
        normalized = normalize(item)
        reference = requested[key]
        for field in ('split', 'modify_audio', 'modify_video', 'source_group'):
            if field in reference and reference[field] != normalized[field]:
                raise ValueError(f'CLIP cohort versus metadata disagreement: {key}, {field}')
        if 'fake_periods' in reference and merge_periods(reference['fake_periods']) != normalized['fake_periods']:
            raise ValueError(f'Annotation disagreement with CLIP cohort: {key}')
        if 'decoded_video_end_s' in reference:
            normalized['reference_decoded_end_s'] = float(reference['decoded_video_end_s'])
        found[key] = normalized
    missing = sorted(set(requested) - set(found))
    if missing:
        raise ValueError(f'{len(missing)} reference IDs absent from metadata: {missing[:5]}')
    groups = {}
    for row in found.values():
        if row['source_group'] in groups and groups[row['source_group']] != row['split']:
            raise ValueError(f'Literal source group crosses splits: {row["source_group"]}')
        groups[row['source_group']] = row['split']
    selected = []
    for split in dict.fromkeys(splits):
        pool = sorted((r for r in found.values() if r['split'] == split), key=lambda r: r['file'])
        if not pool:
            raise ValueError(f'No {split} videos in reference cohort. Do not invent a new split.')
        if limit_per_split:
            if limit_per_split < 2 or limit_per_split % 2:
                raise ValueError('Pilot --limit-per-split must be even and >= 2.')
            rng = random.Random(seed)
            chosen = []
            for label in (False, True):
                part = [r for r in pool if r['modify_video'] == label]
                rng.shuffle(part)
                if len(part) < limit_per_split // 2:
                    raise ValueError(f'Not enough {split} videos of visual class {label} for this pilot.')
                chosen.extend(part[:limit_per_split // 2])
            pool = sorted(chosen, key=lambda r: r['file'])
        selected.extend(pool)
    return selected
