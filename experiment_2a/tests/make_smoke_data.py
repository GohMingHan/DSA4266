"""Make non-speech synthetic media ONLY for software smoke testing, not research."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for split in ("train", "dev", "test"):
        for index, (ma, mv) in enumerate(((False, False), (True, False), (False, True), (True, True))):
            rel = f"videos/{split}_{index}.mp4"
            target = args.out / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y",
                "-f", "lavfi", "-i", "color=size=64x64:rate=25:duration=3",
                "-f", "lavfi", "-i", f"sine=frequency={440 + 70*index}:sample_rate=16000:duration=3",
                "-c:v", "mpeg4", "-c:a", "aac", str(target)], check=True)
            rows.append({"file": rel, "split": split, "duration": 3.0, "original": None,
                         "modify_audio": ma, "modify_video": mv,
                         "fake_periods": [[1.1, 1.4]] if (ma or mv) else []})
    (args.out / "metadata.min.json").write_text(json.dumps(rows), encoding="utf-8")
    print("Synthetic tones with artificial labels only. Not a deepfake benchmark.")


if __name__ == "__main__":
    main()
