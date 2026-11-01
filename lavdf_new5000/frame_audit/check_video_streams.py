import hashlib
import json
import subprocess
from pathlib import Path
from tqdm import tqdm

ROOT = Path("/mnt/c/Users/wuche/Downloads/LAV-DF-5000-NEW")

metadata = json.loads(
    (ROOT / "metadata.new5000.json").read_text()
)

pairs = [
    (r["original"], r["file"])
    for r in metadata
    if r["split"] == "dev"
    and r["modify_audio"] is True
    and r["modify_video"] is False
]

assert len(pairs) == 125


def video_hash(path):
    command = [
        "ffmpeg",
        "-v", "error",
        "-i", str(path),
        "-map", "0:v:0",
        "-c:v", "copy",
        "-f", "h264",
        "-"
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    digest = hashlib.sha256()

    while True:
        chunk = process.stdout.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)

    stderr = process.stderr.read()
    status = process.wait()

    if status != 0:
        raise RuntimeError(stderr.decode(errors="replace"))

    return digest.hexdigest()


same = 0

for original, fake in tqdm(pairs):
    h0 = video_hash(ROOT / original)
    h1 = video_hash(ROOT / fake)

    if h0 == h1:
        same += 1

print("Total pairs:", len(pairs))
print("Identical encoded video streams:", same)
print("Different encoded video streams:", len(pairs) - same)
