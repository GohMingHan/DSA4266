# DSA4266 visual models — terminal submission

Run `run.py` from this downloaded folder. No notebook, Kaggle account, GitHub
token or source fetching is required. `settings.json` is the single user settings
file; relative paths there resolve beside that file. CLI path overrides resolve
from the terminal's current directory. `python run.py --help` lists all options.

## Install and check

Python3.12 was tested on Windows. In PowerShell:

```powershell
cd "final_visual submission"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# Install a matching CUDA-enabled torch/torchvision pair first; see link below.
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py --mode check
```

Choose the PyTorch/torchvision installation command for your OS, NVIDIA driver
and supported CUDA version using the [official selector](https://pytorch.org/get-started/locally/).
Requirements preserve a compatible installed pair. Use a clean environment with
exactly `opencv-contrib-python==4.13.0.92` and `mediapipe==0.10.32`; do not install
overlapping headless/plain OpenCV distributions. Linux uses the same Python
commands with its virtual environment's `python` executable.

Training, fresh full DEV inference and the optional real-model GPU smoke require
CUDA. Startup fails before model execution if CUDA is unavailable; CPU training
is never substituted. Code tests, metadata preparation, saved-result display and
saved-prediction evaluation work on CPU. Grad-CAM can use CPU or available CUDA.
No minimum VRAM or local NVIDIA compatibility is claimed without the real smoke
test. With external inputs configured, run:

```powershell
python run.py --mode check --gpu-smoke --models resnet50_baseline
```

This is a bounded real TRAIN/DEV forward/backward check with **zero optimizer
updates**, not training. Full model training and CUDA execution were not run on
the submission machine: its PyTorch build is CPU-only.

## External inputs

Edit `settings.json` once. Keep inputs and outputs **outside this submission**:

- `data_root`: the exact LAV-DF-5000 subset's `metadata.json`, `train/` and `dev/`
  videos, with original native224x224 pixels. The complete LAV-DF release or a
  different subset is not interchangeable. TEST is unavailable from this entry point.
- `index_path`, `train_pool`: existing verified `candidates.sqlite` and fixed
  `train_pool.json`, or leave null to use `artifact_dir/index/` and `artifact_dir/pool/`.
  No database, annotation manifest, videos or images are bundled. To create missing
  artifacts explicitly from the authorized source data:
  `python run.py --mode prepare --build-metadata`.
  This expensive one-time operation retains the original builders, rejects
  existing incompatible artifacts, validates all SQLite schema/rows against the
  original logical hash, and requires the exact original pool SHA256. A different
  SQLite page layout is allowed; changed IDs/timestamps/labels are not.
- `weights_root`: ImageNet `resnet50-11ad3fa6.pth` and
  `efficientnet_b0_rwightman-7f5810bc.pth`, verified by the original hash rules.
- Masked models: `yunet` and `landmarker`, or the pinned files under
  `weights_root/face_assets/`. CLIP: `clip_weights`, a verified pretrained snapshot
  with its original pinned revision/config/processor/vision safetensors manifest.
  Existing complete CLIP trained checkpoints embed their encoder and processor.
  For explicit public asset downloads use
  `python run.py --mode prepare --models all --download-assets` after metadata is available.
- `output_dir`: a new external directory; its independent model runs, best/last
  checkpoints, histories, diagnostics, CSV/HTML/PNG reports and rolling/final
  recovery ZIPs are saved here. An unrelated existing directory is refused.
- Optional `saved_runs`, `resume_from` map model IDs to run folders or partial
  `last.pt` files. `resnet_checkpoint` can supply this project's completed
  full-frame ResNet50 selected `best.pt` to LSTM. Exact identities remain mandatory.

Expected artifact hashes and source-copy/adaptation provenance are recorded in
`configs/source_manifest.json`. Pool seed4266,10,000Real+10,000Fake candidate frames,
4,000+4,000 frames/epoch, full41,746 DEV frames/500videos and original3,551 DEV clips
are retained. LSTM keeps16frames,10FPS,stride12,one-fake-frame labels and1,500+1,500
clips/epoch. Face-mask geometry/fallback, all architectures/optimizers/LRs/losses,
AMP handling and checkpoint schemas are unchanged. CNN/CLIP select DEV frame
ROC-AUC; LSTM selects DEV clip ROC-AUC. CNN/LSTM schedulers monitor that same ROC-AUC;
CLIP has no scheduler. Comparison ranks by video trapezoidal PR-AUC, then video
ROC-AUC and model name, with original Top-5% mean pooling and threshold0.5 metrics.

## Run

Use your environment's `python` (or `.\.venv\Scripts\python.exe` in PowerShell):

```powershell
python run.py --list-models
python run.py --mode train --models resnet50_baseline
python run.py --mode train --models all
python run.py --mode results --models all
python run.py --mode evaluate --models resnet50_baseline
python run.py --mode evaluate --models resnet50_baseline --fresh-evaluation
```

IDs: `resnet50_baseline`, `resnet50_face_mask`, `resnet50_lstm`,
`efficientnet_b0_baseline`, `efficientnet_b0_face_mask`, `clip_vit_b32_shared`.
Models run sequentially. LSTM automatically adds ResNet50 baseline first if no
completed baseline/checkpoint is available. Completed runs are reused; partial
runs require explicit resume; ambiguous runs require `--run MODEL=RUN_DIRECTORY`.
No per-cell confirmation strings are entered. Only training mode (`--mode train`
or an explicitly edited `mode: "train"` setting) starts training.
Default mode is saved results, which needs no videos or pretrained weights.

```powershell
python run.py --mode train --models resnet50_baseline --resume "resnet50_baseline=../recovered/run/last.pt"
python run.py --mode restore --models resnet50_baseline --restore-zip "../backups/model.zip"
python run.py --mode gradcam --models clip_vit_b32_shared --run "clip_vit_b32_shared=../recovered/clip/run"
```

Evaluation reuses saved best DEV predictions/metrics unless `--fresh-evaluation`
explicitly requests one new full DEV pass. Saved histories, 19-column epoch tables
and all original plots remain available; no values are fabricated for Pending,
Partial or missing checkpoints. Reports print in the terminal and are saved as files.
The default explicitly configured streaming cache is bounded to2GiB PNG/64MiB RAM
per worker. `frame_loading_mode="prepared"` instead prepares all61,746 raw RGBs
(12GiB cap/8GiB free reserve). Neither option alters pixels or epoch sampling.
Optional `--mode prepare --prepare-frames` and `--prepare-clips` run without training.
Clips share compatible raw pixels and a bounded missing-pixel overlay. Download
or retain model ZIPs when using temporary cloud storage. A stale `training.lock`
must be checked against its PID before manual removal; it is never stolen.

## Attribution and validation

LAV-DF is by Cai et al.; obtain authorized data separately and follow its
[CC BY-NC4.0 licence](https://github.com/ControlNet/LAV-DF/blob/master/LICENSE)
and upstream VoxCeleb terms. No media/annotations/weights are redistributed here.
Retained YuNet MIT text is `src/YUNET_LICENSE.txt`; dependency/model notices are in
`THIRD_PARTY_NOTICES.txt`. No new licence is asserted over collaborators' code.

The existing numerical, architecture, sampling, preprocessing, metric, resume,
backup and attribution tests are retained. Notebook/private-source bootstrap
assertions are excluded and replaced by terminal-entry tests. The original
156-definition golden fixture is unchanged. Provide `DSA4266_INDEX` and
`DSA4266_POOL` environment variables to include exact external-reference tests;
without those artifacts these tests are explicitly skipped. `--mode check` runs
the CPU-safe suite and states whether external artifacts were actually validated.
