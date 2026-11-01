"""Inspect a numerical spectrogram. The rendered PNG is NOT a CNN input."""
import argparse
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("file", type=Path, help="A processed clip .npz file")
    p.add_argument("--window", type=int, default=0)
    p.add_argument("--png", type=Path)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--hop-length", type=int, default=160)
    args = p.parse_args()
    with np.load(args.file, allow_pickle=False) as f:
        values = f["log_mel"][args.window]
    print(f"Shape={values.shape}; dtype={values.dtype}; dB range={values.min():.2f} to {values.max():.2f}")
    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4))
        image = ax.imshow(values, origin="lower", aspect="auto",
            extent=(0, (values.shape[1]-1)*args.hop_length/args.sample_rate, 0, values.shape[0]))
        ax.set(xlabel="Time from window start (seconds)", ylabel="Mel-frequency bin",
               title="Log-mel spectrogram — inspection only")
        fig.colorbar(image, ax=ax, label="dB, fixed reference")
        fig.tight_layout()
        args.png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.png, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
