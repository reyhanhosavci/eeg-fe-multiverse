"""
Quick visual check: plot 3 seizure and 3 non-seizure segments from chb01.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from src.chbmit_loader import load_chbmit, FS_CHBMIT, SEGMENT_SEC

# Load only chb01
X, y, pid = load_chbmit(patients=["chb01"], data_dir="data/chbmit/raw", download=False, seed=42)

# Pick 3 seizure + 3 non-seizure
sz_idx = np.where(y == 1)[0][:3]
nsz_idx = np.where(y == 0)[0][:3]

t = np.arange(X.shape[1]) / FS_CHBMIT  # time in seconds

fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True, sharey=True)

for col in range(3):
    # Non-seizure (top row)
    ax = axes[0, col]
    ax.plot(t, X[nsz_idx[col]], color="#1f77b4", linewidth=0.4)
    ax.set_title(f"Non-seizure #{col+1}", fontsize=11)
    if col == 0:
        ax.set_ylabel("Amplitude (uV)")

    # Seizure (bottom row)
    ax = axes[1, col]
    ax.plot(t, X[sz_idx[col]], color="#d62728", linewidth=0.4)
    ax.set_title(f"Seizure #{col+1}", fontsize=11)
    ax.set_xlabel("Time (s)")
    if col == 0:
        ax.set_ylabel("Amplitude (uV)")

fig.suptitle(
    f"CHB-MIT chb01 -- FZ-CZ channel, {SEGMENT_SEC}s segments "
    f"({X.shape[1]} samples @ {FS_CHBMIT} Hz)",
    fontsize=13,
)
fig.tight_layout()

out = Path("results/figures/chbmit_segment_check.png")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out}")
