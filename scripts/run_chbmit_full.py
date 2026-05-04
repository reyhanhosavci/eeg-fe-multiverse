"""
CHB-MIT full pipeline: download + validation in one shot.

Step 1: Download seizure EDF files for all 10 patients (skips existing)
Step 2: Run feature extraction + classification validation
Step 3: Compare sensitivity rankings with Bonn results

Usage:
    python scripts/run_chbmit_full.py
"""

import os
import re
import sys
import time
import urllib.request
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "chbmit" / "raw"
PHYSIONET = "https://physionet.org/files/chbmit/1.0.0"
PATIENTS = [
    "chb01", "chb02", "chb03", "chb05", "chb06",
    "chb07", "chb08", "chb09", "chb10", "chb23",
]


# =========================================================================
# Step 1: Download
# =========================================================================


def download_all():
    """Download summary + seizure EDFs + 2 non-seizure EDFs per patient."""
    print("=" * 60)
    print("  STEP 1: Download CHB-MIT data")
    print("=" * 60)

    total_new = 0

    for patient in PATIENTS:
        pdir = DATA_DIR / patient
        pdir.mkdir(parents=True, exist_ok=True)

        # Summary
        summary_path = pdir / f"{patient}-summary.txt"
        summary_url = f"{PHYSIONET}/{patient}/{patient}-summary.txt"
        if not summary_path.exists():
            print(f"  Fetching {patient} summary...", end=" ", flush=True)
            try:
                urllib.request.urlretrieve(summary_url, str(summary_path))
                print("OK")
            except Exception as e:
                print(f"FAILED: {e}")
                continue

        # Parse summary to find seizure files
        with open(summary_path) as f:
            text = f.read()

        blocks = re.split(r"(?=File Name:)", text)
        seizure_files = []
        all_files = []
        for block in blocks:
            fname_m = re.search(r"File Name:\s*(\S+\.edf)", block, re.IGNORECASE)
            n_m = re.search(r"Number of Seizures in File:\s*(\d+)", block)
            if fname_m:
                fname = fname_m.group(1)
                n_sz = int(n_m.group(1)) if n_m else 0
                all_files.append((fname, n_sz))
                if n_sz > 0:
                    seizure_files.append(fname)

        # Pick 2 non-seizure files for balanced extraction
        non_seizure_files = [f for f, n in all_files if n == 0][:2]
        files_to_get = seizure_files + non_seizure_files

        pending = [f for f in files_to_get
                   if not (pdir / f).exists() or (pdir / f).stat().st_size < 1000]

        if not pending:
            print(f"  {patient}: {len(files_to_get)} files -- all present, skip")
            continue

        print(f"  {patient}: {len(pending)}/{len(files_to_get)} files to download")
        for fname in pending:
            fpath = pdir / fname
            url = f"{PHYSIONET}/{patient}/{fname}"
            print(f"    {fname}...", end=" ", flush=True)
            try:
                urllib.request.urlretrieve(url, str(fpath))
                size_mb = fpath.stat().st_size / 1e6
                print(f"{size_mb:.1f} MB")
                total_new += 1
            except Exception as e:
                print(f"FAILED: {e}")

    print(f"\n  Download complete: {total_new} new files\n")


# =========================================================================
# Step 2 + 3: Validation (delegates to existing script)
# =========================================================================


def run_validation():
    """Import and run the validation pipeline."""
    print("=" * 60)
    print("  STEP 2-3: Feature extraction + classification + comparison")
    print("=" * 60)

    # Import the main function from the validation script
    from scripts.run_chbmit_validation import main as validation_main
    validation_main()


# =========================================================================
# Main
# =========================================================================


if __name__ == "__main__":
    t0 = time.perf_counter()

    download_all()
    run_validation()

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 60}")
    print(f"  FULL PIPELINE DONE -- {elapsed / 60:.1f} min total")
    print(f"{'=' * 60}")
