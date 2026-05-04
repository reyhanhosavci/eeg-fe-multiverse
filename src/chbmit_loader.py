"""
CHB-MIT Scalp EEG dataset loader for cross-dataset validation.

Downloads EDF files from PhysioNet for selected patients (chb01–chb05),
parses seizure annotations from summary files, extracts fixed-length
single-channel segments (seizure + non-seizure), and applies basic
preprocessing (band-pass filter).

Target channel: FZ-CZ (Channel 17 in all chb01–chb05 files).
Segment length: 23.6 seconds (matching Bonn dataset).
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import mne
import numpy as np
import urllib.request

PHYSIONET_BASE = "https://physionet.org/files/chbmit/1.0.0"
TARGET_CHANNEL = "FZ-CZ"
SEGMENT_SEC = 23.6       # Match Bonn segment duration
FS_CHBMIT = 256          # CHB-MIT sampling rate
SAMPLES_PER_SEG = int(SEGMENT_SEC * FS_CHBMIT)  # 6042 samples
MIN_SEIZURE_GAP_SEC = 300  # Non-seizure segments must be ≥5 min from any seizure
DEFAULT_PATIENTS = [
    "chb01", "chb02", "chb03", "chb05", "chb06",
    "chb07", "chb08", "chb09", "chb10", "chb23",
]


# ---------------------------------------------------------------------------
# Summary parser
# ---------------------------------------------------------------------------


def parse_summary(summary_text: str) -> list[dict]:
    """Parse a CHB-MIT summary file into a list of file records.

    Args:
        summary_text: Raw text content of chbXX-summary.txt.

    Returns:
        List of dicts, each with keys: filename, n_seizures,
        seizures (list of (start_sec, end_sec) tuples).
    """
    records = []
    # Split by "File Name:" to get per-file blocks
    blocks = re.split(r"(?=File Name:)", summary_text)

    for block in blocks:
        fname_match = re.search(r"File Name:\s*(\S+\.edf)", block, re.IGNORECASE)
        if not fname_match:
            continue

        filename = fname_match.group(1)

        n_match = re.search(r"Number of Seizures in File:\s*(\d+)", block)
        n_seizures = int(n_match.group(1)) if n_match else 0

        seizures = []
        # Handle both "Seizure Start Time" and "Seizure N Start Time"
        starts = re.findall(
            r"Seizure\s*(?:\d+\s*)?Start Time:\s*(\d+)\s*seconds", block
        )
        ends = re.findall(
            r"Seizure\s*(?:\d+\s*)?End Time:\s*(\d+)\s*seconds", block
        )
        for s, e in zip(starts, ends):
            seizures.append((int(s), int(e)))

        records.append({
            "filename": filename,
            "n_seizures": n_seizures,
            "seizures": seizures,
        })

    return records


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download_file(url: str, dest: Path) -> None:
    """Download a file from URL to dest, skipping if already exists."""
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"    Downloading {dest.name}...", end="", flush=True)
    urllib.request.urlretrieve(url, str(dest))
    size_mb = dest.stat().st_size / 1e6
    print(f" {size_mb:.1f} MB")


def download_patient(
    patient: str, data_dir: Path, records: list[dict] | None = None
) -> tuple[list[dict], Path]:
    """Download summary + seizure EDF files for one patient.

    Only downloads files that contain seizures, plus a few non-seizure
    files for balanced extraction.

    Args:
        patient: Patient ID (e.g. 'chb01').
        data_dir: Root data directory.
        records: Pre-parsed summary records (if None, fetches and parses).

    Returns:
        (records, patient_dir) tuple.
    """
    patient_dir = data_dir / patient

    # Summary file
    summary_path = patient_dir / f"{patient}-summary.txt"
    summary_url = f"{PHYSIONET_BASE}/{patient}/{patient}-summary.txt"
    _download_file(summary_url, summary_path)

    if records is None:
        with open(summary_path) as f:
            records = parse_summary(f.read())

    # Identify which files to download
    seizure_files = {r["filename"] for r in records if r["n_seizures"] > 0}

    # Also download some non-seizure files (first 2 non-seizure files
    # per seizure file, to ensure enough non-seizure data)
    non_seizure_files = set()
    for r in records:
        if r["n_seizures"] == 0 and len(non_seizure_files) < len(seizure_files) * 2:
            non_seizure_files.add(r["filename"])

    files_to_download = seizure_files | non_seizure_files

    for fname in sorted(files_to_download):
        url = f"{PHYSIONET_BASE}/{patient}/{fname}"
        _download_file(url, patient_dir / fname)

    return records, patient_dir


# ---------------------------------------------------------------------------
# Segment extraction
# ---------------------------------------------------------------------------


def extract_segments(
    patient: str,
    data_dir: Path,
    channel: str = TARGET_CHANNEL,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract seizure and non-seizure segments for one patient.

    Seizure segments: extracted from annotated seizure periods, split into
    non-overlapping windows of SEGMENT_SEC duration.

    Non-seizure segments: randomly sampled from periods ≥5 min away from
    any seizure, downsampled to match seizure count.

    Args:
        patient: Patient ID.
        data_dir: Root data directory containing patient subdirectories.
        channel: EEG channel name to extract.
        seed: Random seed for non-seizure sampling.

    Returns:
        (X, y) where X has shape (n_segments, SAMPLES_PER_SEG) and
        y is 0 (non-seizure) or 1 (seizure).
    """
    rng = np.random.RandomState(seed)
    patient_dir = data_dir / patient

    # Parse summary
    summary_path = patient_dir / f"{patient}-summary.txt"
    with open(summary_path) as f:
        records = parse_summary(f.read())

    seizure_segments = []
    non_seizure_pool = []

    for rec in records:
        edf_path = patient_dir / rec["filename"]
        if not edf_path.exists():
            continue

        # Load EDF with MNE (suppress verbose output)
        try:
            raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        except Exception as e:
            warnings.warn(f"Failed to read {edf_path}: {e}")
            continue

        # Find target channel (handle naming variations)
        ch_names = raw.ch_names
        ch_match = None
        for ch in ch_names:
            if channel.replace("-", "").lower() in ch.replace("-", "").lower():
                ch_match = ch
                break

        if ch_match is None:
            warnings.warn(f"Channel {channel} not found in {edf_path.name}, skipping")
            continue

        # Extract single channel data
        raw.pick([ch_match])

        # Band-pass filter: 0.5–45 Hz
        raw.filter(0.5, 45.0, verbose=False)

        data = raw.get_data()[0]  # (n_samples,)
        fs = raw.info["sfreq"]
        seg_samples = int(SEGMENT_SEC * fs)

        # Collect all seizure intervals for this file
        file_seizure_ranges = []

        # Extract seizure segments
        if rec["n_seizures"] > 0:
            for s_start, s_end in rec["seizures"]:
                s_start_samp = int(s_start * fs)
                s_end_samp = int(s_end * fs)
                file_seizure_ranges.append((s_start_samp, s_end_samp))

                # Split seizure period into non-overlapping segments
                duration_samp = s_end_samp - s_start_samp
                n_segs = duration_samp // seg_samples

                for i in range(max(n_segs, 1)):
                    start = s_start_samp + i * seg_samples
                    end = start + seg_samples
                    if end <= len(data):
                        seizure_segments.append(data[start:end])

        # Collect non-seizure candidate windows
        # Must be ≥ MIN_SEIZURE_GAP_SEC from any seizure in this file
        gap_samples = int(MIN_SEIZURE_GAP_SEC * fs)

        for win_start in range(0, len(data) - seg_samples, seg_samples):
            win_end = win_start + seg_samples
            too_close = False
            for sz_start, sz_end in file_seizure_ranges:
                if win_end > (sz_start - gap_samples) and win_start < (sz_end + gap_samples):
                    too_close = True
                    break
            # Also skip if the file has seizures and we're in a non-seizure file
            # (non-seizure files have no seizure ranges, so all windows pass)
            if not too_close:
                non_seizure_pool.append(data[win_start:win_end])

    if not seizure_segments:
        warnings.warn(f"No seizure segments found for {patient}")
        return np.array([]), np.array([])

    # Downsample non-seizure to match seizure count
    n_seizure = len(seizure_segments)
    if len(non_seizure_pool) > n_seizure:
        indices = rng.choice(len(non_seizure_pool), size=n_seizure, replace=False)
        non_seizure_segments = [non_seizure_pool[i] for i in indices]
    else:
        non_seizure_segments = non_seizure_pool

    # Ensure consistent segment length
    seizure_segments = [s[:SAMPLES_PER_SEG] for s in seizure_segments if len(s) >= SAMPLES_PER_SEG]
    non_seizure_segments = [s[:SAMPLES_PER_SEG] for s in non_seizure_segments if len(s) >= SAMPLES_PER_SEG]

    n_sz = len(seizure_segments)
    n_nsz = len(non_seizure_segments)

    X = np.vstack(non_seizure_segments + seizure_segments)
    y = np.array([0] * n_nsz + [1] * n_sz)

    return X, y


# ---------------------------------------------------------------------------
# Multi-patient loader
# ---------------------------------------------------------------------------


def load_chbmit(
    patients: list[str] | None = None,
    data_dir: str | Path = "data/chbmit/raw",
    channel: str = TARGET_CHANNEL,
    download: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and segment CHB-MIT data for multiple patients.

    Args:
        patients: List of patient IDs. Defaults to chb01–chb05.
        data_dir: Root directory for downloaded data.
        channel: Target EEG channel.
        download: If True, download missing files from PhysioNet.
        seed: Random seed.

    Returns:
        (X, y, patient_ids) where:
            X: shape (n_total_segments, SAMPLES_PER_SEG)
            y: 0=non-seizure, 1=seizure
            patient_ids: integer patient index per segment (for LOPO CV)
    """
    if patients is None:
        patients = DEFAULT_PATIENTS

    data_dir = Path(data_dir)
    all_X, all_y, all_pid = [], [], []

    for i, patient in enumerate(patients):
        print(f"\n  [{i+1}/{len(patients)}] {patient}")

        if download:
            print(f"  Downloading seizure files...")
            download_patient(patient, data_dir)

        print(f"  Extracting segments (channel={channel})...")
        X, y = extract_segments(patient, data_dir, channel=channel, seed=seed)

        if len(X) == 0:
            print(f"  WARNING: No segments for {patient}, skipping")
            continue

        n_sz = int(y.sum())
        n_nsz = len(y) - n_sz
        print(f"  Segments: {n_nsz} non-seizure + {n_sz} seizure = {len(y)} total")

        all_X.append(X)
        all_y.append(y)
        all_pid.append(np.full(len(y), i, dtype=int))

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    pid_all = np.concatenate(all_pid)

    print(f"\n  Total: {len(y_all)} segments "
          f"({int((y_all == 0).sum())} non-seizure, {int(y_all.sum())} seizure)")

    return X_all, y_all, pid_all
