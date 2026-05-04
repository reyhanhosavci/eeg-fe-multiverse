"""
Bonn University EEG dataset loader and task preparation utilities.

Dataset: 5 sets (A-E), 100 segments each, 4096 samples per segment at 173.61 Hz.
    A (Z): Healthy, eyes open (surface)
    B (O): Healthy, eyes closed (surface)
    C (N): Epileptic, seizure-free, opposite hemisphere (depth)
    D (F): Epileptic, seizure-free, focal zone (depth)
    E (S): Epileptic, seizure activity (depth)
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# Set name -> file prefix mapping
SET_PREFIX_MAP = {"A": "Z", "B": "O", "C": "N", "D": "F", "E": "S"}

SAMPLES_PER_SEGMENT = 4096
SEGMENTS_PER_SET = 100


def load_bonn_set(set_name: str, raw_dir: str) -> np.ndarray:
    """Load a single Bonn EEG set and return as a numpy array.

    Args:
        set_name: One of 'A', 'B', 'C', 'D', 'E'.
        raw_dir: Path to the directory containing 'Set A/', 'Set B/', etc.

    Returns:
        np.ndarray of shape (100, 4096) with EEG amplitude values.

    Raises:
        ValueError: If set_name is not one of A-E.
        FileNotFoundError: If set directory or individual files are missing.
    """
    set_name = set_name.upper()
    if set_name not in SET_PREFIX_MAP:
        raise ValueError(
            f"Invalid set_name '{set_name}'. Must be one of {list(SET_PREFIX_MAP.keys())}"
        )

    prefix = SET_PREFIX_MAP[set_name]
    set_dir = Path(raw_dir) / f"Set {set_name}"

    if not set_dir.exists():
        raise FileNotFoundError(
            f"Set directory not found: {set_dir}\n"
            f"Expected structure: {raw_dir}/Set {set_name}/{prefix}001.txt ... {prefix}100.txt"
        )

    segments = []
    for i in range(1, SEGMENTS_PER_SET + 1):
        fname = f"{prefix}{i:03d}.txt"
        fpath = set_dir / fname

        if not fpath.exists():
            raise FileNotFoundError(
                f"EEG file not found: {fpath}\n"
                f"Set {set_name} expects files {prefix}001.txt through {prefix}100.txt"
            )

        # Each .txt file has one amplitude value per line.
        # Some files may have a trailing newline (4097 lines), so we take
        # only the first 4096 samples to ensure consistent array shape.
        signal = np.loadtxt(fpath)[:SAMPLES_PER_SEGMENT]
        segments.append(signal)

    return np.array(segments)  # (100, 4096)


def load_bonn_all(raw_dir: str) -> dict[str, np.ndarray]:
    """Load all 5 Bonn EEG sets.

    Args:
        raw_dir: Path to the directory containing 'Set A/' through 'Set E/'.

    Returns:
        Dict mapping set names ('A'-'E') to arrays of shape (100, 4096).
    """
    data = {}
    for set_name in tqdm(SET_PREFIX_MAP.keys(), desc="Loading Bonn sets"):
        data[set_name] = load_bonn_set(set_name, raw_dir)
    return data


def get_task_data(
    data_dict: dict[str, np.ndarray], task: str
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare X, y arrays for a specific classification task.

    Tasks:
        binary:  A+B (0) vs E (1)           -> 300 samples
        ternary: A+B (0) vs C+D (1) vs E (2) -> 500 samples
        five:    A(0) vs B(1) vs C(2) vs D(3) vs E(4) -> 500 samples

    Args:
        data_dict: Dict from load_bonn_all, mapping set names to arrays.
        task: One of 'binary', 'ternary', 'five'.

    Returns:
        Tuple of (X, y) where X has shape (n_samples, 4096) and y has shape (n_samples,).

    Raises:
        ValueError: If task is not one of the valid options.
    """
    task = task.lower()

    if task == "binary":
        # A+B vs E
        X = np.vstack([data_dict["A"], data_dict["B"], data_dict["E"]])
        y = np.array([0] * 200 + [1] * 100)

    elif task == "ternary":
        # A+B vs C+D vs E
        X = np.vstack([
            data_dict["A"], data_dict["B"],
            data_dict["C"], data_dict["D"],
            data_dict["E"],
        ])
        y = np.array([0] * 200 + [1] * 200 + [2] * 100)

    elif task == "five":
        # A vs B vs C vs D vs E
        X = np.vstack([
            data_dict["A"], data_dict["B"], data_dict["C"],
            data_dict["D"], data_dict["E"],
        ])
        y = np.array([0] * 100 + [1] * 100 + [2] * 100 + [3] * 100 + [4] * 100)

    else:
        raise ValueError(
            f"Invalid task '{task}'. Must be one of 'binary', 'ternary', 'five'"
        )

    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    dist_str = ", ".join(f"class {c} = {n}" for c, n in zip(unique, counts))
    print(f"{task.capitalize()} task: {dist_str} (total = {len(y)})")

    return X, y


def get_cv_folds(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate stratified k-fold cross-validation indices.

    Uses the same seed and fold structure across all experiments to ensure
    every hyperparameter combination sees identical train/test splits.

    Args:
        X: Feature array of shape (n_samples, ...).
        y: Label array of shape (n_samples,).
        n_splits: Number of CV folds (default 5).
        seed: Random seed for reproducibility (default 42).

    Returns:
        List of (train_indices, test_indices) tuples, one per fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(X, y))
