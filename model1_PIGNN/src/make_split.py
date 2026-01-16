from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd


# ============================
# 🔧 CONFIG (EDIT HERE ONLY)
# ============================

MASTER_CSV = "data/chem_master.csv"
TARGETS_CSV = "data/chem_targets_params.csv"
ID_COL = "chem_id"

OUT_DIR = Path("data/splits")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SEED = 42

# ============================


def base_id_from_chem_id(chem_id: str) -> str:
    """
    chemical_001_3 -> chemical_001
    """
    parts = chem_id.split("_")
    if len(parts) < 2:
        return chem_id
    return "_".join(parts[:-1])


def group_ids(chem_ids: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for cid in chem_ids:
        base = base_id_from_chem_id(cid)
        groups.setdefault(base, []).append(cid)

    # deterministic order
    for k in groups:
        groups[k] = sorted(groups[k])
    return groups


def split_groups(
    base_ids: List[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[Set[str], Set[str], Set[str]]:

    rng = random.Random(seed)
    base_ids = list(base_ids)
    rng.shuffle(base_ids)

    n = len(base_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_b = set(base_ids[:n_train])
    val_b = set(base_ids[n_train : n_train + n_val])
    test_b = set(base_ids[n_train + n_val :])

    # safety guard
    if n >= 3:
        if not train_b or not val_b or not test_b:
            train_b = {base_ids[0]}
            val_b = {base_ids[1]}
            test_b = set(base_ids[2:])

    return train_b, val_b, test_b


def write_ids(path: Path, ids: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def main():
    master_df = pd.read_csv(MASTER_CSV)
    targets_df = pd.read_csv(TARGETS_CSV)

    if ID_COL not in master_df.columns:
        raise ValueError(f"{MASTER_CSV} missing column '{ID_COL}'")
    if ID_COL not in targets_df.columns:
        raise ValueError(f"{TARGETS_CSV} missing column '{ID_COL}'")

    master_ids = set(master_df[ID_COL].astype(str))
    target_ids = set(targets_df[ID_COL].astype(str))

    common_ids = sorted(master_ids & target_ids)
    if not common_ids:
        raise ValueError("No overlapping chem_id between input and target CSVs.")

    grouped = group_ids(common_ids)
    base_ids = sorted(grouped.keys())

    train_b, val_b, test_b = split_groups(
        base_ids, TRAIN_RATIO, VAL_RATIO, SEED
    )

    train_ids, val_ids, test_ids = [], [], []

    for base, ids in grouped.items():
        if base in train_b:
            train_ids.extend(ids)
        elif base in val_b:
            val_ids.extend(ids)
        else:
            test_ids.extend(ids)

    train_ids.sort()
    val_ids.sort()
    test_ids.sort()

    write_ids(OUT_DIR / "train.txt", train_ids)
    write_ids(OUT_DIR / "val.txt", val_ids)
    write_ids(OUT_DIR / "test.txt", test_ids)

    print("=== Split summary (group-aware) ===")
    print(f"Total rows: {len(common_ids)}")
    print(f"Base chemicals: {len(base_ids)}")
    print(f"Train: {len(train_ids)} rows")
    print(f"Val  : {len(val_ids)} rows")
    print(f"Test : {len(test_ids)} rows")
    print(f"Written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
