from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# ============================
# 🔧 CONFIG
# ============================

MASTER_CSV = "data/chem_master.csv"
TARGETS_CSV = "data/chem_targets_params.csv"
ID_COL = "chem_id"
SMILES_COL = "smiles"   # used to exclude from scalar columns

OUT_DIR = Path("data/splits")
DEBUG_BAD_PATH = Path("data/debug_bad_scalars.csv")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SEED = 42

GROUP_A_RANGE = (1, 69)
GROUP_B_RANGE = (70, 117)

# Desired proportion inside EACH split: A:B = 2:1
A_TO_B_RATIO = (2, 1)

# Debug settings
DEBUG_VALIDATE_SCALARS = True
DEBUG_MAX_ROWS_PRINT = 30

# If you want to force scalar columns (recommended), set explicitly:
# SCALAR_COLS = ["T"]
SCALAR_COLS = None

# ============================


def base_id_from_chem_id(chem_id: str) -> str:
    parts = str(chem_id).split("_")
    if len(parts) < 2:
        return str(chem_id)
    return "_".join(parts[:-1])


def base_id_number(base_id: str) -> int | None:
    m = re.search(r"(\d+)$", str(base_id))
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def which_group(base_id: str) -> str:
    n = base_id_number(base_id)
    if n is None:
        return "OTHER"
    a_lo, a_hi = GROUP_A_RANGE
    b_lo, b_hi = GROUP_B_RANGE
    if a_lo <= n <= a_hi:
        return "A"
    if b_lo <= n <= b_hi:
        return "B"
    return "OTHER"


def group_ids(chem_ids: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for cid in chem_ids:
        base = base_id_from_chem_id(cid)
        groups.setdefault(base, []).append(str(cid))
    for k in groups:
        groups[k] = sorted(groups[k])
    return groups


def split_counts(total_bases: int) -> Tuple[int, int, int]:
    n_train = int(total_bases * TRAIN_RATIO)
    n_val = int(total_bases * VAL_RATIO)
    n_test = total_bases - n_train - n_val
    if total_bases >= 3:
        if n_train == 0:
            n_train = 1
        if n_val == 0:
            n_val = 1
        n_test = total_bases - n_train - n_val
        if n_test == 0 and n_train > 1:
            n_train -= 1
            n_test += 1
    return n_train, n_val, n_test


def pick_with_ratio(
    pool_a: List[str],
    pool_b: List[str],
    n_pick: int,
    ratio_a: int,
    ratio_b: int,
    rng: random.Random,
) -> Tuple[List[str], List[str], List[str]]:
    picked: List[str] = []
    denom = ratio_a + ratio_b
    target_a = int(round(n_pick * (ratio_a / denom)))
    target_b = n_pick - target_a

    target_a = min(target_a, len(pool_a))
    target_b = min(target_b, len(pool_b))

    while target_a + target_b < n_pick:
        if target_a < len(pool_a):
            target_a += 1
        elif target_b < len(pool_b):
            target_b += 1
        else:
            break

    take_a = pool_a[:target_a]
    take_b = pool_b[:target_b]
    picked.extend(take_a)
    picked.extend(take_b)

    remaining_a = pool_a[target_a:]
    remaining_b = pool_b[target_b:]

    rng.shuffle(picked)
    return picked, remaining_a, remaining_b


def write_ids(path: Path, ids: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def validate_scalars(master_df: pd.DataFrame, scalar_cols: List[str]) -> None:
    """
    Try to convert scalar columns to numeric. If any value fails,
    print which row/chem_id/column caused it and also write a debug CSV.
    """
    df = master_df.copy()

    # make sure chem_id is string for reporting
    df[ID_COL] = df[ID_COL].astype(str)

    bad_records = []

    for c in scalar_cols:
        # Work with raw as string to catch invisible whitespace
        raw = df[c].astype(str)

        # Strip whitespace; treat "nan" strings as empty-like
        stripped = raw.map(lambda x: x.strip())
        empty_like = stripped.eq("") | stripped.str.lower().eq("nan")

        # to_numeric with errors='coerce' makes invalid -> NaN
        numeric = pd.to_numeric(stripped, errors="coerce")
        invalid = numeric.isna() & ~empty_like  # non-empty but still invalid
        blanks = empty_like

        # record blanks
        if blanks.any():
            idxs = df.index[blanks].tolist()
            for i in idxs:
                bad_records.append({
                    "chem_id": df.loc[i, ID_COL],
                    "column": c,
                    "raw_value": raw.loc[i],
                    "issue": "blank_or_nan_string",
                })

        # record invalid numeric values
        if invalid.any():
            idxs = df.index[invalid].tolist()
            for i in idxs:
                bad_records.append({
                    "chem_id": df.loc[i, ID_COL],
                    "column": c,
                    "raw_value": raw.loc[i],
                    "issue": "non_numeric",
                })

    if bad_records:
        bad_df = pd.DataFrame(bad_records)
        DEBUG_BAD_PATH.parent.mkdir(parents=True, exist_ok=True)
        bad_df.to_csv(DEBUG_BAD_PATH, index=False, encoding="utf-8")
        print("\n[DEBUG] Found scalar conversion issues!")
        print(f"[DEBUG] Wrote details to: {DEBUG_BAD_PATH}")
        print("[DEBUG] Showing first few problematic entries:\n")
        for row in bad_records[:DEBUG_MAX_ROWS_PRINT]:
            print(f"  chem_id={row['chem_id']} | col={row['column']} | issue={row['issue']} | raw={repr(row['raw_value'])}")
        raise SystemExit(
            "\nStop. Fix the rows above (or force scalar_cols explicitly) then rerun make_split.py."
        )
    else:
        print("[DEBUG] Scalar columns look numeric-clean ✅")


def main():
    rng = random.Random(SEED)

    master_df = pd.read_csv(MASTER_CSV)
    targets_df = pd.read_csv(TARGETS_CSV)

    if ID_COL not in master_df.columns:
        raise ValueError(f"{MASTER_CSV} missing column '{ID_COL}'")
    if ID_COL not in targets_df.columns:
        raise ValueError(f"{TARGETS_CSV} missing column '{ID_COL}'")

    # determine scalar columns
    if SCALAR_COLS is not None:
        scalar_cols = list(SCALAR_COLS)
    else:
        # auto-detect: everything except chem_id + smiles
        scalar_cols = [c for c in master_df.columns if c not in {ID_COL, SMILES_COL}]
    if not scalar_cols:
        raise ValueError("No scalar columns detected. Set SCALAR_COLS explicitly.")

    if DEBUG_VALIDATE_SCALARS:
        validate_scalars(master_df, scalar_cols)

    master_ids = set(master_df[ID_COL].astype(str))
    target_ids = set(targets_df[ID_COL].astype(str))
    common_ids = sorted(master_ids & target_ids)
    if not common_ids:
        raise ValueError("No overlapping chem_id between input and target CSVs.")

    grouped = group_ids(common_ids)
    base_ids = sorted(grouped.keys())

    base_a = [b for b in base_ids if which_group(b) == "A"]
    base_b = [b for b in base_ids if which_group(b) == "B"]
    base_other = [b for b in base_ids if which_group(b) == "OTHER"]

    rng.shuffle(base_a)
    rng.shuffle(base_b)
    rng.shuffle(base_other)

    total = len(base_ids)
    n_train, n_val, n_test = split_counts(total)
    ra, rb = A_TO_B_RATIO

    train_bases, base_a, base_b = pick_with_ratio(base_a, base_b, n_train, ra, rb, rng)
    val_bases, base_a, base_b = pick_with_ratio(base_a, base_b, n_val, ra, rb, rng)
    test_bases = base_a + base_b

    # distribute OTHER bases to meet counts
    def fill_to_target(split_list: List[str], target_n: int, extras: List[str]) -> List[str]:
        while len(split_list) < target_n and extras:
            split_list.append(extras.pop())
        return split_list

    train_bases = fill_to_target(train_bases, n_train, base_other)
    val_bases = fill_to_target(val_bases, n_val, base_other)
    test_bases = test_bases + base_other

    while len(test_bases) < n_test and len(train_bases) > 1:
        test_bases.append(train_bases.pop())

    # expand base -> row chem_id
    def expand(bases: List[str]) -> List[str]:
        out: List[str] = []
        for b in bases:
            out.extend(grouped[b])
        return sorted(out)

    train_ids = expand(train_bases)
    val_ids = expand(val_bases)
    test_ids = expand(test_bases)

    write_ids(OUT_DIR / "train.txt", train_ids)
    write_ids(OUT_DIR / "val.txt", val_ids)
    write_ids(OUT_DIR / "test.txt", test_ids)

    print("\n=== Split summary (debug-enabled) ===")
    print(f"Scalar cols checked: {scalar_cols}")
    print(f"Train rows: {len(train_ids)} | Val rows: {len(val_ids)} | Test rows: {len(test_ids)}")
    print(f"Written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()




# 70-75 강제로 테스트셋에 넣는 코드
# from __future__ import annotations

# import random
# import re
# from pathlib import Path
# from typing import Dict, List, Set, Tuple

# import pandas as pd


# # ============================
# # 🔧 CONFIG (EDIT HERE ONLY)
# # ============================

# MASTER_CSV = "data/chem_master.csv"
# TARGETS_CSV = "data/chem_targets_params.csv"
# ID_COL = "chem_id"

# OUT_DIR = Path("data/splits")

# TRAIN_RATIO = 0.8
# VAL_RATIO = 0.1
# SEED = 42

# # 🔥 Forced-test base-id numeric range (inclusive)
# FORCED_TEST_RANGE = (70, 75)  # 70~75

# # ============================


# def base_id_from_chem_id(chem_id: str) -> str:
#     """
#     chemical_001_3 -> chemical_001
#     """
#     parts = str(chem_id).split("_")
#     if len(parts) < 2:
#         return str(chem_id)
#     return "_".join(parts[:-1])


# def base_id_number(base_id: str) -> int | None:
#     """
#     Extract trailing integer from something like 'chemical_070' -> 70
#     If not found, return None.
#     """
#     m = re.search(r"(\d+)$", str(base_id))
#     if not m:
#         return None
#     try:
#         return int(m.group(1))
#     except ValueError:
#         return None


# def is_forced_test_base(base_id: str) -> bool:
#     n = base_id_number(base_id)
#     if n is None:
#         return False
#     lo, hi = FORCED_TEST_RANGE
#     return lo <= n <= hi


# def group_ids(chem_ids: List[str]) -> Dict[str, List[str]]:
#     groups: Dict[str, List[str]] = {}
#     for cid in chem_ids:
#         base = base_id_from_chem_id(cid)
#         groups.setdefault(base, []).append(str(cid))

#     # deterministic order
#     for k in groups:
#         groups[k] = sorted(groups[k])
#     return groups


# def split_groups(
#     base_ids: List[str],
#     train_ratio: float,
#     val_ratio: float,
#     seed: int,
# ) -> Tuple[Set[str], Set[str], Set[str]]:
#     """
#     Group-aware split with forced-test base IDs.
#     """
#     base_ids = list(base_ids)

#     # 1) Forced test bases
#     forced_test = {b for b in base_ids if is_forced_test_base(b)}

#     # 2) Remaining bases to split randomly
#     remaining = [b for b in base_ids if b not in forced_test]

#     rng = random.Random(seed)
#     rng.shuffle(remaining)

#     n = len(remaining)
#     if n == 0:
#         # Everything forced to test
#         return set(), set(), set(forced_test)

#     n_train = int(n * train_ratio)
#     n_val = int(n * val_ratio)

#     train_b = set(remaining[:n_train])
#     val_b = set(remaining[n_train : n_train + n_val])
#     test_b = set(remaining[n_train + n_val :])

#     # safety guard (ensure non-empty splits when possible)
#     if n >= 3:
#         if not train_b or not val_b or not test_b:
#             train_b = {remaining[0]}
#             val_b = {remaining[1]}
#             test_b = set(remaining[2:])

#     # 3) Add forced test bases
#     test_b |= forced_test

#     # Ensure forced ones are not leaking
#     train_b -= forced_test
#     val_b -= forced_test

#     return train_b, val_b, test_b


# def write_ids(path: Path, ids: List[str]) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     path.write_text("\n".join(ids) + "\n", encoding="utf-8")


# def main():
#     master_df = pd.read_csv(MASTER_CSV)
#     targets_df = pd.read_csv(TARGETS_CSV)

#     if ID_COL not in master_df.columns:
#         raise ValueError(f"{MASTER_CSV} missing column '{ID_COL}'")
#     if ID_COL not in targets_df.columns:
#         raise ValueError(f"{TARGETS_CSV} missing column '{ID_COL}'")

#     master_ids = set(master_df[ID_COL].astype(str))
#     target_ids = set(targets_df[ID_COL].astype(str))

#     common_ids = sorted(master_ids & target_ids)
#     if not common_ids:
#         raise ValueError("No overlapping chem_id between input and target CSVs.")

#     grouped = group_ids(common_ids)
#     base_ids = sorted(grouped.keys())

#     train_b, val_b, test_b = split_groups(base_ids, TRAIN_RATIO, VAL_RATIO, SEED)

#     train_ids, val_ids, test_ids = [], [], []
#     for base, ids in grouped.items():
#         if base in train_b:
#             train_ids.extend(ids)
#         elif base in val_b:
#             val_ids.extend(ids)
#         else:
#             test_ids.extend(ids)

#     train_ids.sort()
#     val_ids.sort()
#     test_ids.sort()

#     write_ids(OUT_DIR / "train.txt", train_ids)
#     write_ids(OUT_DIR / "val.txt", val_ids)
#     write_ids(OUT_DIR / "test.txt", test_ids)

#     forced = sorted([b for b in base_ids if is_forced_test_base(b)])
#     print("=== Split summary (group-aware + forced test) ===")
#     print(f"Total rows: {len(common_ids)}")
#     print(f"Base chemicals: {len(base_ids)}")
#     print(f"Forced test bases ({FORCED_TEST_RANGE[0]}~{FORCED_TEST_RANGE[1]}): {len(forced)}")
#     if forced:
#         print("Forced test list:", forced[:10], ("..." if len(forced) > 10 else ""))
#     print(f"Train: {len(train_ids)} rows")
#     print(f"Val  : {len(val_ids)} rows")
#     print(f"Test : {len(test_ids)} rows")
#     print(f"Written to: {OUT_DIR.resolve()}")


# if __name__ == "__main__":
#     main()



# from __future__ import annotations

# import random
# from pathlib import Path
# from typing import Dict, List, Set, Tuple

# import pandas as pd


# # ============================
# # 🔧 CONFIG (EDIT HERE ONLY)
# # ============================

# MASTER_CSV = "data/chem_master.csv"
# TARGETS_CSV = "data/chem_targets_params.csv"
# ID_COL = "chem_id"

# OUT_DIR = Path("data/splits")

# TRAIN_RATIO = 0.8
# VAL_RATIO = 0.1
# SEED = 42

# # ============================


# def base_id_from_chem_id(chem_id: str) -> str:
#     """
#     chemical_001_3 -> chemical_001
#     """
#     parts = chem_id.split("_")
#     if len(parts) < 2:
#         return chem_id
#     return "_".join(parts[:-1])


# def group_ids(chem_ids: List[str]) -> Dict[str, List[str]]:
#     groups: Dict[str, List[str]] = {}
#     for cid in chem_ids:
#         base = base_id_from_chem_id(cid)
#         groups.setdefault(base, []).append(cid)

#     # deterministic order
#     for k in groups:
#         groups[k] = sorted(groups[k])
#     return groups


# def split_groups(
#     base_ids: List[str],
#     train_ratio: float,
#     val_ratio: float,
#     seed: int,
# ) -> Tuple[Set[str], Set[str], Set[str]]:

#     rng = random.Random(seed)
#     base_ids = list(base_ids)
#     rng.shuffle(base_ids)

#     n = len(base_ids)
#     n_train = int(n * train_ratio)
#     n_val = int(n * val_ratio)

#     train_b = set(base_ids[:n_train])
#     val_b = set(base_ids[n_train : n_train + n_val])
#     test_b = set(base_ids[n_train + n_val :])

#     # safety guard
#     if n >= 3:
#         if not train_b or not val_b or not test_b:
#             train_b = {base_ids[0]}
#             val_b = {base_ids[1]}
#             test_b = set(base_ids[2:])

#     return train_b, val_b, test_b


# def write_ids(path: Path, ids: List[str]) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     path.write_text("\n".join(ids) + "\n", encoding="utf-8")


# def main():
#     master_df = pd.read_csv(MASTER_CSV)
#     targets_df = pd.read_csv(TARGETS_CSV)

#     if ID_COL not in master_df.columns:
#         raise ValueError(f"{MASTER_CSV} missing column '{ID_COL}'")
#     if ID_COL not in targets_df.columns:
#         raise ValueError(f"{TARGETS_CSV} missing column '{ID_COL}'")

#     master_ids = set(master_df[ID_COL].astype(str))
#     target_ids = set(targets_df[ID_COL].astype(str))

#     common_ids = sorted(master_ids & target_ids)
#     if not common_ids:
#         raise ValueError("No overlapping chem_id between input and target CSVs.")

#     grouped = group_ids(common_ids)
#     base_ids = sorted(grouped.keys())

#     train_b, val_b, test_b = split_groups(
#         base_ids, TRAIN_RATIO, VAL_RATIO, SEED
#     )

#     train_ids, val_ids, test_ids = [], [], []

#     for base, ids in grouped.items():
#         if base in train_b:
#             train_ids.extend(ids)
#         elif base in val_b:
#             val_ids.extend(ids)
#         else:
#             test_ids.extend(ids)

#     train_ids.sort()
#     val_ids.sort()
#     test_ids.sort()

#     write_ids(OUT_DIR / "train.txt", train_ids)
#     write_ids(OUT_DIR / "val.txt", val_ids)
#     write_ids(OUT_DIR / "test.txt", test_ids)

#     print("=== Split summary (group-aware) ===")
#     print(f"Total rows: {len(common_ids)}")
#     print(f"Base chemicals: {len(base_ids)}")
#     print(f"Train: {len(train_ids)} rows")
#     print(f"Val  : {len(val_ids)} rows")
#     print(f"Test : {len(test_ids)} rows")
#     print(f"Written to: {OUT_DIR.resolve()}")


# if __name__ == "__main__":
#     main()
