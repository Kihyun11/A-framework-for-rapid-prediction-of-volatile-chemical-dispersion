from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, rdMolDescriptors, rdPartialCharges
from torch_geometric.data import Data


# ----------------------------
# Element tables / lookups
# ----------------------------

# Period and group for common elements (atomic number -> (group, period)).
# (Approximate periodic table mapping; for elements not listed we fall back to (0,0).)
# Groups: 1..18, Periods: 1..7
GROUP_PERIOD = {
    1: (1, 1),   # H
    5: (13, 2),  # B
    6: (14, 2),  # C
    7: (15, 2),  # N
    8: (16, 2),  # O
    9: (17, 2),  # F
    14: (14, 3), # Si
    15: (15, 3), # P
    16: (16, 3), # S
    17: (17, 3), # Cl
    35: (17, 4), # Br
    53: (17, 5), # I
    11: (1, 3),  # Na
    12: (2, 3),  # Mg
    19: (1, 4),  # K
    20: (2, 4),  # Ca
    3: (1, 2),   # Li
    4: (2, 2),   # Be
    13: (13, 3), # Al
    26: (8, 4),  # Fe (transition; group approx)
    29: (11, 4), # Cu
    30: (12, 4), # Zn
}

# Pauling electronegativity for common elements (atomic number -> EN)
ELECTRONEGATIVITY = {
    1: 2.20,  # H
    5: 2.04,  # B
    6: 2.55,  # C
    7: 3.04,  # N
    8: 3.44,  # O
    9: 3.98,  # F
    14: 1.90, # Si
    15: 2.19, # P
    16: 2.58, # S
    17: 3.16, # Cl
    35: 2.96, # Br
    53: 2.66, # I
    11: 0.93, # Na
    12: 1.31, # Mg
    19: 0.82, # K
    20: 1.00, # Ca
    3: 0.98,  # Li
    4: 1.57,  # Be
    13: 1.61, # Al
    26: 1.83, # Fe
    29: 1.90, # Cu
    30: 1.65, # Zn
}

# Approx atomic polarizability (very rough, in Å^3) for common elements.
# For modeling, rough proxies can still help; unknown elements fall back to 0.0.
POLARIZABILITY = {
    1: 0.667,   # H
    5: 3.03,    # B
    6: 1.76,    # C
    7: 1.10,    # N
    8: 0.802,   # O
    9: 0.557,   # F
    14: 5.38,   # Si
    15: 3.63,   # P
    16: 2.90,   # S
    17: 2.18,   # Cl
    35: 3.05,   # Br
    53: 5.35,   # I
}

# Hybridization and bond types (as before)
HYBRIDIZATION = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def one_hot(value, choices: List) -> List[int]:
    return [1 if value == c else 0 for c in choices]


def one_hot_int(value: int, min_v: int, max_v: int) -> List[int]:
    # Clamp out-of-range to all-zeros (or you could add an "other" bin)
    if value < min_v or value > max_v:
        return [0] * (max_v - min_v + 1)
    out = [0] * (max_v - min_v + 1)
    out[value - min_v] = 1
    return out


# ----------------------------
# 3D helpers (optional)
# ----------------------------

def try_embed_3d(mol: Chem.Mol, seed: int = 0) -> Optional[int]:
    """
    Try to add a 3D conformer to mol. Returns conformer id if success, else None.
    """
    mol3d = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    cid = AllChem.EmbedMolecule(mol3d, params)
    if cid < 0:
        return None
    # Optimize (optional, but helps)
    try:
        AllChem.UFFOptimizeMolecule(mol3d, maxIters=200)
    except Exception:
        pass
    # Copy conformer back to original mol (without Hs) by using mol3d for coordinates
    # We'll just return mol3d and use that for distance computations.
    return cid, mol3d


def get_pos(mol3d: Chem.Mol, atom_idx: int) -> np.ndarray:
    conf = mol3d.GetConformer()
    p = conf.GetAtomPosition(atom_idx)
    return np.array([p.x, p.y, p.z], dtype=np.float32)


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    # angle ABC (at B) in radians
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-12)
    cosv = float(np.dot(ba, bc) / denom)
    cosv = max(-1.0, min(1.0, cosv))
    return float(np.arccos(cosv))


def dihedral(p0, p1, p2, p3) -> float:
    # Returns dihedral angle in radians
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1 /= (np.linalg.norm(b1) + 1e-12)

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.arctan2(y, x))


# ----------------------------
# Feature builders
# ----------------------------

def atom_features(atom: Chem.rdchem.Atom, gasteiger_charge: float) -> np.ndarray:
    z = atom.GetAtomicNum()

    feats: List[float] = []

    # (Base)
    feats.append(float(z))  # atomic number as scalar
    feats.append(float(atom.GetMass()))  # atomic mass

    # Element group/period one-hot (1..18, 1..7)
    group, period = GROUP_PERIOD.get(z, (0, 0))
    feats.extend(one_hot_int(group, 1, 18))   # 18 dims
    feats.extend(one_hot_int(period, 1, 7))   # 7 dims

    # Degree features
    total_degree = atom.GetDegree()  # neighbors count (heavy atoms)
    heavy_degree = 0
    for nbr in atom.GetNeighbors():
        if nbr.GetAtomicNum() > 1:
            heavy_degree += 1
    feats.append(float(total_degree))
    feats.append(float(heavy_degree))

    # Radical electrons
    feats.append(float(atom.GetNumRadicalElectrons()))

    # Charge/polarity proxies
    feats.append(float(atom.GetFormalCharge()))
    feats.append(float(gasteiger_charge))  # Gasteiger partial charge
    feats.append(float(ELECTRONEGATIVITY.get(z, 0.0)))
    feats.append(float(POLARIZABILITY.get(z, 0.0)))

    # Aromaticity / hydrogens / hybridization / ring membership
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)
    feats.append(float(atom.GetTotalNumHs(includeNeighbors=True)))
    feats.extend(one_hot(atom.GetHybridization(), HYBRIDIZATION))  # 6 dims
    feats.append(1.0 if atom.IsInRing() else 0.0)

    # Ring size flags (3..8)
    for n in [3, 4, 5, 6, 7, 8]:
        feats.append(1.0 if atom.IsInRingSize(n) else 0.0)

    # Aromatic ring membership (simple, stable proxy)
    feats.append(1.0 if (atom.GetIsAromatic() and atom.IsInRing()) else 0.0)

    return np.array(feats, dtype=np.float32)


def bond_features(
    bond: Chem.rdchem.Bond,
    i: int,
    j: int,
    mol: Chem.Mol,
    mol3d: Optional[Chem.Mol],
) -> np.ndarray:
    feats: List[float] = []

    # One-hot bond type (as before)
    feats.extend(one_hot(bond.GetBondType(), BOND_TYPES))  # 4 dims

    # Flags
    feats.append(1.0 if bond.GetIsConjugated() else 0.0)
    feats.append(1.0 if bond.GetIsAromatic() else 0.0)
    feats.append(1.0 if bond.IsInRing() else 0.0)

    # Bond order float
    feats.append(float(bond.GetBondTypeAsDouble()))

    # "Is in aromatic system" (proxy): aromatic bond OR both atoms aromatic
    a_i = mol.GetAtomWithIdx(i)
    a_j = mol.GetAtomWithIdx(j)
    aromatic_system = bond.GetIsAromatic() or (a_i.GetIsAromatic() and a_j.GetIsAromatic())
    feats.append(1.0 if aromatic_system else 0.0)

    # 3D geometry features (optional)
    # bond length, distance-based weight, angle, dihedral
    if mol3d is not None:
        pi = get_pos(mol3d, i)
        pj = get_pos(mol3d, j)
        dist = float(np.linalg.norm(pi - pj))
        feats.append(dist)                      # bond length
        feats.append(1.0 / (dist + 1e-6))       # distance-based edge weight

        # Angle feature for directed edge i->j: pick a neighbor k of i (k != j)
        # Deterministic: choose lowest-index neighbor if exists
        nbrs_i = sorted([a.GetIdx() for a in mol.GetAtomWithIdx(i).GetNeighbors() if a.GetIdx() != j])
        if len(nbrs_i) > 0:
            k = nbrs_i[0]
            pk = get_pos(mol3d, k)
            ang = angle(pk, pi, pj)  # k - i - j
        else:
            ang = 0.0
        feats.append(float(ang))

        # Dihedral feature i->j: pick neighbor k of i != j and l of j != i
        nbrs_j = sorted([a.GetIdx() for a in mol.GetAtomWithIdx(j).GetNeighbors() if a.GetIdx() != i])
        if len(nbrs_i) > 0 and len(nbrs_j) > 0:
            k = nbrs_i[0]
            l = nbrs_j[0]
            pk = get_pos(mol3d, k)
            pl = get_pos(mol3d, l)
            dih = dihedral(pk, pi, pj, pl)  # k - i - j - l
        else:
            dih = 0.0
        feats.append(float(dih))
    else:
        # Fill zeros if 3D not available
        feats.extend([0.0, 0.0, 0.0, 0.0])

    return np.array(feats, dtype=np.float32)


# ----------------------------
# Main conversion
# ----------------------------

def smiles_to_pyg(
    smiles: str,
    *,
    use_3d: bool = True,
    seed: int = 0,
) -> Tuple[Data, Dict[str, int]]:
    """
    Convert SMILES -> PyG Data:
      - x: [N, node_dim] atom features (includes Gasteiger charge etc.)
      - edge_index: [2, E]
      - edge_attr: [E, edge_dim] bond features + (optional) 3D geometry
      - g: [2] graph-level features (LogP, TPSA)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Compute Gasteiger charges (requires Hs for better stability)
    mol_for_charge = Chem.AddHs(mol)
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol_for_charge)
    except Exception:
        # If charge computation fails, fall back to 0 charges
        pass

    # Build a per-atom charge list mapped to heavy atoms of original mol
    charges: List[float] = []
    # Map heavy atom indices between mol and mol_for_charge: heavy atoms are first in RDKit
    for a in mol.GetAtoms():
        idx = a.GetIdx()
        a_ch = mol_for_charge.GetAtomWithIdx(idx)
        try:
            q = float(a_ch.GetProp("_GasteigerCharge"))
            if not np.isfinite(q):
                q = 0.0
        except Exception:
            q = 0.0
        charges.append(q)

    # Optional 3D embedding (to get bond lengths/angles/dihedrals)
    mol3d: Optional[Chem.Mol] = None
    if use_3d:
        try:
            res = try_embed_3d(mol, seed=seed)
            if res is not None:
                _, mol3d = res
        except Exception:
            mol3d = None

    # Node features
    x_list = [atom_features(a, charges[a.GetIdx()]) for a in mol.GetAtoms()]
    x = torch.tensor(np.stack(x_list, axis=0), dtype=torch.float32)

    # Edges (bidirectional)
    edge_index: List[List[int]] = []
    edge_attr: List[np.ndarray] = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bf_ij = bond_features(bond, i, j, mol, mol3d)
        bf_ji = bond_features(bond, j, i, mol, mol3d)

        edge_index.append([i, j])
        edge_attr.append(bf_ij)
        edge_index.append([j, i])
        edge_attr.append(bf_ji)

    if len(edge_index) == 0:
        # Handle molecules with no bonds
        edge_index_t = torch.zeros((2, 0), dtype=torch.long)
        # Infer edge_dim from a dummy bond_features template (use zeros)
        # edge_dim = 4(onehot) + 3(flags) + 1(bond order) + 1(aromatic system) + 4(3D geom)
        edge_dim = 4 + 3 + 1 + 1 + 4
        edge_attr_t = torch.zeros((0, edge_dim), dtype=torch.float32)
    else:
        edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_t = torch.tensor(np.stack(edge_attr, axis=0), dtype=torch.float32)

    # Graph-level descriptors
    logp = float(Crippen.MolLogP(mol))
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    g = torch.tensor([logp, tpsa], dtype=torch.float32)  # [2]

    data = Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr_t)
    data.g = g  # graph-level feature vector

    meta = {
        "node_dim": int(x.size(-1)),
        "edge_dim": int(edge_attr_t.size(-1)),
        "g_dim": int(g.numel()),
        "num_atoms": int(x.size(0)),
        "use_3d": int(use_3d),
    }
    return data, meta








#Old version

# from __future__ import annotations

# from typing import Dict, List, Tuple

# import numpy as np
# import torch
# from rdkit import Chem
# from torch_geometric.data import Data


# # --- Atom / bond featurization helpers ---

# HYBRIDIZATION = [
#     Chem.rdchem.HybridizationType.S,
#     Chem.rdchem.HybridizationType.SP,
#     Chem.rdchem.HybridizationType.SP2,
#     Chem.rdchem.HybridizationType.SP3,
#     Chem.rdchem.HybridizationType.SP3D,
#     Chem.rdchem.HybridizationType.SP3D2,
# ]

# BOND_TYPES = [
#     Chem.rdchem.BondType.SINGLE,
#     Chem.rdchem.BondType.DOUBLE,
#     Chem.rdchem.BondType.TRIPLE,
#     Chem.rdchem.BondType.AROMATIC,
# ]


# def one_hot(value, choices: List) -> List[int]:
#     return [1 if value == c else 0 for c in choices]


# def atom_features(atom: Chem.rdchem.Atom) -> np.ndarray:
#     # Keep features simple and stable for v1
#     feats = []
#     feats.append(atom.GetAtomicNum())                 # integer
#     feats.append(atom.GetDegree())                    # integer
#     feats.append(atom.GetFormalCharge())              # integer
#     feats.append(1 if atom.GetIsAromatic() else 0)    # bool
#     feats.append(atom.GetTotalNumHs(includeNeighbors=True))  # integer
#     feats.extend(one_hot(atom.GetHybridization(), HYBRIDIZATION))
#     feats.append(1 if atom.IsInRing() else 0)
#     return np.array(feats, dtype=np.float32)


# def bond_features(bond: Chem.rdchem.Bond) -> np.ndarray:
#     feats = []
#     feats.extend(one_hot(bond.GetBondType(), BOND_TYPES))
#     feats.append(1 if bond.GetIsConjugated() else 0)
#     feats.append(1 if bond.GetIsAromatic() else 0)
#     feats.append(1 if bond.IsInRing() else 0)
#     return np.array(feats, dtype=np.float32)


# def smiles_to_pyg(smiles: str) -> Tuple[Data, Dict[str, int]]:
#     """
#     Convert a SMILES string to a PyG Data object:
#       - x: node features [N, node_dim]
#       - edge_index: [2, E]
#       - edge_attr: [E, edge_dim]
#     """
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         raise ValueError(f"Invalid SMILES: {smiles}")

#     # (Optional but recommended) Add hydrogens? Usually keep implicit H for GNN
#     # mol = Chem.AddHs(mol)

#     # Node features
#     x_list = [atom_features(a) for a in mol.GetAtoms()]
#     x = torch.tensor(np.stack(x_list, axis=0), dtype=torch.float)

#     # Edges (bidirectional)
#     edge_index = []
#     edge_attr = []
#     for bond in mol.GetBonds():
#         i = bond.GetBeginAtomIdx()
#         j = bond.GetEndAtomIdx()
#         bf = bond_features(bond)
#         # i -> j
#         edge_index.append([i, j])
#         edge_attr.append(bf)
#         # j -> i
#         edge_index.append([j, i])
#         edge_attr.append(bf)

#     if len(edge_index) == 0:
#         # Handle molecules with no bonds (rare but possible)
#         edge_index = torch.zeros((2, 0), dtype=torch.long)
#         edge_attr = torch.zeros((0, 4 + 3), dtype=torch.float)  # bond feat dim guess
#     else:
#         edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#         edge_attr = torch.tensor(np.stack(edge_attr, axis=0), dtype=torch.float)

#     data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
#     meta = {
#         "node_dim": x.size(-1),
#         "edge_dim": edge_attr.size(-1),
#         "num_atoms": x.size(0),
#     }
#     return data, meta
