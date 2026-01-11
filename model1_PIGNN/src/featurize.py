from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data


# --- Atom / bond featurization helpers ---

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


def atom_features(atom: Chem.rdchem.Atom) -> np.ndarray:
    # Keep features simple and stable for v1
    feats = []
    feats.append(atom.GetAtomicNum())                 # integer
    feats.append(atom.GetDegree())                    # integer
    feats.append(atom.GetFormalCharge())              # integer
    feats.append(1 if atom.GetIsAromatic() else 0)    # bool
    feats.append(atom.GetTotalNumHs(includeNeighbors=True))  # integer
    feats.extend(one_hot(atom.GetHybridization(), HYBRIDIZATION))
    feats.append(1 if atom.IsInRing() else 0)
    return np.array(feats, dtype=np.float32)


def bond_features(bond: Chem.rdchem.Bond) -> np.ndarray:
    feats = []
    feats.extend(one_hot(bond.GetBondType(), BOND_TYPES))
    feats.append(1 if bond.GetIsConjugated() else 0)
    feats.append(1 if bond.GetIsAromatic() else 0)
    feats.append(1 if bond.IsInRing() else 0)
    return np.array(feats, dtype=np.float32)


def smiles_to_pyg(smiles: str) -> Tuple[Data, Dict[str, int]]:
    """
    Convert a SMILES string to a PyG Data object:
      - x: node features [N, node_dim]
      - edge_index: [2, E]
      - edge_attr: [E, edge_dim]
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # (Optional but recommended) Add hydrogens? Usually keep implicit H for GNN
    # mol = Chem.AddHs(mol)

    # Node features
    x_list = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(np.stack(x_list, axis=0), dtype=torch.float)

    # Edges (bidirectional)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        # i -> j
        edge_index.append([i, j])
        edge_attr.append(bf)
        # j -> i
        edge_index.append([j, i])
        edge_attr.append(bf)

    if len(edge_index) == 0:
        # Handle molecules with no bonds (rare but possible)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 4 + 3), dtype=torch.float)  # bond feat dim guess
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.stack(edge_attr, axis=0), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    meta = {
        "node_dim": x.size(-1),
        "edge_dim": edge_attr.size(-1),
        "num_atoms": x.size(0),
    }
    return data, meta
