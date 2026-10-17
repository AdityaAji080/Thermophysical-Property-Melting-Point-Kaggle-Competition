"""Molecular featurisation shared by all models.

Each molecule (a SMILES string) is turned into up to four feature blocks:

    morgan_bits    1024  binary Morgan (feature-invariant, radius 2) fingerprint
    morgan_counts  1024  hashed Morgan counts, log1p-transformed
    maccs           167  MACCS structural keys (optional)
    descriptors      32  hand-picked physical/chemical descriptors

Binary blocks are used as-is. Continuous blocks (counts, descriptors) are standardised with
statistics fitted on the training split only (see BlockScaler).
"""

from collections import Counter

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, MACCSkeys, rdmolops
from rdkit.Chem import rdMolDescriptors as rdMD

RDLogger.DisableLog("rdApp.*")

N_BITS = 1024
RADIUS = 2
ELEMENTS = ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
DESCRIPTOR_NAMES = [
    "mol_wt", "exact_mol_wt", "tpsa", "molar_refractivity", "labute_asa", "valence_electrons",
    "h_acceptors", "h_donors", "hba_lipinski", "hbd_lipinski", "nhoh_count", "no_count", "formal_charge",
    "rotatable_bonds", "aromatic_rings", "aliphatic_rings", "saturated_rings",
    "aromatic_heterocycles", "aliphatic_heterocycles", "aromatic_ring_density", "rotatable_bond_density",
    *[f"frac_{e}" for e in ELEMENTS], "is_multi_fragment",
]


def canonical_smiles(smiles):
    """RDKit canonical SMILES, or None if the string cannot be parsed."""
    mol = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(mol) if mol is not None else None


def morgan_bits(mol):
    arr = np.zeros(N_BITS, np.float32)
    if mol is not None:
        fp = rdMD.GetMorganFingerprintAsBitVect(mol, radius=RADIUS, nBits=N_BITS,
                                                useFeatures=True, useChirality=False)
        DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def morgan_counts(mol):
    arr = np.zeros(N_BITS, np.float32)
    if mol is not None:
        fp = rdMD.GetHashedMorganFingerprint(mol, radius=RADIUS, nBits=N_BITS, useFeatures=True)
        for idx, count in fp.GetNonzeroElements().items():
            arr[idx % N_BITS] = count
    return np.log1p(arr)


def maccs_keys(mol):
    arr = np.zeros(167, np.float32)
    if mol is not None:
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), arr)
    return arr


def descriptors(mol, smiles):
    v = np.zeros(len(DESCRIPTOR_NAMES), np.float32)
    if mol is None:
        return v
    heavy = Descriptors.HeavyAtomCount(mol) or 1.0
    atoms = Counter(a.GetSymbol() for a in mol.GetAtoms())
    total = sum(atoms.values()) or 1
    rot = rdMD.CalcNumRotatableBonds(mol)
    arom = rdMD.CalcNumAromaticRings(mol)
    v[:21] = [
        Descriptors.MolWt(mol), rdMD.CalcExactMolWt(mol), Descriptors.TPSA(mol), Crippen.MolMR(mol),
        rdMD.CalcLabuteASA(mol), Descriptors.NumValenceElectrons(mol),
        Descriptors.NumHAcceptors(mol), Descriptors.NumHDonors(mol), rdMD.CalcNumHBA(mol),
        rdMD.CalcNumHBD(mol), Descriptors.NHOHCount(mol), Descriptors.NOCount(mol),
        rdmolops.GetFormalCharge(mol),
        rot, arom, rdMD.CalcNumAliphaticRings(mol), rdMD.CalcNumSaturatedRings(mol),
        rdMD.CalcNumAromaticHeterocycles(mol), rdMD.CalcNumAliphaticHeterocycles(mol),
        arom / heavy, rot / heavy,
    ]
    v[21:31] = [atoms.get(e, 0) / total for e in ELEMENTS]
    v[31] = float("." in str(smiles))
    return v


def featurize(smiles_list, use_maccs=False):
    """Return a dict of raw feature blocks, each of shape (n_molecules, block_width)."""
    blocks = {"morgan_bits": [], "morgan_counts": [], "descriptors": []}
    if use_maccs:
        blocks["maccs"] = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(str(s))
        blocks["morgan_bits"].append(morgan_bits(mol))
        blocks["morgan_counts"].append(morgan_counts(mol))
        blocks["descriptors"].append(descriptors(mol, s))
        if use_maccs:
            blocks["maccs"].append(maccs_keys(mol))
    return {k: np.vstack(v).astype(np.float32) for k, v in blocks.items()}


class BlockScaler:
    """Standardises the continuous blocks; binary blocks pass through unchanged."""

    CONTINUOUS = ("morgan_counts", "descriptors")

    def fit(self, blocks):
        self.stats = {k: (blocks[k].mean(0), blocks[k].std(0) + 1e-8) for k in self.CONTINUOUS}
        return self

    def transform(self, blocks):
        out = []
        for name in ("morgan_bits", "morgan_counts", "maccs", "descriptors"):
            if name not in blocks:
                continue
            x = blocks[name]
            if name in self.stats:
                mean, std = self.stats[name]
                x = (x - mean) / std
            out.append(x)
        return np.hstack(out).astype(np.float32)


def take(blocks, idx):
    """Row-subset every block."""
    return {k: v[idx] for k, v in blocks.items()}
