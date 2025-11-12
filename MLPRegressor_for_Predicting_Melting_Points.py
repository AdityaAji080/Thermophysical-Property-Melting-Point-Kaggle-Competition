import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors as rdMD, Crippen, rdmolops, MACCSkeys
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# -------------------------------------------------
# Data
# -------------------------------------------------
train_df = pd.read_csv('train_2.csv')
test_df  = pd.read_csv('test.csv')

INCLUDE_GROUPS = False
N_BITS, Radius = 1024, 2

# -------------------------------------------------
# Feature functions
# -------------------------------------------------

def morgan_fingerprint(smiles):
    """Binary Morgan bits (0/1)"""
    arr = np.zeros((N_BITS,), np.float32)
    mol = Chem.MolFromSmiles(str(smiles))
    if not mol:
        return arr
    fp = rdMD.GetMorganFingerprintAsBitVect(
        mol,
        radius=Radius,
        nBits=N_BITS,
        useFeatures=True,
        useChirality=False  # <- this helped you
    )
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def morgan_counts(smiles, n_bits=N_BITS, radius=Radius):
    """Count-based Morgan, then we'll log1p and standardize later"""
    arr = np.zeros((n_bits,), np.float32)
    mol = Chem.MolFromSmiles(str(smiles))
    if not mol:
        return arr
    fp = rdMD.GetHashedMorganFingerprint(
        mol,
        radius=radius,
        nBits=n_bits,
        useFeatures=True
    )
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def maccs_bits(smiles):
    """MACCS 167-bit fingerprint (binary)"""
    mol = Chem.MolFromSmiles(str(smiles))
    if not mol:
        return np.zeros((167,), np.float32)
    fp = MACCSkeys.GenMACCSKeys(mol)  # RDKit MACCS keys
    arr = np.zeros((fp.GetNumBits(),), np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def chem_features(smiles):
    """
    Hand-crafted physical/chemical descriptors (32 floats).
    We'll standardize this block later.
    """
    v = np.zeros(32, np.float32)
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return v

    heavy = Descriptors.HeavyAtomCount(mol) or 1.0
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    from collections import Counter
    counts = Counter(atoms)
    total_atoms = sum(counts.values()) or 1

    # size / mass / polarity
    v[0]  = Descriptors.MolWt(mol)
    v[1]  = rdMD.CalcExactMolWt(mol)
    v[2]  = Descriptors.TPSA(mol)
    v[3]  = Crippen.MolMR(mol)
    v[4]  = rdMD.CalcLabuteASA(mol)
    v[5]  = Descriptors.NumValenceElectrons(mol)

    # H-bond / charge
    v[6]  = Descriptors.NumHAcceptors(mol)
    v[7]  = Descriptors.NumHDonors(mol)
    v[8]  = rdMD.CalcNumHBA(mol)
    v[9]  = rdMD.CalcNumHBD(mol)
    v[10] = Descriptors.NHOHCount(mol)
    v[11] = Descriptors.NOCount(mol)
    v[12] = rdmolops.GetFormalCharge(mol)

    # rigidity / rings / flexibility
    v[13] = rdMD.CalcNumRotatableBonds(mol)
    v[14] = rdMD.CalcNumAromaticRings(mol)
    v[15] = rdMD.CalcNumAliphaticRings(mol)
    v[16] = rdMD.CalcNumSaturatedRings(mol)
    v[17] = rdMD.CalcNumAromaticHeterocycles(mol)
    v[18] = rdMD.CalcNumAliphaticHeterocycles(mol)
    v[19] = v[14] / heavy
    v[20] = rdMD.CalcNumRotatableBonds(mol) / heavy

    # elemental fractions
    elt_list = ["C","H","N","O","S","P","F","Cl","Br","I"]
    base = 21
    for i, e in enumerate(elt_list):
        v[base + i] = counts.get(e, 0) / total_atoms

    # salt / multi-fragment flag
    v[31] = 1.0 if "." in str(smiles) else 0.0

    return v

# -------------------------------------------------
# Build TRAIN feature blocks
# -------------------------------------------------

bits_train    = np.vstack([morgan_fingerprint(s) for s in train_df["SMILES"]])          # (n, 1024)
counts_train  = np.vstack([morgan_counts(s)      for s in train_df["SMILES"]]).astype(np.float32)
counts_train  = np.log1p(counts_train).astype(np.float32)                               # log1p
maccs_train   = np.vstack([maccs_bits(s)          for s in train_df["SMILES"]]).astype(np.float32)  # (n,167)
desc_train    = np.vstack([chem_features(s)       for s in train_df["SMILES"]]).astype(np.float32)  # (n,32)

# standardize continuous blocks: counts and descriptors
counts_mean = counts_train.mean(axis=0, keepdims=True)
counts_std  = counts_train.std(axis=0, keepdims=True) + 1e-8
counts_train_s = (counts_train - counts_mean) / counts_std

desc_mean = desc_train.mean(axis=0, keepdims=True)
desc_std  = desc_train.std(axis=0, keepdims=True) + 1e-8
desc_train_s = (desc_train - desc_mean) / desc_std

# now stack ALL blocks for training
# NOTE: we do NOT scale bits_train or maccs_train (they're binary indicator-ish)
X_full = np.hstack([
    bits_train,        # raw Morgan bits
    counts_train_s,    # scaled log(counts)
    maccs_train,       # raw MACCS bits
    desc_train_s       # scaled chem descriptors
]).astype(np.float32)

Y_full = train_df["Tm"].to_numpy(dtype=np.float32)

# train/val split
X_train, X_val, Y_train, Y_val = train_test_split(
    X_full, Y_full,
    test_size=0.2,
    random_state=42
)

# -------------------------------------------------
# Model: 3-layer tanh
# -------------------------------------------------
pipe = Pipeline([
    ("scaler", "passthrough"),  # we're already doing scaling manually
    ("mlp", MLPRegressor(
        hidden_layer_sizes=(512,256,128),
        activation="tanh",
        solver="adam",
        alpha=1e-3,
        learning_rate_init=1e-4,
        max_iter=2000,
        early_stopping=True,
        n_iter_no_change=100,
        validation_fraction=0.15,
        random_state=42
    ))
])

pd.DataFrame(X_train).to_csv("Final_Features_df.csv", index=False)

pipe.fit(X_train, Y_train)

y_pred = pipe.predict(X_val)
mae  = mean_absolute_error(Y_val, y_pred)
rmse = np.sqrt(mean_squared_error(Y_val, y_pred))
r2   = r2_score(Y_val, y_pred)
print(f"MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.3f}")

# -------------------------------------------------
# Save everything needed for inference
# -------------------------------------------------
joblib.dump({
    "pipeline": pipe,
    "n_bits": N_BITS,
    "radius": Radius,
    "counts_mean": counts_mean,
    "counts_std": counts_std,
    "desc_mean": desc_mean,
    "desc_std": desc_std
}, "mlp_smiles_maccs.joblib")

# -------------------------------------------------
# Build TEST feature blocks (apply SAME transforms)
# -------------------------------------------------
bundle = joblib.load("mlp_smiles_maccs.joblib")
pipe = bundle["pipeline"]
counts_mean = bundle["counts_mean"]
counts_std  = bundle["counts_std"]
desc_mean   = bundle["desc_mean"]
desc_std    = bundle["desc_std"]

ids = test_df["id"].to_numpy() if "id" in test_df.columns else np.arange(len(test_df))

bits_test    = np.vstack([morgan_fingerprint(s) for s in test_df["SMILES"]])
counts_test  = np.vstack([morgan_counts(s)      for s in test_df["SMILES"]]).astype(np.float32)
counts_test  = np.log1p(counts_test).astype(np.float32)
maccs_test   = np.vstack([maccs_bits(s)         for s in test_df["SMILES"]]).astype(np.float32)
desc_test    = np.vstack([chem_features(s)      for s in test_df["SMILES"]]).astype(np.float32)

counts_test_s = (counts_test - counts_mean) / counts_std
desc_test_s   = (desc_test   - desc_mean)   / desc_std

X_test_final = np.hstack([
    bits_test,
    counts_test_s,
    maccs_test,
    desc_test_s
]).astype(np.float32)

test_preds = pipe.predict(X_test_final)

out = pd.DataFrame({"id": ids, "Tm": test_preds})
print(out.head())
out.to_csv("Tm_predictions1.csv", index=False)
