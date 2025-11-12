import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors as rdMD, Crippen, rdmolops
from rdkit import RDLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool
import joblib

RDLogger.DisableLog("rdApp.*")

# --- data
train_df = pd.read_csv('train_2.csv')
test_df  = pd.read_csv('test.csv')

INCLUDE_GROUPS = False
N_BITS, Radius = 1024, 2

def morgan_fingerprint(smiles):
    arr = np.zeros((N_BITS,), np.float32)
    mol = Chem.MolFromSmiles(str(smiles))
    if not mol: return arr
    fp = rdMD.GetMorganFingerprintAsBitVect(
        mol, radius=Radius, nBits=N_BITS,
        useFeatures=True, useChirality=False
    )
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def morgan_counts(smiles, n_bits=N_BITS, radius=Radius):
    arr = np.zeros((n_bits,), np.float32)
    mol = Chem.MolFromSmiles(str(smiles))
    if not mol:
        return arr
    fp = rdMD.GetHashedMorganFingerprint(
        mol, radius=radius, nBits=n_bits, useFeatures=True
    )  # SparseIntVect with counts
    for idx, cnt in fp.GetNonzeroElements().items():
        arr[idx % n_bits] = float(cnt)
    return arr

def chem_features(smiles):
    v = np.zeros(32, np.float32)
    m = Chem.MolFromSmiles(str(smiles))
    if m is None: return v

    heavy = Descriptors.HeavyAtomCount(m) or 1.0
    atoms = [a.GetSymbol() for a in m.GetAtoms()]
    from collections import Counter
    counts = Counter(atoms)
    total_atoms = sum(counts.values()) or 1

    # size / mass / polarity
    v[0]  = Descriptors.MolWt(m)
    v[1]  = rdMD.CalcExactMolWt(m)
    v[2]  = Descriptors.TPSA(m)
    v[3]  = Crippen.MolMR(m)
    v[4]  = rdMD.CalcLabuteASA(m)
    v[5]  = Descriptors.NumValenceElectrons(m)

    # H-bond / charge
    v[6]  = Descriptors.NumHAcceptors(m)
    v[7]  = Descriptors.NumHDonors(m)
    v[8]  = rdMD.CalcNumHBA(m)
    v[9]  = rdMD.CalcNumHBD(m)
    v[10] = Descriptors.NHOHCount(m)
    v[11] = Descriptors.NOCount(m)
    v[12] = rdmolops.GetFormalCharge(m)

    # rigidity / rings / flexibility
    v[13] = rdMD.CalcNumRotatableBonds(m)
    v[14] = rdMD.CalcNumAromaticRings(m)
    v[15] = rdMD.CalcNumAliphaticRings(m)
    v[16] = rdMD.CalcNumSaturatedRings(m)
    v[17] = rdMD.CalcNumAromaticHeterocycles(m)
    v[18] = rdMD.CalcNumAliphaticHeterocycles(m)
    v[19] = v[14] / heavy
    v[20] = (rdMD.CalcNumRotatableBonds(m) / heavy)

    # element fractions
    elt_list = ["C","H","N","O","S","P","F","Cl","Br","I"]
    base = 21
    for i, e in enumerate(elt_list):
        v[base + i] = counts.get(e,0) / total_atoms

    # salt / mixture flag
    v[31] = 1.0 if "." in str(smiles) else 0.0
    return v

# --- TRAIN features
bits_train   = np.vstack([morgan_fingerprint(s) for s in train_df["SMILES"]]).astype(np.float32)
counts_train = np.vstack([morgan_counts(s) for s in train_df["SMILES"]]).astype(np.float32)
counts_train = np.log1p(counts_train).astype(np.float32)
desc_train   = np.vstack([chem_features(s) for s in train_df["SMILES"]]).astype(np.float32)

# standardize counts & descriptors (persist stats)
desc_mean = desc_train.mean(axis=0, keepdims=True); desc_std  = desc_train.std(axis=0, keepdims=True) + 1e-8
counts_mean = counts_train.mean(axis=0, keepdims=True); counts_std = counts_train.std(axis=0, keepdims=True) + 1e-8

desc_train_s   = (desc_train - desc_mean) / desc_std
counts_train_s = (counts_train - counts_mean) / counts_std

X = np.hstack([bits_train, counts_train_s, desc_train_s]).astype(np.float32)
Y = train_df["Tm"].to_numpy(dtype=np.float32)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# --- CatBoost
train_pool = Pool(X_train, Y_train)
val_pool   = Pool(X_val, Y_val)

cat = CatBoostRegressor(
    loss_function="RMSE", eval_metric="MAE",
    task_type="GPU", devices="0",
    depth=8, learning_rate=0.05, n_estimators=4000,
    random_seed=42, od_type="Iter", od_wait=200, verbose=200
)


cat.fit(train_pool, eval_set=val_pool, use_best_model=True)

y_pred = cat.predict(X_val)
mae  = mean_absolute_error(Y_val, y_pred)
rmse = np.sqrt(mean_squared_error(Y_val, y_pred))
r2   = r2_score(Y_val, y_pred)
print(f"[CatBoost] MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.3f}")

# save model + scaling stats
cat.save_model("catboost_tm.cbm")
joblib.dump({
    "n_bits": N_BITS,
    "radius": Radius,
    "desc_mean": desc_mean,
    "desc_std": desc_std,
    "counts_mean": counts_mean,
    "counts_std": counts_std
}, "catboost_bundle.joblib")

# --- TEST features (same transforms)
bundle = joblib.load("catboost_bundle.joblib")
desc_mean, desc_std = bundle["desc_mean"], bundle["desc_std"]
counts_mean, counts_std = bundle["counts_mean"], bundle["counts_std"]

ids         = test_df["id"].to_numpy() if "id" in test_df.columns else np.arange(len(test_df))
bits_test   = np.vstack([morgan_fingerprint(s) for s in test_df["SMILES"]]).astype(np.float32)
counts_test = np.vstack([morgan_counts(s) for s in test_df["SMILES"]]).astype(np.float32)
counts_test = np.log1p(counts_test).astype(np.float32)
desc_test   = np.vstack([chem_features(s) for s in test_df["SMILES"]]).astype(np.float32)

desc_test_s   = (desc_test   - desc_mean)   / desc_std
counts_test_s = (counts_test - counts_mean) / counts_std

X_test_final = np.hstack([bits_test, counts_test_s, desc_test_s]).astype(np.float32)

test_preds = cat.predict(X_test_final)
pd.DataFrame({"id": ids, "Tm": test_preds}).to_csv("Tm_predictions_cat.csv", index=False)
print("Wrote: Tm_predictions_cat.csv")
