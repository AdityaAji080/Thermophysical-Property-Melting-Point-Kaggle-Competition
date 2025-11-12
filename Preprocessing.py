import pandas as pd
import numpy as np

# --- Load ---
train2_raw = pd.read_csv("BradleyMeltingPointDataset.csv")   # may already be canonicalized (SMILES, Tm)
train_raw  = pd.read_csv("train.csv")     # likely raw (smiles, mpC)
test_df    = pd.read_csv("test.csv")

def norm_smiles(s):
    return str(s).strip()

def detect_smiles_col(df):
    for c in ["SMILES", "smiles", "Smiles"]:
        if c in df.columns:
            return c
    raise KeyError("No SMILES/smiles column found.")

def detect_temp_col(df):
    """
    Return (col_name, unit) where unit in {'K','C'}.
    Prefers Kelvin if both present; otherwise detects common names.
    """
    # Kelvin-like
    for c in ["Tm", "tm", "mpK", "Tm_K", "tempK"]:
        if c in df.columns:
            return c, "K"
    # Celsius-like
    for c in ["mpC", "mpc", "Tm_C", "tempC", "melting_point_C"]:
        if c in df.columns:
            return c, "C"
    # If neither exists, allow datasets that only have SMILES (e.g., test set)
    return None, None

def to_smiles_tm(df):
    """Return a DataFrame with columns ['SMILES','Tm'] (Kelvin)."""
    smiles_col = detect_smiles_col(df)
    temp_col, unit = detect_temp_col(df)

    out = pd.DataFrame()
    out["SMILES"] = df[smiles_col].astype(str).map(norm_smiles)

    if temp_col is None:
        # No temperature column (OK for test sets, but not for train rows)
        out["Tm"] = pd.NA
    else:
        if unit == "K":
            out["Tm"] = pd.to_numeric(df[temp_col], errors="coerce")
        else:  # Celsius -> Kelvin
            out["Tm"] = pd.to_numeric(df[temp_col], errors="coerce") + 273.15

    return out[["SMILES", "Tm"]]

# Canonicalize both training sources
df2 = to_smiles_tm(train2_raw)  # from train_2.csv (likely already SMILES/Tm)
df1 = to_smiles_tm(train_raw)   # from train.csv (likely smiles/mpC)

# Exclude anything that appears in test set
test_smiles_col = "SMILES" if "SMILES" in test_df.columns else ("smiles" if "smiles" in test_df.columns else None)
if test_smiles_col is not None:
    test_smiles = set(test_df[test_smiles_col].astype(str).map(norm_smiles))
else:
    test_smiles = set()

# Drop rows with missing SMILES
df2 = df2.dropna(subset=["SMILES"])
df1 = df1.dropna(subset=["SMILES"])

# Remove overlaps with test from df2
before2 = len(df2)
df2 = df2[~df2["SMILES"].isin(test_smiles)].copy()
print(f"Removed {before2 - len(df2)} overlap rows from train_2 vs test")

# Keep only candidates from train.csv not already present in train_2.csv, and not in test
existing = set(df2["SMILES"])
candidates = df1[~df1["SMILES"].isin(existing)]
candidates = candidates[~candidates["SMILES"].isin(test_smiles)].copy()

# It’s possible df1 had NaN Tm (shouldn’t, but be safe); drop those
candidates = candidates.dropna(subset=["Tm"])

# Append and deduplicate
df_merged = pd.concat([df2, candidates], ignore_index=True)
df_merged = df_merged.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)

print(f"Added {len(df_merged) - len(df2)} new rows from train.csv into train_2")
print("Final shape:", df_merged.shape)

# Save back (overwrite train_2.csv)
df_merged.to_csv("train_2.csv", index=False)

