"""Build the training set: Kaggle train + the Bradley open melting-point dataset.

Molecules are matched by RDKit *canonical* SMILES, not raw strings. The same molecule can be
written many ways, and a raw-string check misses most of the Bradley rows that duplicate test
molecules (in practice, it lets ~490 of the 666 test molecules leak into training).

Steps:
  1. Kaggle train (Tm in Kelvin) is kept as-is; it is the label source of record.
  2. Bradley rows flagged `donotuse`, or whose SMILES RDKit cannot parse, are dropped;
     melting points are converted from °C to K.
  3. Bradley molecules that appear in the Kaggle test set or the Kaggle train set are removed.
  4. Remaining Bradley duplicates are collapsed to their median melting point.

Usage:
    python prepare_data.py            # reads data/, writes data/train_merged.csv
"""

import argparse
import os

import pandas as pd

from features import canonical_smiles


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data_dir", default="data")
    args = p.parse_args()
    path = lambda f: os.path.join(args.data_dir, f)

    kaggle = pd.read_csv(path("train.csv"))[["SMILES", "Tm"]]
    test = pd.read_csv(path("test.csv"))
    bradley = pd.read_csv(path("BradleyMeltingPointDataset.csv"))

    kaggle["canon"] = kaggle["SMILES"].map(canonical_smiles)
    test_canon = set(test["SMILES"].map(canonical_smiles)) - {None}

    n0 = len(bradley)
    bradley = bradley[bradley["donotuse"].isna()]
    bradley = pd.DataFrame({"SMILES": bradley["smiles"].astype(str).str.strip(),
                            "Tm": pd.to_numeric(bradley["mpC"], errors="coerce") + 273.15})
    bradley["canon"] = bradley["SMILES"].map(canonical_smiles)
    bradley = bradley.dropna(subset=["canon", "Tm"])
    n_clean = len(bradley)

    in_test = bradley["canon"].isin(test_canon)
    in_train = bradley["canon"].isin(set(kaggle["canon"]))
    bradley = bradley[~in_test & ~in_train]
    bradley = (bradley.groupby("canon", as_index=False)
                      .agg(SMILES=("SMILES", "first"), Tm=("Tm", "median")))

    merged = pd.concat([kaggle.assign(source="kaggle"), bradley.assign(source="bradley")],
                       ignore_index=True)[["SMILES", "canon", "Tm", "source"]]
    merged.to_csv(path("train_merged.csv"), index=False)

    print(f"Bradley: {n0} rows -> {n_clean} usable -> removed {in_test.sum()} test-molecule rows "
          f"and {in_train.sum()} Kaggle-train duplicates -> {len(bradley)} unique molecules added")
    print(f"Kaggle train/test molecules overlapping: {kaggle['canon'].isin(test_canon).sum()}")
    print(f"Wrote {path('train_merged.csv')}: {len(merged)} molecules")


if __name__ == "__main__":
    main()
