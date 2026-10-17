"""Train a melting-point model, report hold-out metrics, and write a Kaggle submission.

Models:
    catboost   gradient-boosted trees on Morgan bits + counts + descriptors
    mlp        3-layer tanh MLP (512-256-128) on the same features
    mlp_maccs  the MLP with MACCS keys added

The hold-out split (default 20%) is stratified by data source, and feature scaling is fitted on
the training part only. Metrics are reported on the whole hold-out and on its Kaggle-train rows,
which are the most comparable to the competition's test set.

Usage:
    python prepare_data.py
    python train.py --model catboost            # add --device gpu to use a CUDA GPU
    python train.py --model mlp
    python train.py --model mlp_maccs
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from features import BlockScaler, featurize, take

MODELS = ("catboost", "mlp", "mlp_maccs")


def build_model(name, device, seed):
    if name == "catboost":
        from catboost import CatBoostRegressor
        kw = dict(task_type="GPU", devices="0") if device == "gpu" else dict(thread_count=-1)
        return CatBoostRegressor(loss_function="RMSE", eval_metric="MAE", depth=8, learning_rate=0.05,
                                 n_estimators=4000, od_type="Iter", od_wait=200, random_seed=seed,
                                 verbose=500, **kw)
    common = dict(hidden_layer_sizes=(512, 256, 128), activation="tanh", solver="adam",
                  learning_rate_init=1e-4, max_iter=2000, early_stopping=True,
                  n_iter_no_change=100, random_state=seed)
    if name == "mlp":
        mlp = MLPRegressor(alpha=1e-4, learning_rate="adaptive", **common)
    else:  # mlp_maccs
        mlp = MLPRegressor(alpha=1e-3, validation_fraction=0.15, **common)
    # Standardise the target: regressing raw kelvin (~300) through tanh layers can collapse to a
    # constant prediction at the mean, which early stopping then locks in.
    return TransformedTargetRegressor(regressor=mlp, transformer=StandardScaler())


def metrics(y, pred):
    return dict(mae=float(mean_absolute_error(y, pred)),
                rmse=float(np.sqrt(mean_squared_error(y, pred))),
                r2=float(r2_score(y, pred)), n=int(len(y)))


def main():
    p = argparse.ArgumentParser(description="Melting-point regression on molecular fingerprints.")
    p.add_argument("--model", choices=MODELS, default="catboost")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--device", choices=("cpu", "gpu"), default="cpu", help="CatBoost only.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(args.data_dir, "train_merged.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    use_maccs = args.model == "mlp_maccs"

    t0 = time.time()
    blocks = featurize(train_df["SMILES"], use_maccs)
    test_blocks = featurize(test_df["SMILES"], use_maccs)
    y = train_df["Tm"].to_numpy(np.float32)
    print(f"Featurised {len(train_df)} train + {len(test_df)} test molecules in {time.time() - t0:.0f}s")

    idx_tr, idx_va = train_test_split(np.arange(len(y)), test_size=args.val_size,
                                      random_state=args.seed, stratify=train_df["source"])
    scaler = BlockScaler().fit(take(blocks, idx_tr))
    X_tr, X_va = scaler.transform(take(blocks, idx_tr)), scaler.transform(take(blocks, idx_va))

    model = build_model(args.model, args.device, args.seed)
    t0 = time.time()
    if args.model == "catboost":
        model.fit(X_tr, y[idx_tr], eval_set=(X_va, y[idx_va]), use_best_model=True)
    else:
        model.fit(X_tr, y[idx_tr])
    train_time = time.time() - t0

    pred = model.predict(X_va)
    kaggle_rows = (train_df["source"].to_numpy()[idx_va] == "kaggle")
    results = dict(model=args.model, n_features=int(X_tr.shape[1]), train_seconds=round(train_time),
                   holdout_all=metrics(y[idx_va], pred),
                   holdout_kaggle=metrics(y[idx_va][kaggle_rows], pred[kaggle_rows]),
                   baseline_mean_mae=float(mean_absolute_error(y[idx_va], np.full_like(pred, y[idx_tr].mean()))))
    print(json.dumps(results, indent=2))
    with open(os.path.join(args.out_dir, f"metrics_{args.model}.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Note: the submission uses the model trained on the training split only. Refit on all data
    # before a final submission if you want every labelled molecule to count.
    test_pred = model.predict(scaler.transform(test_blocks))
    sub = os.path.join(args.out_dir, f"submission_{args.model}.csv")
    pd.DataFrame({"id": test_df["id"], "Tm": test_pred}).to_csv(sub, index=False)
    print(f"Wrote {sub}")


if __name__ == "__main__":
    main()
