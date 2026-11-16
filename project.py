"""Shared data contract, recipe-aware partitions and candidate models."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent
FEATURES = [
    "cement", "slag", "fly_ash", "water", "superplasticizer",
    "coarse_aggregate", "fine_aggregate", "age",
]
TARGET = "strength_mpa"
SEED = 42


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(path=None):
    path = Path(path) if path else ROOT / "data" / "concrete.csv"
    source = ROOT / "data" / "source.json"
    if source.exists():
        expected = json.loads(source.read_text())["normalized_csv_sha256"]
        if sha256(path) != expected:
            raise ValueError("Dataset checksum differs from the documented source.")
    raw = pd.read_csv(path)
    if list(raw.columns) != FEATURES + [TARGET]:
        raise ValueError("Unexpected dataset columns or column order.")
    if not np.isfinite(raw.to_numpy(dtype=float)).all():
        raise ValueError("Missing or non-finite dataset value.")
    if (raw[FEATURES] < 0).any().any() or (raw[TARGET] <= 0).any():
        raise ValueError("Invalid negative input or nonpositive target.")
    return raw.drop_duplicates().reset_index(drop=True)


def split_data(df):
    # All seven ingredient quantities define a recipe; age is deliberately excluded.
    groups = pd.factorize(pd.MultiIndex.from_frame(df[FEATURES[:-1]]))[0]
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    full_train, test = next(outer.split(df, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_local, val_local = next(inner.split(df.iloc[full_train], groups=groups[full_train]))
    parts = {"train": full_train[train_local], "validation": full_train[val_local], "test": test}
    group_sets = [set(groups[parts[name]]) for name in parts]
    assert all(group_sets[i].isdisjoint(group_sets[j]) for i in range(3) for j in range(i))
    assert len(set(np.concatenate(list(parts.values())))) == len(df)
    return parts, groups


def candidates():
    return {
        "ridge": (make_pipeline(StandardScaler(), Ridge()),
                  {"ridge__alpha": [0.1, 1.0, 10.0, 100.0]}),
        "decision_tree": (DecisionTreeRegressor(random_state=SEED),
                          {"max_depth": [3, 5, 8, None], "min_samples_leaf": [2, 5]}),
        "random_forest": (RandomForestRegressor(n_estimators=150, random_state=SEED, n_jobs=1),
                          {"max_depth": [12, None], "min_samples_leaf": [1, 2, 4]}),
    }


def scores(y, prediction):
    return {
        "rmse_mpa": float(root_mean_squared_error(y, prediction)),
        "mae_mpa": float(mean_absolute_error(y, prediction)),
        "r2": float(r2_score(y, prediction)),
    }
