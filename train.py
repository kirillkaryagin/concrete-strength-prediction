"""Train the frozen selected model and evaluate on the held-out recipe groups."""

import argparse
import json
import pickle
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from project import FEATURES, ROOT, SEED, TARGET, candidates, load_data, scores, sha256, split_data


def train_final(data=None, selection=None, output=None, report=None):
    data = Path(data) if data else ROOT / "data/concrete.csv"
    selection = Path(selection) if selection else ROOT / "results/model_selection.json"
    output = Path(output) if output else ROOT / "model.pkl"
    report = Path(report) if report else ROOT / "results/final_metrics.json"
    choice = json.loads(selection.read_text())
    df = load_data(data)
    parts, groups = split_data(df)
    trainval = sorted(set(parts["train"]) | set(parts["validation"]))
    model = candidates()[choice["selected_model"]][0].set_params(**choice["selected_params"])
    model.fit(df.loc[trainval, FEATURES], df.loc[trainval, TARGET])
    prediction = model.predict(df.loc[parts["test"], FEATURES])
    metrics = scores(df.loc[parts["test"], TARGET], prediction)
    bounds = {c: {"min": float(df.loc[trainval, c].min()),
                  "max": float(df.loc[trainval, c].max())} for c in FEATURES}
    packages = {name: version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn"]}
    metadata = {
        "selected_model": choice["selected_model"],
        "selected_params": choice["selected_params"],
        "features": FEATURES, "target": TARGET, "unit": "MPa", "seed": SEED,
        "python": platform.python_version(), "packages": packages,
        "data_sha256": sha256(data), "selection_sha256": sha256(selection),
        "training_ranges": bounds, "test_metrics": metrics,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump({"model": model, "metadata": metadata}, stream, protocol=5)
    result = {
        **metadata, "model_sha256": sha256(output), "clean_rows": len(df),
        "recipe_groups": int(len(set(groups))),
        "split_rows": {name: len(index) for name, index in parts.items()},
        "split_recipe_groups": {name: len(set(groups[index])) for name, index in parts.items()},
        "recipe_overlap": 0, "refit_rows": len(trainval),
        "baseline_test": scores(df.loc[parts["test"], TARGET],
                                [float(df.loc[trainval, TARGET].mean())] * len(parts["test"])),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    pd.DataFrame({"row_id": parts["test"], "actual_mpa": df.loc[parts["test"], TARGET].values,
                  "predicted_mpa": prediction}).to_csv(report.with_name("test_predictions.csv"), index=False)
    labels = pd.Series(index=df.index, dtype="object")
    for name, index in parts.items():
        labels.loc[index] = name
    pd.DataFrame({"row_id": df.index, "recipe_group": groups, "split": labels}).to_csv(
        report.with_name("split_manifest.csv"), index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    print(json.dumps(train_final(**vars(args)), indent=2))
