"""XGBoost Fraud Detection

Usage:
  train_ensemble.py <file_name> [<config>]
  train_ensemble.py (-h | --help)

Arguments:
  file_name   Path to the input data file (required).
  config      Path to an optional config file.

Options:
  -h, --help  Show this screen.
"""

from pathlib import Path

import pandas as pd
import yaml
from docopt import docopt
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

TARGET = "Class"
OUTPUT_DIR = Path("output")


def load_config(path: str | None) -> dict:
    if path is None:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_data(file_name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(file_name)
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    return X, y


def split(X, y, test_size: float, random_state: int):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def build_baseline() -> XGBClassifier:
    return XGBClassifier(
        eval_metric="logloss",
        tree_method="hist",
        device="cuda",
    )


def build_tuned(grid: dict, cv_folds: int) -> GridSearchCV:
    base = XGBClassifier(
        eval_metric="logloss",
        tree_method="hist",
        device="cuda",
    )
    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=433567
    )
    return GridSearchCV(
        base,
        param_grid=grid,
        cv=cv,
        scoring="recall",
        n_jobs=1
    )


def build_bagging(grid: dict, cv_folds: int, bagging_cfg: dict, n_estimators: int) -> BaggingClassifier:
    inner = build_tuned(grid, cv_folds)
    return BaggingClassifier(
        estimator=inner,
        n_estimators=n_estimators,
        max_samples=bagging_cfg.get("max_samples", 1.0),
        max_features=bagging_cfg.get("max_features", 1.0),
        random_state=42,
        n_jobs=1,
    )


def output_path(config_path: str | None, suffix: str = "") -> Path:
    stem = Path(config_path).stem if config_path else "baseline"
    return OUTPUT_DIR / f"{stem}{suffix}_xgb_pred.csv"


def fit_and_save(model, X_train, X_test, y_train, y_test, config_path: str | None, suffix: str = "") -> None:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"recall{suffix}: {recall_score(y_test, y_pred):.4f}")
    out = output_path(config_path, suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred}).to_csv(out, index=False)
    print(f"wrote {out}")


def main(file_name: str, config_path: str | None) -> None:
    cfg = load_config(config_path)
    X, y = load_data(file_name)
    X_train, X_test, y_train, y_test = split(
        X, y, cfg.get("test_size", 0.2), cfg.get("random_state", 433567)
    )

    mode = cfg.get("mode", "baseline")
    print(f"mode: {mode}")

    if mode == "baseline":
        fit_and_save(build_baseline(), X_train, X_test, y_train, y_test, config_path)
    elif mode == "tuning":
        model = build_tuned(cfg.get("grid", {}), cfg.get("cv_folds", 5))
        fit_and_save(model, X_train, X_test, y_train, y_test, config_path)
    elif mode == "bagging":
        bagging_cfg = cfg.get("bagging", {})
        n_est = bagging_cfg.get("n_estimators", 10)
        n_est_list = n_est if isinstance(n_est, list) else [n_est]
        for n in n_est_list:
            suffix = f"_n{n}" if len(n_est_list) > 1 else ""
            model = build_bagging(
                cfg.get("grid", {}), cfg.get("cv_folds", 5), bagging_cfg, n
            )
            fit_and_save(model, X_train, X_test, y_train, y_test, config_path, suffix)
    else:
        raise ValueError(f"unknown mode: {mode}")


if __name__ == "__main__":
    args = docopt(__doc__ or "")
    main(file_name=args["<file_name>"], config_path=args["<config>"])
