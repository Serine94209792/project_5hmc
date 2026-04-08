import os
import shutil
import itertools
import numpy as np
import pandas as pd
import optuna
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
from typing import Union, Callable, List, Tuple, Dict
import traceback

DEFAULT_DATATYPES = [
    "consensus_peak",
    "OCR",
    "gene_counts",
    "window",
]


def infer_datatypes(df: pd.DataFrame) -> List[str]:
    """Infer datatypes from columns like {datatype}_score_{r} or {datatype}_pseudotime."""
    import re
    score_cols = [c for c in df.columns if re.match(r".+_score_\d+$", str(c))]
    pseudo_cols = [c for c in df.columns if str(c).endswith("_pseudotime")]
    dts = {c.rsplit("_score_", 1)[0] for c in score_cols}
    dts.update({c[: -len("_pseudotime")] for c in pseudo_cols})
    return sorted(dts)


def get_score_columns(df: pd.DataFrame, datatypes: List[str]) -> List[str]:
    """Return list of {datatype}_score_{r} columns that exist in df for given datatypes (sorted by r)."""
    cols = []
    for dt in datatypes:
        prefix = f"{dt}_score_"
        dt_cols = [c for c in df.columns if str(c).startswith(prefix)]
        cols.extend(sorted(dt_cols, key=lambda x: int(str(x).rsplit("_score_", 1)[1])))
    return cols


def get_pseudotime_columns(df: pd.DataFrame, datatypes: List[str]) -> List[str]:
    """Return list of {datatype}_pseudotime columns that exist in df for given datatypes."""
    cols = []
    for dt in datatypes:
        col = f"{dt}_pseudotime"
        if col in df.columns:
            cols.append(col)
    return cols


def build_feature_columns(
    df: pd.DataFrame, datatypes: List[str], require_pseudotime: bool = True
) -> List[str]:
    """
    For each datatype, use {datatype}_pseudotime and all {datatype}_score_{r} columns.
    """
    cols: List[str] = []
    for dt in datatypes:
        pseudo_col = f"{dt}_pseudotime"
        if pseudo_col not in df.columns:
            if require_pseudotime:
                raise ValueError(f"missing required feature column: {pseudo_col}")
            else:
                continue
        cols.append(pseudo_col)

        prefix = f"{dt}_score_"
        score_cols = [c for c in df.columns if str(c).startswith(prefix)]
        score_cols = sorted(score_cols, key=lambda x: int(str(x).rsplit("_score_", 1)[1]))
        if not score_cols:
            raise ValueError(f"missing required score columns for datatype={dt} (prefix={prefix})")
        cols.extend(score_cols)

    return cols


def enumerate_combinations(datatypes: List[str], min_size: int = 2) -> List[Tuple[int, Tuple[str, ...]]]:
    """Enumerate datatype subsets of size >= min_size. Returns [(index, tuple of datatypes), ...]."""
    out = []
    idx = 0
    for r in range(min_size, len(datatypes) + 1):
        for subset in itertools.combinations(datatypes, r):
            out.append((idx, subset))
            idx += 1
    return out


def get_cv_score(
    model: LogisticRegression,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: Union[str, Callable] = "neg_log_loss",
    cv_splits: int = 4,
    n_jobs: int = -1,
):

    cv = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=3, random_state=42)
    cv_results = cross_validate(model, X, y, scoring=scoring, cv=cv, n_jobs=n_jobs, return_estimator=True)
    val_scores = cv_results["test_score"]
    models = cv_results["estimator"]
    return val_scores, models


def train_model(X_train, y_train, output_dir: str):
    def objective(trial):
        C = trial.suggest_float("C", 1e-3, 1e3, log=True)

        model = LogisticRegression(
            C=C,
            penalty='l2',
            random_state=42,
            class_weight='balanced',
            max_iter=2000,
        )

        val_scores, models = get_cv_score(model, X_train, y_train, scoring='neg_log_loss', cv_splits=5, n_jobs=-1)
        trial.set_user_attr("val_scores", val_scores)
        trial.set_user_attr("models", models)
        return val_scores.mean()

    study = optuna.create_study(direction="maximize")  # maximize neg_log_loss = minimize log loss
    study.optimize(objective, n_trials=30)

    val_scores = study.best_trial.user_attrs['val_scores']
    best_params = study.best_trial.params
    print(f"best neg_log_loss (CV mean): {study.best_trial.value}\n")
    print(f"best params: {best_params}\n")
    print(f"best val_scores (neg_log_loss per fold): {val_scores}\n")
    
    models = study.best_trial.user_attrs['models']
    best_idx = int(np.argmax(val_scores))
    best_model = models[best_idx]
    joblib.dump(best_model, f"{output_dir}/best_model.pkl", compress=3, protocol=4)
    joblib.dump(best_params, f"{output_dir}/best_params.pkl", compress=3, protocol=4)
    return best_model, best_params, val_scores.mean()


def test_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str,
):
    y_pred_test = model.predict(X_test)
    y_true_test = y_test
    test_accuracy = accuracy_score(y_true_test, y_pred_test)
    y_prob = model.predict_proba(X_test)
    n_classes = y_prob.shape[1]
    # AUC can fail when some classes are missing in y_test for a fold.
    test_micro_auc = float("nan")
    test_macro_auc = float("nan")
    if n_classes >= 2 and len(np.unique(y_test)) >= 2:
        try:
            test_micro_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="micro")
            test_macro_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        except ValueError:
            pass
    print(f"test accuracy: {test_accuracy}\n")
    print(f"test_micro_auc: {test_micro_auc}\ntest_macro_auc: {test_macro_auc}\n")
    return {
        "test_accuracy": test_accuracy,
        "test_micro_auc": test_micro_auc,
        "test_macro_auc": test_macro_auc,
    }


def nestcv(
    X: pd.DataFrame,
    y: pd.Series,
    outer_cv: int,
    outer_repeats: int,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)
    cv = RepeatedStratifiedKFold(n_splits=outer_cv, n_repeats=outer_repeats, random_state=42)
    results_list = []
    for fold_idx, (train_index, test_index) in enumerate(cv.split(X, y)):
        print(f"\n{'='*50}")
        print(f"第 {fold_idx + 1} 折开始...")
        print(f"{'='*50}\n")
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        np.save(os.path.join(fold_dir, "X_train.npy"), X_train.to_numpy(dtype=float))
        np.save(os.path.join(fold_dir, "X_test.npy"), X_test.to_numpy(dtype=float))
        np.save(os.path.join(fold_dir, "y_train.npy"), y_train)
        np.save(os.path.join(fold_dir, "y_test.npy"), y_test)
        np.save(os.path.join(fold_dir, "train_index.npy"), train_index)
        np.save(os.path.join(fold_dir, "test_index.npy"), test_index)
        np.save(
            os.path.join(fold_dir, "feature_names.npy"),
            np.array(X.columns.tolist(), dtype=object),
        )

        best_model, best_params, val_neg_log_loss = train_model(
            X_train, y_train, fold_dir
        )

        test_metrics = test_model(
            best_model, X_test, y_test, fold_dir
        )

        results_list.append({
            "fold": fold_idx,
            "best_val_neg_log_loss": val_neg_log_loss,
            "test_accuracy": test_metrics["test_accuracy"],
            "test_micro_auc": test_metrics["test_micro_auc"],
            "test_macro_auc": test_metrics["test_macro_auc"],
            "best_params": best_params,
        })

    results_df = pd.DataFrame(results_list)
    return results_df


def run_single_combination(
    df: pd.DataFrame,
    comb_index: int,
    comb_datatypes: Tuple[str, ...],
    output_dir: str,
    cv_folds: int,
    cv_repeats: int,
    datatypes_to_feature_cols: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, float, float]:
    """Run nestcv for one datatype subset. Uses {datatype}_pseudotime + {datatype}_score_{r} as features."""
    subset_cols: List[str] = []
    for d in comb_datatypes:
        cols = datatypes_to_feature_cols.get(d, [])
        if not cols:
            return pd.DataFrame(), float("nan"), float("nan")
        subset_cols.extend(cols)
    X = df[subset_cols].copy()
    y = df["y_test"]
    temp_dir = os.path.join(output_dir, "_search_temp", f"comb_{comb_index}")
    os.makedirs(temp_dir, exist_ok=True)
    try:
        results_df = nestcv(X, y, cv_folds, cv_repeats, temp_dir)
        mean_val = float(results_df["best_val_neg_log_loss"].mean())
        mean_test = float(results_df["test_accuracy"].mean())
        results_df.to_csv(os.path.join(temp_dir, "nestcv_results.csv"), index=False)
        return results_df, mean_val, mean_test
    except Exception:
        with open(os.path.join(temp_dir, "error.txt"), "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        return pd.DataFrame(), float("nan"), float("nan")


def main(args):
    df = pd.read_csv(args.input, sep="\t")
    required = ["y_test", "sample", "true_state"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Input must contain column: {c}")

    # Decide datatypes
    inferred = infer_datatypes(df)
    if args.datatypes:
        datatypes = [d.strip() for d in args.datatypes if d and d.strip()]
    else:
        datatypes = inferred or list(DEFAULT_DATATYPES)

    # Build datatype->feature_cols map, only keep datatypes that exist in df
    datatypes_to_feature_cols: Dict[str, List[str]] = {}
    for dt in datatypes:
        feature_cols = build_feature_columns(df, [dt], require_pseudotime=True)
        datatypes_to_feature_cols[dt] = feature_cols

    available_datatypes = sorted(datatypes_to_feature_cols.keys())
    if len(available_datatypes) < 2:
        raise ValueError(
            f"Need at least 2 datatypes with {{dt}}_pseudotime + {{dt}}_score_{{r}} features; found: {available_datatypes}"
        )

    if args.mode == "single":
        feature_cols = build_feature_columns(df, available_datatypes, require_pseudotime=True)
        X = df[feature_cols].copy()
        y = df["y_test"]
        os.makedirs(args.output_dir, exist_ok=True)
        results_df = nestcv(X, y, args.cv_folds, args.cv_repeats, args.output_dir)
        results_df.to_csv(os.path.join(args.output_dir, "nestcv_results.csv"), index=False)
        return

    # full_search
    search_temp = os.path.join(args.output_dir, "_search_temp")
    os.makedirs(search_temp, exist_ok=True)
    combinations = enumerate_combinations(available_datatypes, min_size=2)
    log_rows = []
    for comb_index, comb_datatypes in combinations:
        results_df, mean_val, mean_test = run_single_combination(
            df, comb_index, comb_datatypes,
            args.output_dir, args.cv_folds, args.cv_repeats,
            datatypes_to_feature_cols=datatypes_to_feature_cols,
        )
        log_rows.append({
            "comb_index": comb_index,
            "datatypes": "+".join(comb_datatypes),
            "mean_best_val_neg_log_loss": mean_val,
            "mean_test_accuracy": mean_test,
        })
    log_df = pd.DataFrame(log_rows)
    valid = log_df.dropna(subset=["mean_best_val_neg_log_loss"])
    if len(valid) == 0:
        shutil.rmtree(search_temp, ignore_errors=True)
        raise RuntimeError("All combinations failed (no valid mean_best_val_neg_log_loss).")
    best_row = valid.loc[valid["mean_best_val_neg_log_loss"].idxmax()]
    best_i = int(best_row["comb_index"])
    best_temp = os.path.join(search_temp, f"comb_{best_i}")
    for name in os.listdir(best_temp):
        src = os.path.join(best_temp, name)
        dst = os.path.join(args.output_dir, name)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    shutil.rmtree(search_temp, ignore_errors=True)
    log_df = log_df.sort_values("mean_best_val_neg_log_loss", ascending=False).reset_index(drop=True)
    log_df.to_csv(os.path.join(args.output_dir, "datatype_search_log.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="path to input file")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="path to output directory")
    parser.add_argument("--cv_folds", type=int, default=3, help="number of cv folds")
    parser.add_argument("--cv_repeats", type=int, default=5, help="number of cv repeats")
    parser.add_argument("--mode", type=str, choices=["single", "full_search"], default="single",
                        help="single: one run with all proba columns; full_search: exhaustive datatype combination search")
    parser.add_argument(
        "--datatypes",
        type=str,
        nargs="+",
        default=None,
        help="Optional datatype names to consider (default: infer from *_score_{r} / *_pseudotime columns).",
    )
    args = parser.parse_args()
    main(args)
