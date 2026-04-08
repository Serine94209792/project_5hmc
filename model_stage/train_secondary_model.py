import os
import shutil
import itertools
import numpy as np
import pandas as pd
import optuna
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import joblib
from typing import Union, Callable, List, Tuple

DATATYPES = [
    "artemis",
    "consensus_peak",
    "end_motif",
    "FSD",
    "OCR",
    "gene_counts",
    "window",
]


def get_proba_columns(df: pd.DataFrame) -> List[str]:
    """Return list of *_proba_0, *_proba_1, *_proba_2 columns that exist in df for each DATATYPES."""
    cols = []
    for dt in DATATYPES:
        for k in (0, 1, 2):
            c = f"{dt}_proba_{k}"
            if c in df.columns:
                cols.append(c)
    return cols


def enumerate_combinations(min_size: int = 2) -> List[Tuple[int, Tuple[str, ...]]]:
    """Enumerate datatype subsets of size >= min_size. Returns [(index, tuple of datatypes), ...]."""
    out = []
    idx = 0
    for r in range(min_size, len(DATATYPES) + 1):
        for subset in itertools.combinations(DATATYPES, r):
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
    test_log_loss = log_loss(y_test, y_prob)
    n_classes = y_prob.shape[1]
    if n_classes >= 2:
        test_micro_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="micro")
        test_macro_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    else:
        test_micro_auc = test_macro_auc = float("nan")
    print(f"test accuracy: {test_accuracy}\n")
    print(f"test log_loss: {test_log_loss}\n")
    print(f"test_micro_auc: {test_micro_auc}\ntest_macro_auc: {test_macro_auc}\n")
    return {
        "test_accuracy": test_accuracy,
        "test_log_loss": test_log_loss,
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
        
        np.save(os.path.join(fold_dir, "X_train.npy"), X_train)
        np.save(os.path.join(fold_dir, "X_test.npy"), X_test)
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
        
        # best_model, best_params, val_neg_log_loss = train_model(
        #     X_train, y_train, X_test, y_test, fold_dir
        # )
                
        test_metrics = test_model(
            best_model, X_test, y_test, fold_dir
        )
        
        results_list.append({
            "fold": fold_idx,
            "best_val_neg_log_loss": val_neg_log_loss,
            "test_accuracy": test_metrics["test_accuracy"],
            "test_log_loss": test_metrics["test_log_loss"],
            "test_micro_auc": test_metrics["test_micro_auc"],
            "test_macro_auc": test_metrics["test_macro_auc"],
            "best_params": best_params,
        })
            
    results_df = pd.DataFrame(results_list)
    return results_df


def run_single_combination(
    df: pd.DataFrame,
    proba_cols: List[str],
    comb_index: int,
    comb_datatypes: Tuple[str, ...],
    output_dir: str,
    cv_folds: int,
    cv_repeats: int,
) -> Tuple[pd.DataFrame, float, float, float]:
    """Run nestcv for one datatype subset. Uses proba_0, proba_1, proba_2 per datatype. Writes to output_dir/_search_temp/comb_{comb_index}. Returns (results_df, mean_val_neg_log_loss, mean_test_accuracy, mean_test_log_loss)."""
    subset_cols = []
    for d in comb_datatypes:
        for k in (0, 1, 2):
            c = f"{d}_proba_{k}"
            if c not in proba_cols:
                return pd.DataFrame(), float("nan"), float("nan"), float("nan")
            subset_cols.append(c)
    X = df[subset_cols].copy()
    y = df["y_test"]
    temp_dir = os.path.join(output_dir, "_search_temp", f"comb_{comb_index}")
    os.makedirs(temp_dir, exist_ok=True)
    try:
        results_df = nestcv(X, y, cv_folds, cv_repeats, temp_dir)
        mean_val = float(results_df["best_val_neg_log_loss"].mean())
        mean_test = float(results_df["test_accuracy"].mean())
        mean_test_log_loss = float(results_df["test_log_loss"].mean())
        results_df.to_csv(os.path.join(temp_dir, "nestcv_results.csv"), index=False)
        return results_df, mean_val, mean_test, mean_test_log_loss
    except Exception:
        return pd.DataFrame(), float("nan"), float("nan"), float("nan")


def main(args):
    df = pd.read_csv(args.input, sep="\t")
    required = ["y_test", "test_index"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Input must contain column: {c}")
    proba_cols = get_proba_columns(df)
    available_datatypes = [dt for dt in DATATYPES if all(f"{dt}_proba_{k}" in df.columns for k in (0, 1, 2))]
    if len(available_datatypes) < 2:
        raise ValueError(
            f"Need at least 2 datatypes with proba_0, proba_1, proba_2; found: {available_datatypes}"
        )

    if args.mode == "single":
        X = df[proba_cols].copy()
        y = df["y_test"]
        os.makedirs(args.output_dir, exist_ok=True)
        results_df = nestcv(X, y, args.cv_folds, args.cv_repeats, args.output_dir)
        results_df.to_csv(os.path.join(args.output_dir, "nestcv_results.csv"), index=False)
        return

    # full_search
    search_temp = os.path.join(args.output_dir, "_search_temp")
    os.makedirs(search_temp, exist_ok=True)
    combinations = enumerate_combinations(min_size=2)
    log_rows = []
    for comb_index, comb_datatypes in combinations:
        results_df, mean_val, mean_test, mean_test_log_loss = run_single_combination(
            df, proba_cols, comb_index, comb_datatypes,
            args.output_dir, args.cv_folds, args.cv_repeats,
        )
        log_rows.append({
            "comb_index": comb_index,
            "datatypes": "+".join(comb_datatypes),
            "mean_best_val_neg_log_loss": mean_val,
            "mean_test_accuracy": mean_test,
            "mean_test_log_loss": mean_test_log_loss,
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
    args = parser.parse_args()
    main(args)