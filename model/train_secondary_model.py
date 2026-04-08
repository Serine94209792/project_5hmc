import os
import shutil
import itertools
import numpy as np
import pandas as pd
import optuna
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score
import joblib
from typing import Union, Callable, List, Tuple

# Datatype names matching integrate_model aggregate_df column names: {datatype}_proba_1
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
    """Return list of *_proba_1 columns that exist in df and match DATATYPES."""
    cols = []
    for dt in DATATYPES:
        c = f"{dt}_proba_1"
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
    scoring: Union[str, Callable] = "accuracy",
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
        )
        
        val_scores, models = get_cv_score(model, X_train, y_train, scoring='accuracy', cv_splits=5, n_jobs=-1)
        trial.set_user_attr("val_scores", val_scores)
        trial.set_user_attr("models", models)
        return val_scores.mean()
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    
    val_scores = study.best_trial.user_attrs['val_scores']
    best_params = study.best_trial.params
    print(f"best score: {study.best_trial.value}\n")  # val_scores.mean() = study.best_trial.value
    print(f"best params: {best_params}\n")
    print(f"best val_scores: {val_scores}\n")
    
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
    return test_accuracy


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
        
        best_model, best_params, val_accuracy = train_model(
            X_train, y_train, fold_dir
        )
        
        # best_model, best_params, val_accuracy = train_model(
        #     X_train, y_train, X_test, y_test, fold_dir
        # )
                
        test_accuracy = test_model(
            best_model, X_test, y_test, fold_dir
        )
        
        results_list.append({
            'fold': fold_idx,
            'best_val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'best_params': best_params,
        })
            
    results_df = pd.DataFrame(results_list)
    return results_df


def run_single_combination(
    comb_index: int,
    subset: Tuple[str, ...],
    df: pd.DataFrame,
    output_dir: str,
    cv_folds: int,
    cv_repeats: int,
) -> Tuple[pd.DataFrame, float, float]:
    """
    Run nestcv for one datatype subset. Writes to output_dir/_search_temp/comb_{comb_index}.
    Returns (results_df, mean_best_val_accuracy, mean_test_accuracy).
    On failure returns (None, np.nan, np.nan).
    """
    proba_cols = [f"{dt}_proba_1" for dt in subset]
    missing = [c for c in proba_cols if c not in df.columns]
    if missing:
        print(f"[comb {comb_index}] skip: missing columns {missing}")
        return None, np.nan, np.nan
    X = df[proba_cols].copy()
    y = df["y_test"]
    temp_dir = os.path.join(output_dir, "_search_temp", f"comb_{comb_index}")
    try:
        results_df = nestcv(X, y, cv_folds, cv_repeats, temp_dir)
        results_df.to_csv(os.path.join(temp_dir, "nestcv_results.csv"), index=False)
        mean_val = float(results_df["best_val_accuracy"].mean())
        mean_test = float(results_df["test_accuracy"].mean())
        return results_df, mean_val, mean_test
    except Exception as e:
        print(f"[comb {comb_index}] failed: {e}")
        return None, np.nan, np.nan


def main(args):
    df = pd.read_csv(args.input, sep="\t")
    for col in ["y_test", "test_index", "stage"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    proba_cols = get_proba_columns(df)
    if len(proba_cols) < 2:
        raise ValueError(
            f"Need at least 2 datatype proba columns; found {proba_cols}. "
            "Check that aggregate_df has columns like artemis_proba_1, ..."
        )
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "single":
        X = df[proba_cols].copy()
        y = df["y_test"]
        results_df = nestcv(X, y, args.cv_folds, args.cv_repeats, args.output_dir)
        results_df.to_csv(os.path.join(args.output_dir, "nestcv_results.csv"), index=False)
        return

    if args.mode == "full_search":
        combinations = enumerate_combinations(min_size=2)
        if len(combinations) == 0:
            raise ValueError("No datatype combinations to search")
        search_temp = os.path.join(args.output_dir, "_search_temp")
        os.makedirs(search_temp, exist_ok=True)
        records = []
        for i, subset in combinations:
            _, mean_val, mean_test = run_single_combination(
                i, subset, df, args.output_dir, args.cv_folds, args.cv_repeats
            )
            records.append({
                "combination_index": i,
                "selected_datatypes": ",".join(subset),
                "mean_best_val_accuracy": mean_val,
                "mean_test_accuracy": mean_test,
            })
            print(f"[{i + 1}/{len(combinations)}] datatypes={subset} mean_best_val_accuracy={mean_val:.4f} mean_test_accuracy={mean_test:.4f}")

        log_df = pd.DataFrame(records)
        log_df = log_df.sort_values("mean_best_val_accuracy", ascending=False).reset_index(drop=True)
        valid = log_df.dropna(subset=["mean_best_val_accuracy"])
        if len(valid) == 0:
            shutil.rmtree(search_temp)
            raise RuntimeError("All 120 combinations failed; no valid mean_best_val_accuracy.")
        best_i = int(valid.iloc[0]["combination_index"])
        best_subset = valid.iloc[0]["selected_datatypes"]
        best_temp = os.path.join(search_temp, f"comb_{best_i}")
        if not os.path.isdir(best_temp):
            raise RuntimeError(f"Best combination {best_i} temp dir missing: {best_temp}")
        for name in os.listdir(best_temp):
            src = os.path.join(best_temp, name)
            dst = os.path.join(args.output_dir, name)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        shutil.rmtree(search_temp)
        log_df.to_csv(os.path.join(args.output_dir, "datatype_search_log.csv"), index=False)
        print(f"Best combination (mean_best_val_accuracy): index={best_i} datatypes={best_subset}")
        return

    raise ValueError(f"Unknown mode: {args.mode}. Use 'single' or 'full_search'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="path to input file")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="path to output directory")
    parser.add_argument("--cv_folds", type=int, default=3, help="number of cv folds")
    parser.add_argument("--cv_repeats", type=int, default=3, help="number of cv repeats")
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "full_search"],
        help="single: one nestcv on all proba columns; full_search: try 120 datatype subsets, keep best",
    )
    args = parser.parse_args()
    main(args)