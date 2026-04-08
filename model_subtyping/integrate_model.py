import os
import numpy as np
import pandas as pd
from typing import Optional
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from pyHSICLasso import HSICLasso
import torch
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import StratifiedShuffleSplit, RepeatedStratifiedKFold
import random

# Class colors shared by plot_proba_boxplot and plot_multi_feature_boxplot (class 0, 1, 2)
CLASS_COLORS = ["#348ABD", "#E24A33", "#2ecc71"]


class HSICLassoTransformer(HSICLasso, BaseEstimator, TransformerMixin):
    """
    Multiple inheritance: sklearn Transformer + HSICLasso.
    Enables fit/transform in Pipeline while still using HSICLasso methods (dump, get_index, plot_path, etc.).
    """

    def __init__(self,
                 mode="classification",  # or "regression"
                 num_feat=20,
                 B=0,
                 M=1,
                 discrete_x=False,
                 max_neighbors=5,
                 n_jobs=-1,
                 covars=np.array([]),
                 covars_kernel="Gaussian"):
        """
        Expose main HSICLasso parameters for HSICLassoTransformer instantiation.
        """
        # Initialize parent HSICLasso first
        HSICLasso.__init__(self)

        # Wrapper attributes
        self.mode = mode
        self.num_feat = num_feat
        self.B = B
        self.M = M
        self.discrete_x = discrete_x
        self.max_neighbors = max_neighbors
        self.n_jobs = n_jobs
        self.covars = covars
        self.covars_kernel = covars_kernel
        self.feature_importances_ = None    # feature importance from input
        self.selected_idx_ = None  # selected column indices for transform()
        self.n_features_ = None  # number of features at fit time

    def fit(self, X, y=None):
        """
        Sklearn-style fit: run classification or regression, record selected indices and feature importances.
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape
        
        self.n_features_ = n_features

        if y is None:
            raise ValueError("HSICLasso requires supervised target y.")
        y = np.asarray(y).ravel()

        X_in, Y_in = X, y
        self.input(X_in, Y_in)

        if self.mode == "classification":
            self.classification(
                num_feat=self.num_feat, B=self.B, M=self.M,
                discrete_x=self.discrete_x, max_neighbors=self.max_neighbors,
                n_jobs=self.n_jobs, covars=self.covars, covars_kernel=self.covars_kernel
            )
        elif self.mode == "regression":
            self.regression(
                num_feat=self.num_feat, B=self.B, M=self.M,
                discrete_x=self.discrete_x, max_neighbors=self.max_neighbors,
                n_jobs=self.n_jobs, covars=self.covars, covars_kernel=self.covars_kernel
            )
        else:
            raise ValueError("mode must be 'classification' or 'regression'.")

        # Record selected column indices (HSICLasso stores them in self.A)
        if not hasattr(self, 'A') or self.A is None:
            raise RuntimeError("HSICLasso did not select features; self.A is missing.")
        
        self.selected_idx_ = self.A

        # Compute and store feature_importances_ (beta-based, same idea as HSICLasso dump())
        feature_importances = np.zeros(n_features)
        if hasattr(self, 'A') and len(self.A) > 0 and hasattr(self, 'beta') and self.beta is not None:
            maxval = float(self.beta[self.A[0]][0])  # first selected feature has largest beta
            
            if maxval != 0:
                for idx in self.A:
                    feature_importances[idx] = float(self.beta[idx][0]) / maxval
            else:
                for idx in self.A:
                    feature_importances[idx] = 1.0

        self.feature_importances_ = feature_importances

        return self

    def transform(self, X):
        """
        Return only the columns selected by HSICLasso.
        """
        if self.selected_idx_ is None:
            raise RuntimeError("Call fit() before transform().")
        
        if len(self.selected_idx_) == 0:
            raise ValueError("No features selected; cannot transform.")

        X = np.asarray(X)
        return X[:, self.selected_idx_]
    
    def fit_transform(self, X, y=None):
        """
        Fit then transform.
        """
        return self.fit(X, y).transform(X)

    def get_support(self, indices: bool = False):
        """
        indices=True: return index array (e.g. [0, 2, 5, ...]).
        indices=False: return boolean array of length n_features_, True for selected.
        """
        if self.selected_idx_ is None:
            raise ValueError("HSICLassoTransformer not fitted; cannot call get_support.")
        
        if self.n_features_ is None:
            raise ValueError("n_features_ not set; call fit() first.")

        if indices:
            return self.selected_idx_
        else:
            mask = np.zeros(self.n_features_, dtype=bool)
            mask[self.selected_idx_] = True
            return mask


class _FCN(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers: List[torch.nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(torch.nn.Linear(prev, h))
            layers.append(torch.nn.ReLU())
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        self.net = torch.nn.Sequential(*layers)
        self.output_activation = torch.nn.Softplus()

    def forward(self, x):
        return self.output_activation(self.net(x)).squeeze(-1)


class FCNRegressor(BaseEstimator, RegressorMixin):
    """sklearn-compatible FCN regressor optimized with Pearson + Logistic ranking loss."""

    def __init__(
        self,
        network_shape: List[int] | Tuple[int, ...] = (16, 16),
        learning_rate: float = 1e-3,
        early_stopping_rounds: int = 10,
        n_epochs: int = 30,
        coef_pearson: float = 1.0,
        coef_rank: float = 1.0,
        validation_strategy: str = "single_split",
        validation_size: float = 0.2,
        cv_splits: int = 5,
        cv_repeats: int = 1,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.network_shape = list(network_shape)
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.n_epochs = n_epochs
        self.coef_pearson = coef_pearson
        self.coef_rank = coef_rank
        self.validation_strategy = validation_strategy
        self.validation_size = validation_size
        self.cv_splits = cv_splits
        self.cv_repeats = cv_repeats
        self.random_state = random_state
        self.verbose = verbose
        self.model_: _FCN | None = None
        self.best_epoch_: int | None = None
        self.validation_total_loss_: float | None = None
        self.validation_pearson_: float | None = None
        self.validation_rank_loss_: float | None = None
        self.cv_mean_total_loss_: float | None = None
        self.cv_mean_pearson_: float | None = None
        self.cv_mean_rank_loss_: float | None = None

    def _set_seed(self):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

    @staticmethod
    def _pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        pred_c = pred - pred.mean()
        target_c = target - target.mean()
        denom = torch.sqrt(torch.sum(pred_c**2) + eps) * torch.sqrt(torch.sum(target_c**2) + eps)
        r = torch.sum(pred_c * target_c) / denom
        return (1.0 - r) / 2.0

    @staticmethod
    def _logistic_rank_loss(pred: torch.Tensor, stage: torch.Tensor) -> torch.Tensor:
        # rank constraints: stage0 < stage1 and stage1 < stage2
        losses = []
        for s_low, s_high in [(0, 1), (1, 2)]:
            low = pred[stage == s_low]
            high = pred[stage == s_high]
            if low.numel() == 0 or high.numel() == 0:
                continue
            delta = high[:, None] - low[None, :]
            losses.append(torch.nn.functional.softplus(-delta).mean())
        if not losses:
            return torch.tensor(0.0, dtype=pred.dtype, device=pred.device)
        return torch.stack(losses).mean()

    def fit(
        self,
        X,
        y,
        stage_labels=None,
    ):
        if float(self.coef_pearson) < 0 or float(self.coef_rank) < 0:
            raise ValueError("coef_pearson and coef_rank must be >= 0.")
        if float(self.coef_pearson) == 0.0 and float(self.coef_rank) == 0.0:
            raise ValueError("coef_pearson and coef_rank cannot both be 0.")

        if stage_labels is None:
            raise ValueError("FCNRegressor.fit requires stage_labels for ranking loss.")
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).ravel()
        s_np = np.asarray(stage_labels, dtype=np.int64).ravel()
        if X_np.shape[0] != y_np.shape[0] or X_np.shape[0] != s_np.shape[0]:
            raise ValueError("X, y, stage_labels must have same number of rows.")

        self._set_seed()

        n_features = X_np.shape[1]
        if len(self.network_shape) == 0:
            raise ValueError("network_shape cannot be empty.")
        # Accept both [n_features, ...] and hidden-only shape.
        if self.network_shape[0] == n_features:
            hidden_dims = list(self.network_shape[1:])
        else:
            hidden_dims = list(self.network_shape)
        if len(hidden_dims) == 0:
            hidden_dims = [16]

        strategy = str(self.validation_strategy).strip().lower()
        split_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        if strategy == "single_split":
            if not (0.0 < float(self.validation_size) < 1.0):
                raise ValueError("validation_size must be in (0, 1) when validation_strategy=single_split.")
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=float(self.validation_size),
                random_state=self.random_state,
            )
            split_pairs = list(splitter.split(X_np, s_np))
        elif strategy == "stratified_cv":
            if int(self.cv_splits) < 2:
                raise ValueError("cv_splits must be >= 2 when validation_strategy=stratified_cv.")
            splitter = RepeatedStratifiedKFold(
                n_splits=int(self.cv_splits),
                n_repeats=max(1, int(self.cv_repeats)),
                random_state=self.random_state,
            )
            split_pairs = list(splitter.split(X_np, s_np))
        else:
            raise ValueError("validation_strategy must be one of {'single_split', 'stratified_cv'}.")

        if len(split_pairs) == 0:
            raise RuntimeError("No validation split generated.")

        split_results: List[Dict[str, Any]] = []
        for tr_idx, val_idx in split_pairs:
            X_tr = torch.from_numpy(X_np[tr_idx])
            y_tr = torch.from_numpy(y_np[tr_idx])
            s_tr = torch.from_numpy(s_np[tr_idx])
            X_val_t = torch.from_numpy(X_np[val_idx])
            y_val_t = torch.from_numpy(y_np[val_idx])
            s_val_t = torch.from_numpy(s_np[val_idx])

            model = _FCN(input_dim=n_features, hidden_dims=hidden_dims)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

            best_state = None
            best_total_loss = float("inf")
            best_loss_reg = float("inf")
            best_loss_rank = float("inf")
            best_epoch = 0
            no_improve = 0

            for epoch in range(self.n_epochs):
                model.train()
                optimizer.zero_grad()
                pred_tr = model(X_tr)
                loss_reg_tr = self._pearson_loss(pred_tr, y_tr)
                loss_rank_tr = self._logistic_rank_loss(pred_tr, s_tr)
                loss_tr = self.coef_pearson * loss_reg_tr + self.coef_rank * loss_rank_tr
                loss_tr.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    pred_val = model(X_val_t)
                    loss_reg_val = self._pearson_loss(pred_val, y_val_t)
                    loss_rank_val = self._logistic_rank_loss(pred_val, s_val_t)
                    loss_val = float((self.coef_pearson * loss_reg_val + self.coef_rank * loss_rank_val).item())

                if loss_val < best_total_loss - 1e-7:
                    best_total_loss = loss_val
                    best_loss_reg = float(loss_reg_val.item())
                    best_loss_rank = float(loss_rank_val.item())
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = epoch
                    no_improve = 0
                else:
                    no_improve += 1

                if self.verbose:
                    print(
                        f"[FCN] epoch={epoch + 1} "
                        f"train={loss_tr.item():.6f} val={loss_val:.6f}"
                    )
                if no_improve >= self.early_stopping_rounds:
                    break

            if best_state is not None:
                model.load_state_dict(best_state)

            split_results.append(
                {
                    "model": model,
                    "best_epoch": int(best_epoch),
                    "total_loss": float(best_total_loss),
                    "loss_reg": float(best_loss_reg),
                    "loss_rank": float(best_loss_rank),
                    "pearson": float(1.0 - 2.0 * best_loss_reg),
                }
            )

        best_idx = int(np.argmin([r["total_loss"] for r in split_results]))
        best_result = split_results[best_idx]
        self.model_ = best_result["model"]
        self.best_epoch_ = int(best_result["best_epoch"])
        self.validation_total_loss_ = float(best_result["total_loss"])
        self.validation_pearson_ = float(best_result["pearson"])
        self.validation_rank_loss_ = float(best_result["loss_rank"])
        self.cv_mean_total_loss_ = float(np.mean([r["total_loss"] for r in split_results]))
        self.cv_mean_pearson_ = float(np.mean([r["pearson"] for r in split_results]))
        self.cv_mean_rank_loss_ = float(np.mean([r["loss_rank"] for r in split_results]))
        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("Call fit before predict.")
        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(X_t).cpu().numpy()
        return pred


def fold_probs(
    datatype_dir: str,
    repeat: int,
    cv_folds: Optional[int] = None,
    sample_type: str = "test",
):
    """
    Load and aggregate repeated fold prediction results for a single datatype.
    
    Args:
        datatype_dir: Path to datatype directory under machine_learning_subtyping/{datatype}.
        repeat: Number of repeats (k). Files are expected at:
            machine_learning_subtyping/{datatype}/fold_{fold_idx}/fold_predictions.tsv
        cv_folds: Number of folds per repeat (e.g. outer_cv). If None, infer from existing fold_* dirs
            by requiring total_folds % repeat == 0.
        sample_type: Which rows to take from fold_predictions.tsv (default: "test").
    
    Returns:
        df: pandas.DataFrame with columns:
            - sample
            - true_state
            - {datatype}_pseudotime
            - {datatype}_score_{r} for r in [0, repeat)
    """
    if int(repeat) < 1:
        raise ValueError("repeat must be >= 1")

    datatype = os.path.basename(os.path.normpath(datatype_dir))
    # Detect available fold indices (fold_0, fold_1, ...)
    fold_indices: list[int] = []
    i = 0
    while True:
        fold_dir = os.path.join(datatype_dir, f"fold_{i}")
        fold_file = os.path.join(fold_dir, "fold_predictions.tsv")
        if os.path.exists(fold_file):
            fold_indices.append(i)
            i += 1
            continue
        break
    if not fold_indices:
        raise FileNotFoundError(f"no fold_*/fold_predictions.tsv found under {datatype_dir}")

    total_folds = len(fold_indices)
    if cv_folds is None:
        if total_folds % int(repeat) != 0:
            raise ValueError(
                f"cannot infer cv_folds: total_folds={total_folds} not divisible by repeat={repeat} "
                f"under {datatype_dir}. Please pass --cv_folds."
            )
        cv_folds = total_folds // int(repeat)
    if int(cv_folds) < 1:
        raise ValueError("cv_folds must be >= 1")
    expected_total = int(cv_folds) * int(repeat)
    if total_folds != expected_total:
        raise ValueError(
            f"{datatype}: fold count mismatch. detected total_folds={total_folds}, "
            f"but expected cv_folds*repeat={cv_folds}*{repeat}={expected_total}."
        )

    # For each repeat r, collect test predictions across its folds, then assemble one score per sample.
    per_repeat_tables: list[pd.DataFrame] = []
    for r in range(int(repeat)):
        parts: list[pd.DataFrame] = []
        for f in range(int(cv_folds)):
            fold_idx = r * int(cv_folds) + f
            fold_file = os.path.join(datatype_dir, f"fold_{fold_idx}", "fold_predictions.tsv")
            if not os.path.exists(fold_file):
                raise FileNotFoundError(f"missing fold_predictions.tsv: {fold_file}")

            df = pd.read_csv(fold_file, sep="\t")
            required = {"sample", "type", "true_state", "true_pseudotime", "progression_score"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{fold_file} missing columns: {sorted(missing)}")

            # Min-max normalize progression_score within this fold file,
            # then take the requested split values (e.g., type == "test").
            score = df["progression_score"].astype(float)
            score_min = float(score.min())
            score_max = float(score.max())
            if score_max == score_min:
                df["progression_score"] = 0.0
            else:
                df["progression_score"] = (score - score_min) / (score_max - score_min)

            df = df.loc[df["type"] == sample_type].copy()
            if df.empty:
                raise ValueError(f"{fold_file}: no rows after type == {sample_type!r} filter")

            df = df[["sample", "true_state", "true_pseudotime", "progression_score"]].copy()
            parts.append(df)

        rr = pd.concat(parts, ignore_index=True)
        # In one repeat, each sample should appear exactly once as test across all folds.
        dup = rr["sample"].astype(str).duplicated(keep=False)
        if dup.any():
            bad = rr.loc[dup, "sample"].astype(str).unique().tolist()
            raise ValueError(f"{datatype}: repeat {r} has duplicated test samples: {bad[:10]}")

        rr = rr.rename(
            columns={
                "true_pseudotime": f"{datatype}_pseudotime",
                "progression_score": f"{datatype}_score_{r}",
            }
        )
        per_repeat_tables.append(rr)

    # Merge repeats within the datatype on sample (score columns differ per repeat)
    merged = per_repeat_tables[0]
    for r in range(1, len(per_repeat_tables)):
        score_col = f"{datatype}_score_{r}"
        merged = merged.merge(
            per_repeat_tables[r][["sample", score_col]],
            on="sample",
            how="inner",
        )

    # Validate consistency of true_state and pseudotime across repeats
    state_nuniq = merged.groupby("sample")["true_state"].nunique()
    if not (state_nuniq == 1).all():
        bad = state_nuniq[state_nuniq != 1].index.tolist()
        raise ValueError(f"{datatype}: inconsistent true_state across repeats for samples: {bad[:10]}")

    pseudo_col = f"{datatype}_pseudotime"
    pseudo_nuniq = merged.groupby("sample")[pseudo_col].nunique()
    if not (pseudo_nuniq == 1).all():
        bad = pseudo_nuniq[pseudo_nuniq != 1].index.tolist()
        raise ValueError(f"{datatype}: inconsistent pseudotime across repeats for samples: {bad[:10]}")

    merged = merged.drop_duplicates(subset=["sample"]).sort_values("sample").reset_index(drop=True)
    return merged


def aggregate_fold_probs(prob_dfs: dict):
    """
    Merge prediction DataFrames from multiple datatypes by sample.
    
    Args:
        prob_dfs: dict, key=datatype name, value=DataFrame from fold_probs.
    
    Returns:
        df: pandas.DataFrame with all datatype results, merged on sample:
            - sample, true_state, {datatype}_pseudotime, {datatype}_score_{k}, ...
    
    Notes:
        - Enforces that each datatype has the same set of samples (order can differ).
    """
    
    if not prob_dfs or len(prob_dfs) == 0:
        raise ValueError("prob_dfs must be a non-empty dictionary")

    # Validate sample sets match across datatypes
    keys = list(prob_dfs.keys())
    base_key = keys[0]
    base_df = prob_dfs[base_key].copy()
    if "sample" not in base_df.columns:
        raise ValueError(f"{base_key}: missing 'sample' column")
    base_samples = set(base_df["sample"].astype(str).tolist())
    if len(base_samples) != len(base_df):
        raise ValueError(f"{base_key}: duplicate sample names detected")

    for k in keys[1:]:
        df = prob_dfs[k]
        if "sample" not in df.columns:
            raise ValueError(f"{k}: missing 'sample' column")
        samples = set(df["sample"].astype(str).tolist())
        if samples != base_samples:
            missing = sorted(list(base_samples - samples))[:10]
            extra = sorted(list(samples - base_samples))[:10]
            raise ValueError(
                f"sample set mismatch between {base_key} and {k}. "
                f"missing_in_{k}={missing} extra_in_{k}={extra}"
            )

    # Merge on sample, keep one true_state
    merged = base_df.copy()
    for k in keys[1:]:
        df = prob_dfs[k].copy()
        # Ensure consistent true_state
        check = merged.merge(df[["sample", "true_state"]], on="sample", how="inner", suffixes=("", "_rhs"))
        if not (check["true_state"].astype(str) == check["true_state_rhs"].astype(str)).all():
            bad = check.loc[check["true_state"].astype(str) != check["true_state_rhs"].astype(str), "sample"].tolist()
            raise ValueError(f"inconsistent true_state between datatypes for samples: {bad[:10]}")

        # Drop redundant true_state, merge remaining columns
        cols_to_add = [c for c in df.columns if c not in {"true_state"}]
        merged = merged.merge(df[cols_to_add], on="sample", how="inner")

    merged = merged.sort_values("sample").reset_index(drop=True)
    return merged


def add_y_test(df: pd.DataFrame, true_state_col: str = "true_state") -> pd.DataFrame:
    """
    Map true_state to 3-class labels:
        - S00 -> class0
        - S10/S01 -> class1
        - S11 -> class2
    Result column is named y_test (int).
    """
    if true_state_col not in df.columns:
        raise ValueError(f"missing column: {true_state_col}")

    def _to_class(x: Any) -> int:
        s = str(x).strip()
        s_upper = s.upper()
        if s_upper == "S00":
            return 0
        if s_upper in {"S10", "S01"}:
            return 1
        if s_upper == "S11":
            return 2
        raise ValueError(f"unknown true_state: {x!r}")

    out = df.copy()
    out["y_test"] = out[true_state_col].apply(_to_class).astype(int)
    return out


def plot_multi_feature_boxplot(
    merged_df: pd.DataFrame,
    output_path: str = "multi_feature_boxplot.png",
    figsize: tuple = (12, 6),
):
    """
    Draw boxplots with panels:
      - horizontal axis: datatype
      - vertical axis: repeat k
    Inside each panel:
      - x axis: class (C0/C1/C2) from y_test
      - y axis: {datatype}_score_{k} values
    """
    import re

    if "y_test" not in merged_df.columns:
        raise ValueError("merged_df missing y_test. Call add_y_test_from_true_state() first.")

    score_cols = [c for c in merged_df.columns if re.match(r".+_score_\d+$", c)]
    if not score_cols:
        raise ValueError("No {datatype}_score_{k} columns found")

    datatypes = sorted({c.rsplit("_score_", 1)[0] for c in score_cols})
    repeats = sorted({int(c.rsplit("_score_", 1)[1]) for c in score_cols})
    n_rows = len(repeats)      # vertical: repeat
    n_cols = len(datatypes)    # horizontal: datatype
    if n_rows == 0 or n_cols == 0:
        raise ValueError("No datatypes/repeats detected from score columns")

    sns.set_style("whitegrid")

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        sharex=False,
        sharey=False,
        figsize=figsize,
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Fix class order for stable panels
    hue_order = [0, 1, 2]

    for i, k in enumerate(repeats):  # vertical
        for j, dt in enumerate(datatypes):  # horizontal
            ax = axes[i, j]
            col = f"{dt}_score_{k}"
            if col not in merged_df.columns:
                ax.set_visible(False)
                continue

            subset = merged_df[["y_test", col]].copy()
            subset = subset.rename(columns={col: "score"})
            subset = subset.dropna(subset=["score"])
            if subset.empty:
                ax.set_visible(False)
                continue

            sns.boxplot(
                data=subset,
                x="y_test",
                y="score",
                order=hue_order,
                hue="y_test",
                palette=CLASS_COLORS[: len(hue_order)],
                showfliers=False,
                linewidth=1.5,
                ax=ax,
            )
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

            ax.set_xticklabels([f"C{c}" for c in hue_order])
            ax.set_xlabel("")

            if i == 0:
                ax.set_title(dt, fontsize=12, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"repeat {k}\nscore", fontsize=10, fontweight="bold")
            else:
                ax.set_ylabel("")

            ax.grid(True, alpha=0.2)

    fig.suptitle(
        "Progression score boxplots by y_test (per repeat & datatype)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Multi-feature boxplot saved to {output_path}")
    return None


def main(args):
    feature_dirs = args.feature_dirs
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    prob_dfs = {}
    for feature_dir in feature_dirs:
        datatype_dir = feature_dir
        prob_df = fold_probs(
            datatype_dir=datatype_dir,
            repeat=int(args.cv_repeats),
            cv_folds=getattr(args, "cv_folds", None),
            sample_type="test",
        )
        datatype = os.path.basename(os.path.normpath(datatype_dir))
        prob_dfs[datatype] = prob_df

    aggregate_df = aggregate_fold_probs(prob_dfs=prob_dfs)
    aggregate_df = add_y_test(aggregate_df, true_state_col="true_state")
    aggregate_df.to_csv(os.path.join(output_path, "aggregate_df.tsv"), sep="\t", index=False)
    multi_feature_plot = os.path.join(output_path, "multi_feature_boxplot.png")
    plot_multi_feature_boxplot(aggregate_df, multi_feature_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--feature_dirs", type=str, nargs='+', required=True, help="path to feature directories")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="path to output directory")
    parser.add_argument("--cv_repeats", type=int, required=True, help="number of repeats (outer repeats).")
    parser.add_argument("--cv_folds", type=int, default=None, help="number of CV folds per repeat (e.g. 5)")
    args = parser.parse_args()
    main(args)
