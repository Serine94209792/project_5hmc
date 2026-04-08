#!/usr/bin/env python3
"""
Tissue of Origin (TOO) analysis: merge all tissue peaks, one coverage per sample (Plan B),
then aggregate counts by tissue and compute score/percent/CV and stability plot.
"""
from __future__ import annotations
import argparse
import os
import re
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import pybedtools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_peak_paths(peak_paths: list[str]) -> list[tuple[str, str, str]]:
    """Parse peak file paths to (full_path, tissue, replicate)."""
    result = []
    re_tissue_rep = re.compile(r"^.+_([a-zA-Z]+)(\d+)_peaks\.narrowPeak$")
    for fp in sorted(peak_paths):
        basename = os.path.basename(fp)
        m = re_tissue_rep.match(basename)
        if m:
            tissue, rep = m.group(1), m.group(2)
        else:
            stem = basename.replace("_peaks.narrowPeak", "").replace(".narrowPeak", "")
            tissue = stem
            rep = "0"
        result.append((fp, tissue, rep))
    return result


def sample_from_fragment_path(frag_path: str) -> str:
    """Extract sample name from path like .../sample_fragments.bed."""
    basename = os.path.basename(frag_path)
    if basename.endswith("_fragments.bed"):
        return basename.replace("_fragments.bed", "")
    return basename.replace(".bed", "")


def total_bp_from_narrowpeak(peak_path: str) -> int:
    """Sum (end - start) for all intervals in narrowPeak (first 3 cols are BED)."""
    df = pd.read_csv(peak_path, sep="\t", comment="#", header=None, usecols=[1, 2])
    df.columns = ["start", "end"]
    return int((df["end"] - df["start"]).sum())


def _coverage_one_sample(args: tuple) -> tuple:
    """Run one coverage(merged_peaks, frag_path), aggregate count by col_name, return (sample, {col: score})."""
    sample, frag_path, merged_peak_path, total_bp_per_col, col_names = args
    peaks = pybedtools.BedTool(merged_peak_path)
    frags = pybedtools.BedTool(frag_path)
    cov = peaks.coverage(frags, counts=True, sorted=True)
    # coverage adds last column = count; merged BED has 4 cols: chr, start, end, name
    count_by_col = {c: 0 for c in col_names}
    for interval in cov:
        if len(interval.fields) < 5:
            continue
        name = interval.fields[3]
        cnt = int(interval.fields[-1])
        if name in count_by_col:
            count_by_col[name] += cnt
    scores = {}
    for c in col_names:
        tb = total_bp_per_col.get(c, 0)
        if tb <= 0:
            scores[c] = 0.0
        else:
            scores[c] = (count_by_col.get(c, 0) / tb) * 1e3
    return (sample, scores)


def run_too(
    fragment_paths: list[str],
    peak_paths: list[str],
    out_score: str,
    out_cv: str,
    out_plot: str,
    out_percent: str,
    n_jobs: int = 1,
) -> None:
    samples = [sample_from_fragment_path(p) for p in fragment_paths]
    peak_list = parse_peak_paths(peak_paths)
    if not peak_list:
        raise ValueError("No peak files provided")

    col_names = [os.path.basename(fp).replace("_peaks.narrowPeak", "") for fp, _, _ in peak_list]
    peak_bp = {}
    for fp, tissue, rep in peak_list:
        peak_bp[fp] = total_bp_from_narrowpeak(fp)
    total_bp_per_col = {col_names[i]: peak_bp[fp] for i, (fp, _, _) in enumerate(peak_list)}

    tmpdir = tempfile.mkdtemp(prefix="too_")
    try:
        # 1) Build merged BED: chr, start, end, col_name (one row per interval, col_name = which peak file)
        merged_rows = []
        for i, (fp, tissue, rep) in enumerate(peak_list):
            col = col_names[i]
            df = pd.read_csv(fp, sep="\t", comment="#", header=None, usecols=[0, 1, 2])
            df.columns = ["chr", "start", "end"]
            df["name"] = col
            merged_rows.append(df)
        merged_df = pd.concat(merged_rows, ignore_index=True)
        merged_bed = os.path.join(tmpdir, "merged_peaks.bed")
        merged_df.to_csv(merged_bed, sep="\t", index=False, header=False)
        merged_sorted = os.path.join(tmpdir, "merged_peaks.sorted.bed")
        pybedtools.BedTool(merged_bed).sort().saveas(merged_sorted)

        # 2) One coverage per sample (optionally in parallel)
        score_df = pd.DataFrame(index=samples, columns=col_names, dtype=float)
        frag_by_sample = {sample_from_fragment_path(p): p for p in fragment_paths}
        task_args = [
            (sample, frag_by_sample[sample], merged_sorted, total_bp_per_col, col_names)
            for sample in samples
            if frag_by_sample.get(sample) and os.path.isfile(frag_by_sample[sample])
        ]
        missing = set(samples) - {t[0] for t in task_args}
        for s in missing:
            score_df.loc[s, :] = np.nan

        if n_jobs <= 1:
            for (sample, frag_path, merged_path, tbp, cols) in task_args:
                _, scores = _coverage_one_sample((sample, frag_path, merged_path, tbp, cols))
                for c, v in scores.items():
                    score_df.loc[sample, c] = v
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                futs = {ex.submit(_coverage_one_sample, t): t for t in task_args}
                for fut in as_completed(futs):
                    sample, scores = fut.result()
                    for c, v in scores.items():
                        score_df.loc[sample, c] = v
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

    score_df = score_df.astype(float)
    os.makedirs(os.path.dirname(out_score) or ".", exist_ok=True)
    score_df.to_csv(out_score, sep="\t")

    peak_to_tissue = {}
    peak_to_rep = {}
    for fp, tissue, rep in peak_list:
        col = os.path.basename(fp).replace("_peaks.narrowPeak", "")
        peak_to_tissue[col] = tissue
        peak_to_rep[col] = rep

    tissues = sorted(set(peak_to_tissue.values()))
    cv_rows = []
    for t in tissues:
        cols_t = [c for c in score_df.columns if peak_to_tissue.get(c) == t]
        if not cols_t:
            continue
        vals = score_df[cols_t].values.flatten()
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            cv_rows.append({"tissue": t, "CV": np.nan})
        else:
            mean_v = np.mean(vals)
            cv = np.std(vals) / mean_v if mean_v != 0 else np.nan
            cv_rows.append({"tissue": t, "CV": cv})
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(out_cv, sep="\t", index=False)

    plot_data = []
    for t in tissues:
        cols_t = [c for c in score_df.columns if peak_to_tissue.get(c) == t]
        if not cols_t:
            continue
        rep_means = score_df[cols_t].mean(axis=0)
        tissue_mean = rep_means.mean()
        for col in cols_t:
            rep_id = peak_to_rep.get(col, col)
            plot_data.append({
                "tissue": t,
                "replicate": rep_id,
                "mean_score": rep_means[col],
                "deviation": rep_means[col] - tissue_mean,
            })
    plot_df = pd.DataFrame(plot_data)
    fig, ax = plt.subplots(figsize=(max(8, len(tissues) * 0.5), 5))
    x_pos = {t: i for i, t in enumerate(tissues)}
    for _, row in plot_df.iterrows():
        x = x_pos[row["tissue"]]
        ax.scatter(x, row["deviation"], alpha=0.7, s=40)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xticks(range(len(tissues)))
    ax.set_xticklabels(tissues, rotation=45, ha="right")
    ax.set_xlabel("Tissue")
    ax.set_ylabel("Replicate deviation from tissue mean")
    ax.set_title("TOO: Replicate stability (deviation from tissue mean)")
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)
    plt.close()

    score_by_tissue = pd.DataFrame(index=samples, columns=tissues, dtype=float)
    for t in tissues:
        cols_t = [c for c in score_df.columns if peak_to_tissue.get(c) == t]
        if cols_t:
            score_by_tissue[t] = score_df[cols_t].mean(axis=1)
        else:
            score_by_tissue[t] = np.nan
    score_by_tissue = score_by_tissue.astype(float)
    row_sums = score_by_tissue.sum(axis=1)
    percent_df = score_by_tissue.div(row_sums.replace(0, np.nan), axis=0)
    percent_df = percent_df.fillna(0)
    percent_df.to_csv(out_percent, sep="\t")


def main():
    parser = argparse.ArgumentParser(description="TOO: Tissue of Origin analysis (Plan B: merged peaks, one coverage per sample)")
    parser.add_argument("--fragments", nargs="+", required=True, help="Fragment BED files (one per sample)")
    parser.add_argument("--peaks", nargs="+", required=True, help="Tissue-specific peak narrowPeak files")
    parser.add_argument("--out-score", required=True, help="Output: tissue_score.tsv (replicate-level)")
    parser.add_argument("--out-cv", required=True, help="Output: tissue_CV.tsv")
    parser.add_argument("--out-plot", required=True, help="Output: tissue_replicate_deviation.png")
    parser.add_argument("--out-percent", required=True, help="Output: tissue_percent.tsv")
    parser.add_argument("--n-jobs", type=int, default=4, help="Parallel workers (one coverage per sample; default 4, use fewer for memory)")
    args = parser.parse_args()
    run_too(
        fragment_paths=args.fragments,
        peak_paths=args.peaks,
        out_score=args.out_score,
        out_cv=args.out_cv,
        out_plot=args.out_plot,
        out_percent=args.out_percent,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
