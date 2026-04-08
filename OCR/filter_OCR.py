import argparse
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def remove_high_zero_columns(df: pd.DataFrame,
                             threshold: float = 0.9) -> pd.DataFrame:
    """
    remove columns with high zero ratio
    """
    zero_ratio = (df == 0).sum(axis=0) / df.shape[0]
    cols_to_keep = zero_ratio < threshold
    return df.loc[:, cols_to_keep]


def main(args):
    df = pd.read_csv(args.input, sep="\t", header=0)
    valid_chr = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
    df = df[df["chr"].isin(valid_chr)]
    print(f"After filtering chromosomes: {len(df)} rows")
    
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=["chr", "start", "end"], keep="first")
    n_after_dedup = len(df)
    if n_before_dedup != n_after_dedup:
        print(f"Removed {n_before_dedup - n_after_dedup} duplicate rows")
    
    ocr = df[["chr", "start", "end"]]
    mat = df.drop(columns=["chr", "start", "end"])
    sample_names = mat.columns.tolist()
    mat = mat.transpose()
    mat = remove_high_zero_columns(mat, threshold=args.zero_threshold)
    vt = VarianceThreshold(threshold=args.variance_threshold)
    mat = vt.fit_transform(mat)
    kept_indices = vt.get_support(indices=True)
    ocr = ocr.iloc[kept_indices].reset_index(drop=True)
    mat = pd.DataFrame(mat, index=sample_names, columns=range(len(kept_indices)))
    mat = mat.transpose()
    mat = pd.concat([ocr, mat], axis=1)
    mat.to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get final OCR count")
    parser.add_argument("-i", "--input", required=True, help="input tsv file")
    parser.add_argument("-o", "--output", required=True, help="output tsv file")
    parser.add_argument("-vt", "--variance_threshold", type=float, required=True, help="variance threshold")
    parser.add_argument("-zt", "--zero_threshold", type=float, required=True, help="zero threshold")
    args = parser.parse_args()
    main(args)

