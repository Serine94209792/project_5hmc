import argparse
import pybedtools as pbt
import pandas as pd
import os


def calculate_ocr_coverage(ocr_bed: str, fragment_bed: str) -> pd.DataFrame:
    """
    计算单个样本的fragment在OCR上的coverage
    
    Parameters:
    -----------
    ocr_bed: str
        OCR bed文件路径
    fragment_bed: str
        Fragment bed文件路径
    
    Returns:
    --------
    pd.DataFrame
        包含chr, start, end, count的DataFrame
    """
    ocr_bedtool = pbt.BedTool(ocr_bed)
    if not os.path.exists(fragment_bed):
        raise FileNotFoundError(f"fragment file not found: {fragment_bed}")
    
    fragment_bedtool = pbt.BedTool(fragment_bed)
    
    if len(fragment_bedtool) == 0:
        raise ValueError(f"fragment file is empty: {fragment_bed}")
    
    coverage_result = ocr_bedtool.coverage(fragment_bedtool, counts=True)
    df = coverage_result.to_dataframe(names=["chr", "start", "end", "count"])
    df_result = df.drop_duplicates(subset=["chr", "start", "end"], keep="first")
    
    return df_result


def get_sample_name(fragment_path: str) -> str:
    """
    从fragment文件路径中提取样本名称
    
    Parameters:
    -----------
    fragment_path: str
        Fragment文件路径，格式如: fragments/{sample}_fragments.bed
    
    Returns:
    --------
    str
        样本名称
    """
    basename = os.path.basename(fragment_path)
    if basename.endswith("_fragments.bed"):
        return basename.replace("_fragments.bed", "")
    elif basename.endswith(".bed"):
        return basename.replace(".bed", "")
    else:
        return basename


def merge_all_samples(ocr_bed: str, fragment_files: list, output_file: str):
    """
    计算所有样本在OCR上的coverage并合并为一个tsv文件
    
    Parameters:
    -----------
    ocr_bed: str
        OCR bed文件路径
    fragment_files: list
        所有样本的fragment bed文件路径列表
    output_file: str
        输出tsv文件路径
    """
    ocr_bedtool = pbt.BedTool(ocr_bed)
    ocr_df = ocr_bedtool.to_dataframe()
    
    if len(ocr_df) == 0:
        raise ValueError(f"OCR file is empty: {ocr_bed}")
    
    result_df = ocr_df.iloc[:, [0, 1, 2]].copy()
    result_df.columns = ["chr", "start", "end"]
    result_df = result_df.set_index(["chr", "start", "end"])
    
    for fragment_file in fragment_files:
        sample_name = get_sample_name(fragment_file)
        coverage_df = calculate_ocr_coverage(ocr_bed, fragment_file)
        print(f"coverage_df shape for {sample_name}: {coverage_df.shape}")
        
        if len(coverage_df) > 0:
            coverage_df = coverage_df.rename(columns={"count": sample_name})
            coverage_df_indexed = coverage_df.set_index(["chr", "start", "end"])[[sample_name]]
            result_df = result_df.join(coverage_df_indexed, how="left")
            result_df[sample_name] = result_df[sample_name].fillna(0).astype(int)
            print(f"result_df shape after {sample_name}: {result_df.shape}")
        else:
            raise ValueError(f"coverage file is empty: {fragment_file}")
        
    result_df = result_df.reset_index()
        
    result_df.to_csv(output_file, sep="\t", index=False)
    

def main():
    parser = argparse.ArgumentParser(
        description="计算多个样本的fragment在OCR上的coverage并合并为一个tsv文件"
    )
    parser.add_argument(
        "--ocr",
        required=True,
        help="OCR bed文件路径"
    )
    parser.add_argument(
        "-i", "--fragments",
        nargs="+",
        help="Fragment bed文件路径列表（可指定多个）"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="输出tsv文件路径"
    )
    
    args = parser.parse_args()
    
    fragment_files = args.fragments
    
    merge_all_samples(args.ocr, fragment_files, args.output)


if __name__ == "__main__":
    main()

