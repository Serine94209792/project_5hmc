import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pybedtools import BedTool


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='计算差异区域之间的Jaccard系数并绘制热图'
    )
    parser.add_argument(
        '--diff_peak', 
        required=True,
        help='差异peak的BED文件路径'
    )
    parser.add_argument(
        '--diff_window', 
        required=True,
        help='差异window的BED文件路径'
    )
    parser.add_argument(
        '--diff_OCR', 
        required=True,
        help='差异OCR的BED文件路径'
    )
    parser.add_argument(
        '--ts_promoter', 
        required=True,
        help='组织特异性启动子BED文件路径'
    )
    parser.add_argument(
        '--ts_enhancer', 
        required=True,
        help='组织特异性增强子BED文件路径'
    )
    parser.add_argument(
        '--genome_size',
        required=True,
        help='基因组大小文件路径（.genome文件），用于Fisher检验'
    )
    parser.add_argument(
        '--output_plot', 
        required=True,
        help='输出热图的PNG文件路径'
    )
    parser.add_argument(
        '--output_matrix', 
        default='jaccard_matrix.tsv',
        help='输出Jaccard矩阵的TSV文件路径'
    )
    parser.add_argument(
        '--output_fisher_ratio',
        default='fisher_ratio_matrix.tsv',
        help='输出Fisher富集ratio矩阵的TSV文件路径'
    )
    parser.add_argument(
        '--output_fisher_pvalue',
        default='fisher_pvalue_matrix.tsv',
        help='输出Fisher two-tail p值矩阵的TSV文件路径'
    )
    parser.add_argument(
        '--output_fisher_plot',
        required=True,
        help='输出Fisher ratio热图的PNG文件路径'
    )
    
    return parser.parse_args()


def calculate_jaccard_matrix(bed_files_dict):
    """
    计算所有BED文件之间的Jaccard系数矩阵
    
    参数:
        bed_files_dict: 字典，键为区域名称，值为BedTool对象
    
    返回:
        jaccard_matrix: DataFrame，Jaccard系数矩阵
    """
    names = list(bed_files_dict.keys())
    n = len(names)
    jaccard_matrix = np.zeros((n, n))
    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            if i == j:
                jaccard_matrix[i, j] = 1.0
            elif i < j:
                bed1 = bed_files_dict[name1]
                bed2 = bed_files_dict[name2]
                if len(bed1) == 0 or len(bed2) == 0:
                    jaccard_score = 0.0
                else:
                    bed1_sorted = bed1.sort()
                    bed2_sorted = bed2.sort()
                    jaccard_result = bed1_sorted.jaccard(bed2_sorted)
                    jaccard_score = jaccard_result['jaccard']
                
                jaccard_matrix[i, j] = jaccard_score
                jaccard_matrix[j, i] = jaccard_score
    
    jaccard_df = pd.DataFrame(
        jaccard_matrix,
        index=names,
        columns=names
    )
    
    return jaccard_df


def calculate_fisher_matrix(bed_files_dict, genome_file):
    """
    计算所有BED文件之间的Fisher精确检验矩阵
    
    参数:
        bed_files_dict: 字典，键为区域名称，值为BedTool对象
        genome_file: 基因组大小文件路径
    
    返回:
        fisher_ratio_df: DataFrame，Fisher富集ratio矩阵
        fisher_pvalue_df: DataFrame，Fisher two-tail p值矩阵
    """
    names = list(bed_files_dict.keys())
    n = len(names)
    fisher_ratio_matrix = np.full((n, n), np.nan)
    fisher_pvalue_matrix = np.full((n, n), np.nan)
    
    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            if i == j:
                fisher_ratio_matrix[i, j] = 1.0
                fisher_pvalue_matrix[i, j] = 1.0
            else:
                bed1 = bed_files_dict[name1]
                bed2 = bed_files_dict[name2]
                bed1_sorted = bed1.sort()
                bed2_sorted = bed2.sort()
                fisher_result = bed1_sorted.fisher(bed2_sorted, g=genome_file)
                fisher_ratio = fisher_result.ratio
                fisher_pvalue = fisher_result.two_tail
                fisher_ratio_matrix[i, j] = fisher_ratio
                fisher_pvalue_matrix[i, j] = fisher_pvalue
    
    fisher_ratio_df = pd.DataFrame(
        fisher_ratio_matrix,
        index=names,
        columns=names
    )
    
    fisher_pvalue_df = pd.DataFrame(
        fisher_pvalue_matrix,
        index=names,
        columns=names
    )
    
    return fisher_ratio_df, fisher_pvalue_df


def plot_jaccard_heatmap(jaccard_matrix, output_file):
    """
    绘制Jaccard系数热图
    
    参数:
        jaccard_matrix: DataFrame，Jaccard系数矩阵
        output_file: 输出文件路径
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        jaccard_matrix,
        annot=True,  # 显示数值
        fmt='.3f',   # 数值格式
        cmap='YlOrRd',  # 颜色映射
        square=True,  # 正方形单元格
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Jaccard Index'},
        linewidths=0.5,
        linecolor='gray'
    )
    
    plt.title('Jaccard Similarity Matrix', fontsize=16, pad=20)
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"heatmap saved to: {output_file}")


def plot_fisher_heatmap(fisher_ratio_matrix, output_file):
    """
    绘制Fisher富集ratio热图
    
    参数:
        fisher_ratio_matrix: DataFrame，Fisher富集ratio矩阵
        output_file: 输出文件路径
    """
    # 对ratio矩阵进行log2转换以便更好地可视化
    # 避免log(0)，将0替换为一个小值
    fisher_ratio_log = fisher_ratio_matrix.copy()
    fisher_ratio_log = fisher_ratio_log.replace(0, 0.001)
    fisher_ratio_log = np.log2(fisher_ratio_log)
    
    # 创建注释矩阵，将NaN值标记为"N/A"
    annot_matrix = fisher_ratio_matrix.copy().astype(str)
    for i in range(annot_matrix.shape[0]):
        for j in range(annot_matrix.shape[1]):
            val = fisher_ratio_matrix.iloc[i, j]
            if np.isnan(val):
                annot_matrix.iloc[i, j] = 'NaN'
            else:
                annot_matrix.iloc[i, j] = f'{val:.2f}'

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        fisher_ratio_log,
        annot=annot_matrix,  # 显示自定义的注释矩阵
        fmt='',   # 使用字符串格式，因为已经格式化好了
        cmap='RdBu_r',  # 红蓝颜色映射，红色表示富集，蓝色表示贫集
        square=True,  # 正方形单元格
        center=0,  # 以0为中心（log2(1)=0表示无富集）
        cbar_kws={'label': 'log2(Fisher Enrichment Ratio)'},
        linewidths=0.5,
        linecolor='gray',
        mask=np.isnan(fisher_ratio_matrix)  # 将NaN值用灰色遮罩
    )
    
    plt.title('Fisher Enrichment Ratio Matrix', fontsize=16, pad=20)
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Fisher heatmap saved to: {output_file}")


def main():
    args = parse_args()
    
    bed_files = {}
    bed_files['Diff Peak'] = BedTool(args.diff_peak)
    bed_files['Diff Window'] = BedTool(args.diff_window)
    bed_files['Diff OCR'] = BedTool(args.diff_OCR)
    bed_files['TS Promoter'] = BedTool(args.ts_promoter)
    bed_files['TS Enhancer'] = BedTool(args.ts_enhancer)
    
    jaccard_matrix = calculate_jaccard_matrix(bed_files)
    jaccard_matrix.to_csv(args.output_matrix, sep='\t', float_format='%.4f')
    
    fisher_ratio_matrix, fisher_pvalue_matrix = calculate_fisher_matrix(bed_files, args.genome_size)
    fisher_ratio_matrix.to_csv(args.output_fisher_ratio, sep='\t', float_format='%.4f')
    fisher_pvalue_matrix.to_csv(args.output_fisher_pvalue, sep='\t', float_format='%.6e')
    
    plot_jaccard_heatmap(jaccard_matrix, args.output_plot)
    plot_fisher_heatmap(fisher_ratio_matrix, args.output_fisher_plot)


if __name__ == "__main__":
    main()

