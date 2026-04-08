#!/bin/bash
#SBATCH -J generate_features
#SBATCH -N 1
#SBATCH -p normal
#SBATCH --cpus-per-task=48
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --mem=320g

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 设置基础路径
BASE_DIR="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

METADATA_FILE="${BASE_DIR}/cfDNA_metadata2_TNM.csv"
LABEL="type"

# 定义datatype和对应的heatmap_top_n和violin_top_n参数
# 格式: datatype:heatmap_top_n:violin_top_n
declare -A DATATYPE_PARAMS=(
    ["artemis"]="50:6"
    ["end_motif"]="50:6"
    ["FSD"]="50:6"
    ["OCR"]="50:6"
    ["gene_counts"]="50:6"
    ["consensus_peak"]="50:6"
    ["window"]="50:6"
)

# 检查metadata文件是否存在
if [ ! -f "$METADATA_FILE" ]; then
    echo "Error: Metadata file not found: $METADATA_FILE"
    exit 1
fi

# 创建临时脚本目录
TEMP_SCRIPT_DIR="${SCRIPT_DIR}/sbatch_scripts"
mkdir -p "$TEMP_SCRIPT_DIR"

# 遍历每个datatype生成sbatch脚本
for datatype in artemis end_motif FSD OCR gene_counts consensus_peak window; do
    echo "=========================================="
    echo "Processing datatype: $datatype"
    echo "=========================================="
    
    # 获取输入目录（即differential_analysis的输出目录）
    input_dir="${SCRIPT_DIR}/${datatype}"
    
    # 检查输入目录是否存在
    if [ ! -d "$input_dir" ]; then
        echo "Warning: Input directory not found for $datatype: $input_dir"
        echo "Skipping $datatype..."
        continue
    fi
    
    # 检查feature_analysis_matrix.tsv是否存在
    if [ ! -f "${input_dir}/feature_analysis_matrix.tsv" ]; then
        echo "Warning: feature_analysis_matrix.tsv not found in ${input_dir}"
        echo "Skipping $datatype..."
        continue
    fi
    
    # 获取heatmap_top_n和violin_top_n参数
    params="${DATATYPE_PARAMS[$datatype]}"
    if [ -z "$params" ]; then
        echo "Warning: Parameters not defined for $datatype"
        echo "Skipping $datatype..."
        continue
    fi
    
    IFS=':' read -r heatmap_top_n violin_top_n <<< "$params"
    
    # 输出目录与输入目录相同（即feature_selection/${datatype}）
    output_dir="${input_dir}"
    
    # 生成独立的sbatch脚本
    sbatch_script="${TEMP_SCRIPT_DIR}/generate_${datatype}.sh"
    
    cat > "$sbatch_script" << EOF
#!/bin/bash
#SBATCH -J generate_${datatype}
#SBATCH -N 1
#SBATCH -p normal
#SBATCH --cpus-per-task=48
#SBATCH -o ${output_dir}/%x_%j.out
#SBATCH -e ${output_dir}/%x_%j.err
#SBATCH --mem=320g

cd "$SCRIPT_DIR"
source activate py310

python ${SCRIPT_DIR}/generate_diff_features.py \\
    -i "$input_dir" \\
    -t $datatype \\
    -m "$METADATA_FILE" \\
    -l $LABEL \\
    -o "$output_dir" \\
    -hn $heatmap_top_n \\
    -vn $violin_top_n

echo "Completed generate diff features for ${datatype}"
EOF
    
    chmod +x "$sbatch_script"
    
    # 显示生成的脚本信息
    echo "Generated sbatch script for $datatype:"
    echo "  Input dir: $input_dir"
    echo "  Output dir: $output_dir"
    echo "  Heatmap top N: $heatmap_top_n"
    echo "  Violin top N: $violin_top_n"
    echo "  Script: $sbatch_script"
    echo ""
    
    sbatch "$sbatch_script"
    if [ $? -eq 0 ]; then
        echo "Successfully submitted job for $datatype"
    else
        echo "Error: Failed to submit job for $datatype"
    fi
done

echo "=========================================="
echo "All scripts generated!"
echo "Scripts saved in: ${TEMP_SCRIPT_DIR}"
echo "=========================================="
