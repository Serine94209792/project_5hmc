#!/bin/bash
#SBATCH -J diff_analysis
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
CONFIG_FILE="${BASE_DIR}/config.yaml"

# 定义datatype和对应的输入文件
declare -A DATATYPE_INPUTS=(
    ["artemis"]="${BASE_DIR}/artemis2024/code/ARTEMIS_Pipeline/04_artemis_pipeline/artemis.csv"
    ["end_motif"]="${BASE_DIR}/end_motif/end_motif_freq.tsv"
    ["FSD"]="${BASE_DIR}/FSD/fsd.tsv"
    ["OCR"]="${BASE_DIR}/OCR/final_OCR_count.tsv"
    ["gene_counts"]="${BASE_DIR}/gene_counts/filtered_genebody_counts.tsv"
    ["consensus_peak"]="${BASE_DIR}/consensus_peak/filtered_peak_counts.tsv"
    ["window"]="${BASE_DIR}/window/window_count/final_window_count.tsv"
)

# 从config.yaml读取阈值配置的函数
get_threshold_from_config() {
    local datatype=$1
    local threshold_type=$2  # "logfc" or "qvalue"
    
    # 将datatype转换为config.yaml中的键名格式
    local config_key=""
    case $datatype in
        artemis)
            if [ "$threshold_type" == "logfc" ]; then
                config_key="logfc_threshold_artemis"
            elif [ "$threshold_type" == "qvalue" ]; then
                config_key="qvalue_threshold_artemis"
            fi
            ;;
        end_motif)
            if [ "$threshold_type" == "logfc" ]; then
                config_key="coefficient_threshold_end_motif"
            elif [ "$threshold_type" == "qvalue" ]; then
                config_key="qvalue_threshold_end_motif"
            fi
            ;;
        FSD)
            if [ "$threshold_type" == "logfc" ]; then
                config_key="logfc_threshold_fsd"
            elif [ "$threshold_type" == "qvalue" ]; then
                config_key="qvalue_threshold_fsd"
            fi
            ;;
        OCR)
            if [ "$threshold_type" == "logfc" ]; then
                config_key="logfc_threshold_OCR"
            elif [ "$threshold_type" == "qvalue" ]; then
                config_key="qvalue_threshold_OCR"
            fi
            ;;
        gene_counts)
            if [ "$threshold_type" == "logfc" ]; then
                config_key="logfc_threshold_genebody"
            elif [ "$threshold_type" == "qvalue" ]; then
                config_key="qvalue_threshold_genebody"
            fi
            ;;
        consensus_peak)
            if [ "$threshold_type" == "logfc" ]; then
                config_key="logfc_threshold_peak"
            elif [ "$threshold_type" == "qvalue" ]; then
                config_key="qvalue_threshold_peak"
            fi
            ;;
        window)
            if [ "$threshold_type" == "logfc" ]; then
                config_key="logfc_threshold_window"
            elif [ "$threshold_type" == "qvalue" ]; then
                config_key="qvalue_threshold_window"
            fi
            ;;
    esac
    
    if [ -n "$config_key" ]; then
        grep "^${config_key}:" "$CONFIG_FILE" | awk '{print $2}'
    fi
}

# 检查metadata和config文件是否存在
if [ ! -f "$METADATA_FILE" ]; then
    echo "Error: Metadata file not found: $METADATA_FILE"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# 创建临时脚本目录
TEMP_SCRIPT_DIR="${SCRIPT_DIR}/sbatch_scripts"
mkdir -p "$TEMP_SCRIPT_DIR"

# 遍历每个datatype生成sbatch脚本并提交
for datatype in artemis end_motif FSD OCR gene_counts consensus_peak window; do
    echo "=========================================="
    echo "Processing datatype: $datatype"
    echo "=========================================="
    
    # 获取输入文件路径
    input_file="${DATATYPE_INPUTS[$datatype]}"
    
    # 检查输入文件是否存在
    if [ ! -f "$input_file" ]; then
        echo "Warning: Input file not found for $datatype: $input_file"
        echo "Skipping $datatype..."
        continue
    fi
    
    # 从config.yaml读取阈值
    logfc_threshold=$(get_threshold_from_config "$datatype" "logfc")
    qvalue_threshold=$(get_threshold_from_config "$datatype" "qvalue")
    
    # 创建输出目录
    output_dir="${SCRIPT_DIR}/${datatype}"
    mkdir -p "$output_dir"
    
    # 生成独立的sbatch脚本
    sbatch_script="${TEMP_SCRIPT_DIR}/run_${datatype}.sh"
    
    cat > "$sbatch_script" << EOF
#!/bin/bash
#SBATCH -J diff_${datatype}
#SBATCH -N 1
#SBATCH -p normal
#SBATCH --cpus-per-task=48
#SBATCH -o ${output_dir}/%x_%j.out
#SBATCH -e ${output_dir}/%x_%j.err
#SBATCH --mem=320g

cd "$SCRIPT_DIR"
source activate py310

N_THREADS=\${SLURM_CPUS_PER_TASK:-48}

python ${SCRIPT_DIR}/differential_analysis.py \\
    -i "$input_file" \\
    -t $datatype \\
    -m "$METADATA_FILE" \\
    -l $LABEL \\
    -o "$output_dir" \\
    -q $qvalue_threshold \\
    -fc $logfc_threshold \\
    --threads \$N_THREADS

echo "Completed differential analysis for ${datatype}"
EOF
    
    chmod +x "$sbatch_script"
    
    # 提交sbatch任务
    echo "Submitting sbatch job for $datatype..."
    echo "  Input file: $input_file"
    echo "  Output dir: $output_dir"
    echo "  LogFC threshold: $logfc_threshold"
    echo "  Qvalue threshold: $qvalue_threshold"
    echo "  Script: $sbatch_script"
    echo ""
    
    sbatch "$sbatch_script"
    
    if [ $? -eq 0 ]; then
        echo "Successfully submitted job for $datatype"
    else
        echo "Error: Failed to submit job for $datatype"
    fi
    echo ""
done

echo "=========================================="
echo "All jobs submitted!"
echo "Check job status with: squeue -u \$USER"
echo "=========================================="
