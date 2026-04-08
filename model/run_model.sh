#! /bin/bash
#SBATCH -J model_pipeline
#SBATCH -N 1
#SBATCH -p normal
#SBATCH -n 48
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --mem 320g

source activate py310
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

# 设置输出目录变量
OUTPUT_DIR="model/"

# Step 2 模式: single=单次 nestcv 全列; full_search=120 种 datatype 组合搜索后保留最佳
SECONDARY_MODE="full_search"

echo "Output directory: ${OUTPUT_DIR}"

# Step 1: integrate_model.py
# Input: machine_learning folder下的artemis, end_motif, FSD, consensus_peak, OCR, window, gene_counts
# Output: OUTPUT_DIR/aggregate_df.tsv
if [ -f "${OUTPUT_DIR}/aggregate_df.tsv" ]; then
    echo "Step 1: aggregate_df.tsv already exists, skipping."
else
    echo "Step 1: Running integrate_model.py..."
    python model/integrate_model.py \
        -f machine_learning_tumor/artemis \
        machine_learning_tumor/end_motif \
        machine_learning_tumor/FSD \
        machine_learning_tumor/consensus_peak \
        machine_learning_tumor/OCR \
        machine_learning_tumor/window \
        machine_learning_tumor/gene_counts \
        -m cfDNA_metadata2_TNM.csv \
        -o ${OUTPUT_DIR}
    if [ ! -f "${OUTPUT_DIR}/aggregate_df.tsv" ]; then
        echo "Error: Step 1 failed - aggregate_df.tsv not found"
        exit 1
    fi
fi

# Step 2: train_secondary_model.py
# Input: aggregate_df.tsv
# Output: OUTPUT_DIR/ML_results (nestcv_results.csv; full_search 时另有 datatype_search_log.csv)
if [ -f "${OUTPUT_DIR}/ML_results/nestcv_results.csv" ]; then
    echo "Step 2: nestcv_results.csv already exists, skipping."
else
    echo "Step 2: Running train_secondary_model.py (--mode ${SECONDARY_MODE})..."
    python model/train_secondary_model.py \
        -i ${OUTPUT_DIR}/aggregate_df.tsv \
        -o ${OUTPUT_DIR}/ML_results \
        --cv_folds 3 \
        --cv_repeats 3 \
        --mode ${SECONDARY_MODE}
    if [ ! -f "${OUTPUT_DIR}/ML_results/nestcv_results.csv" ]; then
        echo "Error: Step 2 failed - nestcv_results.csv not found"
        exit 1
    fi
fi

# Step 3: evaluate_model.py
# Input: OUTPUT_DIR (aggregate_df 等)
# Output: OUTPUT_DIR/ML_results/auc.png
if [ -f "${OUTPUT_DIR}/ML_results/auc.png" ]; then
    echo "Step 3: evaluate_model output (auc.png) already exists, skipping."
else
    echo "Step 3: Running evaluate_model.py..."
    python model/evaluate_model.py \
        -f ${OUTPUT_DIR} \
        -o ${OUTPUT_DIR}/ML_results
    if [ ! -f "${OUTPUT_DIR}/ML_results/auc.png" ]; then
        echo "Error: Step 3 failed - auc.png not found"
        exit 1
    fi
fi

echo "All steps completed successfully!"
