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
OUTPUT_DIR="model_subtyping/"
# 二级模型训练模式：single（默认）或 full_search
SECONDARY_MODE="full_search"

echo "Output directory: ${OUTPUT_DIR}"
echo "Secondary model mode: ${SECONDARY_MODE}"

# Step 1: integrate_model.py
# Input: machine_learning_subtyping 下各 datatype（OCR, gene_counts, consensus_peak）
# Output: OUTPUT_DIR folder (需新建)
if [ -f "${OUTPUT_DIR}/aggregate_df.tsv" ]; then
    echo "Step 1: Skipping integrate_model.py (aggregate_df.tsv already exists)."
else
    echo "Step 1: Running integrate_model.py..."
    python model_subtyping/integrate_model.py \
        -f machine_learning_subtyping/OCR \
        machine_learning_subtyping/gene_counts \
        machine_learning_subtyping/consensus_peak \
        machine_learning_subtyping/window \
        --cv_repeats 2 \
        --cv_folds 5 \
        -o ${OUTPUT_DIR}
    if [ ! -f "${OUTPUT_DIR}/aggregate_df.tsv" ]; then
        echo "Error: Step 1 failed - aggregate_df.tsv not found"
        exit 1
    fi
fi

# Step 2: train_secondary_model.py
# Input: first file output中的aggregate_df.tsv
# Output: OUTPUT_DIR/ML_results
if [ -f "${OUTPUT_DIR}/ML_results/nestcv_results.csv" ]; then
    echo "Step 2: Skipping train_secondary_model.py (nestcv_results.csv already exists)."
else
    echo "Step 2: Running train_secondary_model.py..."
    python model_subtyping/train_secondary_model.py \
        -i ${OUTPUT_DIR}/aggregate_df.tsv \
        -o ${OUTPUT_DIR}/ML_results \
        --cv_folds 5 \
        --cv_repeats 3 \
        --mode ${SECONDARY_MODE}
    if [ ! -f "${OUTPUT_DIR}/ML_results/nestcv_results.csv" ]; then
        echo "Error: Step 2 failed - nestcv_results.csv not found"
        exit 1
    fi
fi

# Step 3: evaluate_model.py
# Input: first file output
# Output: OUTPUT_DIR/ML_results
if [ -f "${OUTPUT_DIR}/ML_results/auc.png" ]; then
    echo "Step 3: Skipping evaluate_model.py (auc.png already exists)."
else
    echo "Step 3: Running evaluate_model.py..."
    python model_subtyping/evaluate_model.py \
        -f ${OUTPUT_DIR} \
        -o ${OUTPUT_DIR}/ML_results \
        --cv_repeats 3
    if [ $? -ne 0 ] || [ ! -f "${OUTPUT_DIR}/ML_results/auc.png" ]; then
        echo "Error: Step 3 failed"
        exit 1
    fi
fi

echo "All steps completed successfully!"
