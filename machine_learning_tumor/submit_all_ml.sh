#!/bin/bash
#SBATCH -J ML_test
#SBATCH -N 1
#SBATCH -p normal
#SBATCH --cpus-per-task=48
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --mem=320g

source activate py310
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
METADATA="${METADATA:-${PROJECT_ROOT}/cfDNA_metadata2_TNM.csv}"
LABEL="type"
INNER_CV=5
INNER_REPEATS=2
OUTER_CV=3
OUTER_REPEATS=3
MAX_ITER=30
N_TRIALS=10


declare -A DATA_FILES=(
    ["artemis"]="../feature_selection/artemis/artemis_log1p.tsv"
    ["FSD"]="../feature_selection/FSD/vst_counts.tsv"
    ["gene_counts"]="../feature_selection/gene_counts/vst_counts.tsv"
    ["consensus_peak"]="../feature_selection/consensus_peak/vst_counts.tsv"
    ["end_motif"]="../feature_selection/end_motif/end_motif_matrix.tsv"
    ["window"]="../feature_selection/window/vst_counts.tsv"
    ["OCR"]="../feature_selection/OCR/vst_counts.tsv"
)

for DATA_TYPE in "${!DATA_FILES[@]}"; do
    INPUT_FILE="${DATA_FILES[$DATA_TYPE]}"
    OUTPUT_DIR="${DATA_TYPE}"
    mkdir -p "${OUTPUT_DIR}"

    if [ ! -f "$INPUT_FILE" ]; then
        echo "Warning: Input file not found for $DATA_TYPE: $INPUT_FILE"
        echo "Skipping $DATA_TYPE..."
        continue
    fi
    
    echo "Submitting job for $DATA_TYPE..."
    
    SBATCH_DIR="${PROJECT_ROOT}/machine_learning_tumor/sbatch_scripts"
    mkdir -p "$SBATCH_DIR"
    
    JOB_SCRIPT="${SBATCH_DIR}/run_${DATA_TYPE}.sh"
    
    cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH -J ML_${DATA_TYPE}
#SBATCH -N 1
#SBATCH -p normal
#SBATCH --cpus-per-task=48
#SBATCH -o ${OUTPUT_DIR}/%x_%j.out
#SBATCH -e ${OUTPUT_DIR}/%x_%j.err
#SBATCH --mem=320g

cd ${PROJECT_ROOT}/machine_learning_tumor

source activate py310

python ML_class.py \\
    -i ${INPUT_FILE} \\
    -m ${METADATA} \\
    -l ${LABEL} \\
    -o ${OUTPUT_DIR} \\
    -t ${DATA_TYPE} \\
    --n_jobs -1 \\
    --automl_n_jobs -1 \\
    --n_trials ${N_TRIALS} \\
    --inner_cv ${INNER_CV} \\
    --inner_repeats ${INNER_REPEATS} \\
    --outer_cv ${OUTER_CV} \\
    --outer_repeats ${OUTER_REPEATS} \\
    --max_iter ${MAX_ITER}

echo "Completed training for ${DATA_TYPE}"
EOF
    
    sbatch "$JOB_SCRIPT"
    echo "Submitted job for $DATA_TYPE"
    echo "---"
done

echo "All jobs submitted!"
