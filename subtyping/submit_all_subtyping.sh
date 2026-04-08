#!/bin/bash
# Batch submit subtyping pipeline jobs for multiple datatypes

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SUBTYPING_DIR="${PROJECT_ROOT}/subtyping"
SBATCH_DIR="${SUBTYPING_DIR}/sbatch_scripts"
METADATA="${PROJECT_ROOT}/cfDNA_metadata2_TNM.csv"

# HPC environments
DEFAULT_RSCRIPT="${DEFAULT_RSCRIPT:-Rscript}"
DEFAULT_PYTHON="${DEFAULT_PYTHON:-python}"

declare -A DATA_FILES=(
    ["artemis"]="${PROJECT_ROOT}/feature_selection/artemis/artemis_log1p.tsv"
    ["FSD"]="${PROJECT_ROOT}/feature_selection/FSD/vst_counts.tsv"
    ["gene_counts"]="${PROJECT_ROOT}/feature_selection/gene_counts/vst_counts.tsv"
    ["consensus_peak"]="${PROJECT_ROOT}/feature_selection/consensus_peak/vst_counts.tsv"
    ["end_motif"]="${PROJECT_ROOT}/feature_selection/end_motif/end_motif_matrix.tsv"
    ["window"]="${PROJECT_ROOT}/feature_selection/window/vst_counts.tsv"
    ["OCR"]="${PROJECT_ROOT}/feature_selection/OCR/vst_counts.tsv"
)

mkdir -p "${SBATCH_DIR}"

for DATA_TYPE in "${!DATA_FILES[@]}"; do
    MATRIX_PATH="${DATA_FILES[$DATA_TYPE]}"
    TARGET_DIR="${SUBTYPING_DIR}/${DATA_TYPE}"
    mkdir -p "${TARGET_DIR}"

    if [ ! -f "${MATRIX_PATH}" ]; then
        echo "Warning: matrix not found for ${DATA_TYPE}: ${MATRIX_PATH}"
        echo "Skipping ${DATA_TYPE}..."
        echo "---"
        continue
    fi

    echo "Preparing subtyping job for ${DATA_TYPE}..."

    JOB_SCRIPT="${SBATCH_DIR}/run_${DATA_TYPE}.sh"
    LOG_DIR="${SUBTYPING_DIR}/${DATA_TYPE}"

    cat > "${JOB_SCRIPT}" <<EOF
#!/bin/bash
#SBATCH -J ST_${DATA_TYPE}
#SBATCH -N 1
#SBATCH -p normal
#SBATCH --cpus-per-task=48
#SBATCH --mem=320g
#SBATCH -o ${LOG_DIR}/%x_%j.out
#SBATCH -e ${LOG_DIR}/%x_%j.err

set -euo pipefail

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

"${DEFAULT_PYTHON}" -m subtyping.run_pipeline \\
  --datatype "${DATA_TYPE}" \\
  --matrix "${MATRIX_PATH}" \\
  --metadata "${METADATA}" \\
  --subtyping-root "subtyping" \\
  --rscript "${DEFAULT_RSCRIPT}" \\
  --python-bin "${DEFAULT_PYTHON}"

echo "Completed subtyping for ${DATA_TYPE}"
EOF

    chmod +x "${JOB_SCRIPT}"
    sbatch "${JOB_SCRIPT}"
    echo "Submitted ${JOB_SCRIPT}"
    echo "---"
done

echo "All subtyping jobs submitted."
