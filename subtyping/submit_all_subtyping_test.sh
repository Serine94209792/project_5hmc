#!/bin/bash
#SBATCH -J trajectory_test
#SBATCH -N 1
#SBATCH -p normal
#SBATCH --cpus-per-task=48
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --mem=320g


# 可选环境变量（在运行前 export）:
#   PAIRWISE_LIST        空格分隔，默认 "S00_vs_S10 S00_vs_S01"；可加入 S00_vs_S11 或改为单项
#   TEST_TOP_N           传给 step3 的 --top-n，默认 100
#   PROJECT_ROOT         工程根目录（与主脚本一致时为 HPC 路径）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SUBTYPING_DIR="${PROJECT_ROOT}/subtyping"
SBATCH_DIR="${SUBTYPING_DIR}/sbatch_scripts_test"
METADATA="${PROJECT_ROOT}/cfDNA_metadata2_TNM.csv"

# 默认：S00 vs S10 与 S00 vs S01 都跑
PAIRWISE_LIST="${PAIRWISE_LIST:-S00_vs_S10 S00_vs_S01}"
TEST_TOP_N="${TEST_TOP_N:-100}"

validate_pairwise() {
  case "$1" in
    S00_vs_S10|S00_vs_S01|S00_vs_S11) return 0 ;;
    *)
      echo "Error: invalid pairwise in PAIRWISE_LIST: $1 (allowed: S00_vs_S10 S00_vs_S01 S00_vs_S11)" >&2
      return 1
      ;;
  esac
}

for pw in ${PAIRWISE_LIST}; do
  validate_pairwise "${pw}" || exit 1
done

N_PAIRWISE=0
for _ in ${PAIRWISE_LIST}; do
  N_PAIRWISE=$((N_PAIRWISE + 1))
done

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

N_DATATYPE="${#DATA_FILES[@]}"
PLAN_MAX=$((N_DATATYPE * N_PAIRWISE))

mkdir -p "${SBATCH_DIR}"

echo "================================================================"
echo "submit_all_subtyping_test"
echo "  PROJECT_ROOT=${PROJECT_ROOT}"
echo "  PAIRWISE_LIST=${PAIRWISE_LIST}  (每项 1 个作业 / datatype)"
echo "  TEST_TOP_N=${TEST_TOP_N}"
echo "----------------------------------------------------------------"
echo "  配置的 datatype 数: ${N_DATATYPE}"
echo "  pairwise 数:        ${N_PAIRWISE}"
echo "  矩阵齐全时计划提交: ${N_DATATYPE} × ${N_PAIRWISE} = ${PLAN_MAX} 个作业"
echo "================================================================"
echo ""

SUBMITTED=0
SKIPPED_DATATYPES=()

for DATA_TYPE in "${!DATA_FILES[@]}"; do
  MATRIX_PATH="${DATA_FILES[$DATA_TYPE]}"

  if [ ! -f "${MATRIX_PATH}" ]; then
    echo "[跳过] datatype=${DATA_TYPE}"
    echo "       原因: 矩阵文件不存在"
    echo "       路径: ${MATRIX_PATH}"
    echo "       提示: 该 datatype 将少提交 ${N_PAIRWISE} 个作业（每个 pairwise 各 1 个）。"
    echo "       处理: 生成或同步矩阵后重新运行本脚本。"
    echo "---"
    SKIPPED_DATATYPES+=("${DATA_TYPE}")
    continue
  fi

  for PAIRWISE_COMPARISON in ${PAIRWISE_LIST}; do
    # 与 run_test_pipeline_steps1_4 默认输出目录一致: subtyping/<datatype>_test_<pairwise>
    LOG_DIR="${SUBTYPING_DIR}/${DATA_TYPE}_test_${PAIRWISE_COMPARISON}"
    mkdir -p "${LOG_DIR}"

    echo "Preparing subtyping TEST job for ${DATA_TYPE} (pairwise=${PAIRWISE_COMPARISON})..."

    PW_SHORT="${PAIRWISE_COMPARISON//_vs_/v}"
    JOB_SCRIPT="${SBATCH_DIR}/run_test_${DATA_TYPE}_${PAIRWISE_COMPARISON}.sh"

    cat > "${JOB_SCRIPT}" <<EOF
#!/bin/bash
#SBATCH -J ST_${DATA_TYPE}_${PW_SHORT}
#SBATCH -N 1
#SBATCH -p normal
#SBATCH --cpus-per-task=48
#SBATCH --mem=320g
#SBATCH -o ${LOG_DIR}/%x_%j.out
#SBATCH -e ${LOG_DIR}/%x_%j.err

set -euo pipefail

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

"${DEFAULT_PYTHON}" -m subtyping.run_test_pipeline_steps1_4 \\
  --datatype "${DATA_TYPE}" \\
  --matrix "${MATRIX_PATH}" \\
  --metadata "${METADATA}" \\
  --subtyping-root "subtyping" \\
  --rscript "${DEFAULT_RSCRIPT}" \\
  --python-bin "${DEFAULT_PYTHON}" \\
  --pairwise-comparison "${PAIRWISE_COMPARISON}" \\
  --top-n ${TEST_TOP_N}

echo "Completed subtyping TEST (steps 1-4) for ${DATA_TYPE} pairwise=${PAIRWISE_COMPARISON}"
EOF

    chmod +x "${JOB_SCRIPT}"
    sbatch "${JOB_SCRIPT}"
    SUBMITTED=$((SUBMITTED + 1))
    echo "Submitted ${JOB_SCRIPT}"
    echo "---"
  done
done

echo "================================================================"
echo "汇总"
echo "  本次 sbatch 提交作业数: ${SUBMITTED}"
if [ "${#SKIPPED_DATATYPES[@]}" -gt 0 ]; then
  echo "  因矩阵缺失跳过的 datatype (${#SKIPPED_DATATYPES[@]} 个): ${SKIPPED_DATATYPES[*]}"
  echo "  上述每个跳过的 datatype 少 ${N_PAIRWISE} 个作业；矩阵齐全时本批最多为 ${PLAN_MAX} 个作业。"
else
  echo "  无矩阵缺失；与计划一致时为 ${PLAN_MAX} 个作业 (${N_DATATYPE}×${N_PAIRWISE})。"
fi
echo "  PAIRWISE_LIST=${PAIRWISE_LIST}"
echo "================================================================"
