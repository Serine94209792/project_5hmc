#!/bin/bash
#SBATCH -J tumorlikeness_generate
#SBATCH -N 1
#SBATCH -p normal
#SBATCH --cpus-per-task=48
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --mem=320g

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
FEATURE_SELECTION_DIR="${BASE_DIR}/feature_selection"
TUMOR_LIKENESS_DIR="${BASE_DIR}/tumor_likeness"
METADATA_FILE="${BASE_DIR}/cfDNA_metadata2_TNM.csv"

# 各 datatype 的 pvalue_threshold 与 n_components，格式: "pvalue_threshold:n_components"
declare -A DATATYPE_PARAMS=(
    ["artemis"]="0.05:3"
    ["end_motif"]="0.05:3"
    ["FSD"]="0.05:2"
    ["OCR"]="0.05:3"
    ["gene_counts"]="0.05:3"
    ["consensus_peak"]="0.05:3"
    ["window"]="0.005:3"
)

if [ ! -f "$METADATA_FILE" ]; then
    echo "Error: Metadata file not found: $METADATA_FILE"
    exit 1
fi

TEMP_SCRIPT_DIR="${TUMOR_LIKENESS_DIR}/sbatch_scripts"
mkdir -p "$TEMP_SCRIPT_DIR"

for datatype in artemis end_motif FSD OCR gene_counts consensus_peak window; do
    echo "=========================================="
    echo "Processing datatype: $datatype"
    echo "=========================================="

    input_dir="${FEATURE_SELECTION_DIR}/${datatype}"
    output_dir="${TUMOR_LIKENESS_DIR}/${datatype}"

    if [ ! -d "$input_dir" ]; then
        echo "Warning: Input directory not found for $datatype: $input_dir"
        echo "Skipping $datatype..."
        continue
    fi

    if [ ! -f "${input_dir}/feature_analysis_matrix.tsv" ]; then
        echo "Warning: feature_analysis_matrix.tsv not found in ${input_dir}"
        echo "Skipping $datatype..."
        continue
    fi

    params="${DATATYPE_PARAMS[$datatype]}"
    if [ -z "$params" ]; then
        echo "Warning: Parameters not defined for $datatype"
        echo "Skipping $datatype..."
        continue
    fi

    IFS=':' read -r pvalue_threshold n_components <<< "$params"
    mkdir -p "$output_dir"

    sbatch_script="${TEMP_SCRIPT_DIR}/generate_${datatype}.sh"
    cat > "$sbatch_script" << EOF
#!/bin/bash
#SBATCH -J tumorlikeness_${datatype}
#SBATCH -N 1
#SBATCH -p normal
#SBATCH --cpus-per-task=48
#SBATCH -o ${output_dir}/%x_%j.out
#SBATCH -e ${output_dir}/%x_%j.err
#SBATCH --mem=320g

cd "$TUMOR_LIKENESS_DIR"
source activate py310

python ${TUMOR_LIKENESS_DIR}/generate_diff_features.py \\
    -i "$input_dir" \\
    -t $datatype \\
    -m "$METADATA_FILE" \\
    -o "$output_dir" \\
    -p $pvalue_threshold \\
    -n $n_components

echo "Completed generate diff features for ${datatype}"
EOF

    chmod +x "$sbatch_script"
    echo "Generated sbatch script for $datatype:"
    echo "  Input dir:  $input_dir"
    echo "  Output dir: $output_dir"
    echo "  pvalue_threshold: $pvalue_threshold"
    echo "  n_components: $n_components"
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
