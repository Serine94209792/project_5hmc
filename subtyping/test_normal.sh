#! /bin/bash
#SBATCH -J normal
#SBATCH -N 1
#SBATCH -p normal
#SBATCH -n 48
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --mem 320g

source activate shyR
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dir="${PROJECT_ROOT:-${SCRIPT_DIR}}"
cd "$dir" || {
  echo "ERROR: cd failed: $dir" >&2
  exit 1
}

Rscript plot_integrated_trajectory_heatmaps.R \
  --base_dir "$dir" \
  --datatypes gene_counts,OCR,window,consensus_peak

echo "plot_integrated_trajectory_heatmaps.R OK (4 datatypes). Outputs under: ${dir}plot/"
