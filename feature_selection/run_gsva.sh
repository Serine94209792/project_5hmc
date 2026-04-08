#!/bin/bash
#SBATCH -J gsva
#SBATCH -N 1
#SBATCH -p normal
#SBATCH --cpus-per-task=48
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --mem=320g

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ulimit -v/-m often not allowed on Slurm nodes; memory is set by #SBATCH --mem

source activate shyR

Rscript "${SCRIPT_DIR}/run_gsva.R" --base_dir "$SCRIPT_DIR"

echo "Completed GSVA; output: ${SCRIPT_DIR}/gsva_scores.tsv"
