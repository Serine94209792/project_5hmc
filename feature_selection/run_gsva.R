#!/usr/bin/env Rscript
# GSVA for 6 datatypes: each datatype = one gene set from feature_analysis_matrix.tsv.
# Reads all 6, runs GSVA per datatype, merges to one sample x datatype matrix.
# Usage: Rscript run_gsva.R --base_dir <feature_selection_dir> [--output <path>]

if (!requireNamespace("optparse", quietly = TRUE)) stop("install optparse: install.packages('optparse')")
library(optparse)
if (!requireNamespace("readr", quietly = TRUE)) stop("install readr: install.packages('readr')")
if (!requireNamespace("GSVA", quietly = TRUE)) stop("install GSVA: BiocManager::install('GSVA')")
library(GSVA)

# readr avoids data.table segfault on this HPC; handles quoted fields / uneven columns
read_tsv <- function(path) {
  as.data.frame(readr::read_tsv(path, progress = FALSE))
}

option_list <- list(
  make_option(c("--base_dir"), type = "character", default = ".",
              help = "Feature selection root directory (default: .)"),
  make_option(c("--output"), type = "character", default = NA,
              help = "Output TSV path (default: <base_dir>/gsva_scores.tsv)")
)
opt <- parse_args(OptionParser(option_list = option_list))
base_dir <- normalizePath(opt$base_dir, mustWork = FALSE)
out_file <- if (is.na(opt$output) || opt$output == "") file.path(base_dir, "gsva_scores.tsv") else opt$output

DATATYPES <- c("artemis", "FSD", "OCR", "gene_counts", "consensus_peak", "window", "end_motif")

# For each datatype: read expr + feature set, run GSVA, return named vector of scores (sample -> score)
score_list <- list()
sample_sets <- list()

for (datatype in DATATYPES) {
  if (datatype == "artemis") {
    expr_file <- file.path(base_dir, "artemis", "artemis_log1p.tsv")
  } else if (datatype == "end_motif") {
    expr_file <- file.path(base_dir, "end_motif", "end_motif_matrix.tsv")
  } else {
    expr_file <- file.path(base_dir, datatype, "vst_counts.tsv")
  }
  fam_file <- file.path(base_dir, datatype, "feature_analysis_matrix.tsv")

  if (!file.exists(expr_file)) {
    warning("Expression matrix not found for ", datatype, ": ", expr_file, " — skipping")
    next
  }
  if (!file.exists(fam_file)) {
    warning("feature_analysis_matrix not found for ", datatype, ": ", fam_file, " — skipping")
    next
  }

  dt <- read_tsv(expr_file)
  sample_ids_expr <- as.character(dt[[1L]])
  expr_wide <- as.matrix(dt[, -1L, drop = FALSE])
  rownames(expr_wide) <- sample_ids_expr

  fam_dt <- read_tsv(fam_file)
  feature_set <- as.character(fam_dt[[1L]])

  # 基因集仅用 feature_analysis_matrix 中的特征；保证这些特征都在表达矩阵中（不取交集缩小表达矩阵）
  feature_set_in_expr <- feature_set[feature_set %in% colnames(expr_wide)]
  if (length(feature_set_in_expr) < length(feature_set)) {
    warning("Datatype ", datatype, ": ", length(feature_set) - length(feature_set_in_expr),
            " features in feature_analysis_matrix are missing from expression matrix — dropped from gene set")
  }
  if (length(feature_set_in_expr) == 0) {
    warning("No feature_analysis_matrix features found in expression matrix for ", datatype, " — skipping")
    next
  }

  expr_t <- t(expr_wide)
  gset <- list(set1 = feature_set_in_expr)

  param <- gsvaParam(exprData = expr_t, geneSets = gset, kcdf = "Gaussian")
  scores <- gsva(param, verbose = FALSE)
  if (inherits(scores, "SummarizedExperiment")) scores <- SummarizedExperiment::assay(scores)
  vec <- as.numeric(scores[1, ])
  names(vec) <- colnames(scores)
  score_list[[datatype]] <- vec
  sample_sets[[datatype]] <- names(vec)
}

if (length(score_list) == 0) stop("No datatype could be processed; check inputs.")

# Align by sample: use intersection of all samples so each row has all datatypes
common_samples <- Reduce(intersect, sample_sets)
if (length(common_samples) == 0) stop("No common samples across datatypes.")

out_mat <- matrix(NA_real_, nrow = length(common_samples), ncol = length(score_list),
                  dimnames = list(common_samples, names(score_list)))
for (d in names(score_list)) {
  out_mat[, d] <- score_list[[d]][common_samples]
}

out_df <- as.data.frame(out_mat)
out_df <- cbind(sample = rownames(out_df), out_df, stringsAsFactors = FALSE)
rownames(out_df) <- NULL

dir.create(dirname(out_file), recursive = TRUE, showWarnings = FALSE)
write.table(out_df, out_file, sep = "\t", row.names = FALSE, quote = FALSE)
message("Written ", nrow(out_df), " x ", ncol(out_df) - 1L, " matrix to ", out_file)
