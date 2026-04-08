args <- commandArgs(trailingOnly = TRUE)
parse_args <- function(args) {
  out <- list()
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, '--')) stop(paste('Unexpected argument:', key))
    val <- args[[i + 1]]
    out[[substring(key, 3)]] <- val
    i <- i + 2
  }
  out
}
opt <- parse_args(args)
required <- c('input_matrix', 'annotation', 'output_matrix', 'output_overlap')
missing <- required[!required %in% names(opt)]
if (length(missing) > 0) stop(paste('Missing required args:', paste(missing, collapse=', ')))

# HPC/NFS 上 data.table::fread 可能 mmap 导致 segfault（invalid permissions），无法用 tryCatch 捕获。
# 默认：readr（较快、通常安全）> utils::read.delim。
# 仅在本地盘等确定安全的环境设 STEP2_USE_FREAD=1 使用 fread（需 data.table）。
read_tsv_safe <- function(path) {
  if (nzchar(Sys.getenv('STEP2_USE_FREAD'))) {
    if (!requireNamespace('data.table', quietly = TRUE)) {
      stop('STEP2_USE_FREAD 需要 data.table: install.packages("data.table")')
    }
    args <- list(
      file = path,
      sep = '\t',
      check.names = FALSE,
      data.table = FALSE,
      showProgress = FALSE,
      na.strings = c('', 'NA'),
      nThread = 1L
    )
    if ('mmap' %in% names(formals(data.table::fread))) {
      args$mmap <- FALSE
    }
    return(do.call(data.table::fread, args))
  }
  if (requireNamespace('readr', quietly = TRUE)) {
    rt_args <- list(file = path, show_col_types = FALSE, progress = FALSE)
    if ('name_repair' %in% names(formals(readr::read_tsv))) {
      rt_args$name_repair <- 'minimal'
    }
    df <- do.call(readr::read_tsv, rt_args)
    return(as.data.frame(df))
  }
  utils::read.delim(path, check.names = FALSE)
}

# 写出：默认 readr 或 write.table；设 STEP2_USE_FWRITE=1 时用 data.table::fwrite（需 data.table）。
write_tsv_safe <- function(df, path) {
  if (nzchar(Sys.getenv('STEP2_USE_FWRITE'))) {
    if (!requireNamespace('data.table', quietly = TRUE)) {
      stop('STEP2_USE_FWRITE 需要 data.table')
    }
    data.table::fwrite(df, file = path, sep = '\t', quote = FALSE, row.names = FALSE, nThread = 1L)
    return(invisible(NULL))
  }
  if (requireNamespace('readr', quietly = TRUE)) {
    readr::write_tsv(df, path, progress = FALSE)
    return(invisible(NULL))
  }
  utils::write.table(df, file = path, sep = '\t', quote = FALSE, row.names = FALSE)
  invisible(NULL)
}

matrix_df <- read_tsv_safe(opt$input_matrix)
ann <- read_tsv_safe(opt$annotation)

sample_col <- if ('sample' %in% colnames(matrix_df)) 'sample' else colnames(matrix_df)[1]
colnames(matrix_df)[colnames(matrix_df) == sample_col] <- 'sample'
matrix_df$sample <- as.character(matrix_df$sample)
ann$sample <- as.character(ann$sample)

ann_use <- ann[ann$include_main %in% c(TRUE, 'True', 'TRUE', 1), c('sample', 'state', 'batch_group')]
overlap <- ann_use[ann_use$sample %in% matrix_df$sample, , drop=FALSE]
if (nrow(overlap) == 0) stop('No overlapping samples between annotation and matrix.')

matrix_use <- matrix_df[match(overlap$sample, matrix_df$sample), , drop=FALSE]
rownames(matrix_use) <- matrix_use$sample
expr <- as.matrix(matrix_use[, setdiff(colnames(matrix_use), 'sample'), drop=FALSE])
storage.mode(expr) <- 'numeric'

batch <- factor(overlap$batch_group)
corrected <- limma::removeBatchEffect(t(expr), batch=batch)
corrected <- t(corrected)

out_df <- data.frame(sample=rownames(corrected), state=overlap$state, corrected, check.names=FALSE)
write_tsv_safe(out_df, opt$output_matrix)

overlap$batch_method <- 'limma::removeBatchEffect'
overlap$feature_selection_method <- 'none'
overlap$feature_selection_top_n <- NA_integer_
write_tsv_safe(overlap, opt$output_overlap)
cat('Saved full batch-corrected matrix to', opt$output_matrix, 'without HVG/MAD filtering\n')
