# project_5hmc

This repository contains a research-oriented cfDNA 5hmC analysis workflow. It combines raw sequencing preprocessing, feature generation, visualization, biological interpretation, and downstream machine learning for tumor detection, stage prediction, molecular subtyping, and tissue-of-origin analysis.

## Repository Overview

### `5hmc.smk`

This is the main Snakemake pipeline and the best entry point for understanding the core data-processing workflow. It defines the end-to-end analysis from raw FASTQ files to processed feature matrices and QC outputs.

Major analysis blocks in this file include:

- raw read QC and trimming
- alignment and BAM post-processing
- window-based signal quantification
- gene body counting
- fragment size distribution analysis
- end motif analysis
- consensus peak generation and counting
- ChromHMM enrichment and Jaccard-based enrichment
- OCR counting
- QC plots and coverage plots
- tissue-of-origin related intermediate outputs


### `config.yaml`

This file defines the main runtime configuration for the Snakemake workflow.

### `cfDNA_metadata2_TNM.csv`

This file provide sample-level metadata used throughout downstream analyses. The `cfDNA_metadata2_TNM.csv` is important for label assignment, stage information, and subtyping-related workflows.

Notes for encoded columns in `cfDNA_metadata2_TNM.csv`:
- For binary columns, `0` indicates absence and `1` indicates presence.
- In the `type` column, `0` represents non-PDAC and `1` represents PDAC.
- For the sex column, `0` represents female and `1` represents male.
- In the `differentiation` column, a higher numeric value indicates a lower degree of differentiation.

## Directory Guide

### `feature_selection/`

This directory contains scripts used to derive differential or biologically informative features from upstream count matrices.

Representative tasks include:

- differential analysis
- feature filtering
- GSVA-related processing
- generation of selected feature tables for later modules

This directory is important because several later analyses depend on the selected features generated here.

### `subtyping/`

This directory contains a dedicated molecular subtyping workflow. The main orchestrator is [`run_pipeline.py`], which links together several numbered steps:

- build state annotation from metadata
- preprocess one datatype matrix
- run differential analysis
- infer diffusion pseudotime
- identify pseudotime-associated genes/features

The numbered scripts (`step1_...` to `step5_...`) make the subtyping logic easier to inspect step by step. 

### `tumor_likeness/`

This directory contains scripts for stage-associated feature generation(NMF program). 

### `model/`

This directory contains the downstream `machine_learning_tumor/` workflow for PDAC vs benign design. The driver script is [`run_model.sh`], which performs three main steps:

- integrate multidimensional 5hmC features into one aggregate table
- train a second-level model, searching across featuretype combinations
- evaluate the final model and generate summary plots

This directory is second-level detection model.

### `model_stage/`

This directory mirrors the structure of `model/`, but is intended for stage prediction. It combines selected featuretype outputs from `machine_learning_stage/`, trains stage-related models, and writes evaluation results into `model_stage/ML_results/`.

This directory is second-level stage model.

### `model_subtyping/`

This directory contains the downstream `machine_learning_subtyping/` workflow for subtype classification. As with the other model directories, it integrates feature tables, trains a model, and evaluates performance.

This directory is second-level subtyping model.

### `machine_learning_tumor/`

This directory stores first-level detection model

### `machine_learning_stage/`

This directory stores first-level stage model

### `machine_learning_subtyping/`

This directory stores first-level subtying model

### `too/`

This directory contains tissue-of-origin analysis code. The main script, [`run_too.py`], merges tissue peak sets, computes fragment coverage over those peaks, and summarizes tissue-level scores, percentages, and variability.

Typical outputs include:

- tissue score matrix
- tissue percentage matrix
- tissue CV table
- replicate stability plot


### `OCR/`

This directory contains scripts for open chromatin region counting. For example, [`OCR_count.py`] computes per-sample fragment coverage over OCR regions and merges all samples into a single table.

This is one of the 5hmC feature types that later feeds downstream differential analysis and modeling.

### `FSD/`

This directory contains fragment size distribution related resources and outputs. It supports genome bin construction, GC/mappability-aware filtering, and generation of FSD feature matrices.

This is one of the 5hmC feature types that later feeds downstream differential analysis and modeling.

### `end_motif/`

This directory contains end motif analysis and output sample-motif proportion matrix. 

This is one of the 5hmC feature types that later feeds downstream differential analysis and modeling.

### `consensus_peak/`

This directory contains consensus peak generation and peak count matrix. 

This is one of the 5hmC feature types that later feeds downstream differential analysis and modeling.

### `gene_counts/`

This directory stores gene body counting outputs and filtered count matrices. 

This is one of the 5hmC feature types that later feeds downstream differential analysis and modeling.

### `window/`

This directory stores window-based signal quantification outputs, including genome windows and per-sample window count matrix.

This is one of the 5hmC feature types that later feeds downstream differential analysis and modeling.

### `enrichment/`

This directory contains enrichment analyses such as Jaccard similarity and Fisher-based comparisons, typically used to compare selected features against annotated genomic categories.

### `chromHMM/`

This directory contains ChromHMM-related annotation and enrichment outputs. It helps interpret selected genomic regions in terms of chromatin state categories.

### `coverage_plots/` and `QC_plots/`

These directories collect quality-control and visualization outputs, including PCA, correlation matrices, GC bias plots, coverage heatmaps, and line plots. They are mainly used for sanity checking sample quality and global signal structure.

### `figures/`

This directory stores figures in manuscript

### `tables/`

This directory stores tables in manuscript

### `ref/`

This directory stores reference files required by the workflow, such as blacklist regions and OCR references. However, some files are too large so it will not be stored in the repo. 

