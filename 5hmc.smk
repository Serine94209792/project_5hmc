####指定配置文件
configfile: "config.yaml"

####导入样本
####samples.txt每一行都为一个样本名称
samples_file=config["samples_file"]
SAMPLES=[]
fr=open(samples_file,"r")
for i in fr.readlines():
    s=i.strip()
    SAMPLES.append(s)
fr.close()
label=config["label"]
CHRS = ["chr" + str(i) for i in range(1, 23)] + ["chrX"]
# QUALIMAP_JAVA_MEM = config.get("qualimap_java_mem", "1G")

def get_ref_peak_files(wildcards):
    import glob
    return sorted(glob.glob(config["ref_peak_dir"] + "/" + config["ref_peak_glob"]))

# ----指定收集的目标文件，记住所有位置参数要在关键字参数之前-----
rule all:   
    input:
        # ====================================================================
        # 1. 数据质控和预处理
        # ====================================================================
        raw_qc_html=expand("rawqc/{sample}_{i}_fastqc.html",sample=SAMPLES, i=[1,2]),
        fastpgz=expand("clean/{sample}_{i}.fastq.gz",sample=SAMPLES, i=[1,2]),
        clean_qc_html=expand("cleanqc/{sample}_{i}_fastqc.html",sample=SAMPLES, i=[1,2]),
        
        # ====================================================================
        # 2. 序列比对和BAM文件处理
        # ====================================================================
        bwa_map=expand("map/{sample}.raw.bam",sample=SAMPLES),
        map_qc=expand("mapqc/{sample}/qualimapReport.html",sample=SAMPLES),
        rm_dup=expand("rmdup/{sample}.rmdup.bam",sample=SAMPLES),
        last_bam=expand("last/{sample}.last.bam",sample=SAMPLES),
        last_qc=expand("lastqc/{sample}/qualimapReport.html",sample=SAMPLES),
        paired_bam=expand("paired_bam/{sample}.paired.bam",sample=SAMPLES),
        paired_bam_stat=expand("paired_bam/{sample}.paired.stat",sample=SAMPLES),
        sample_stats_matrix="statistic/sample_stats_matrix.csv",

        # ====================================================================
        # 3. Window 分析
        # ====================================================================
        window="window/make_window/windows.bed",
        filtered_window="window/make_window/filtered_windows.bed",
        window_count=expand("window/window_count/{sample}_window_count.bed",sample=SAMPLES),
        window_count_merge="window/window_count/window_count_merged.tsv",
        final_window_count="window/window_count/final_window_count.tsv",


        # ====================================================================
        # 4. Genebody 分析
        # ====================================================================
        genebody_counts="gene_counts/genebody_counts.txt",
        filtered_counts="gene_counts/filtered_genebody_counts.tsv",

        # ====================================================================
        # 7. FSD 分析
        # ====================================================================
        bins="FSD/hg38_bins.bed",
        GC_content="FSD/hg38_bin_GC.bed",
        map_bins="FSD/hg38_bin_mappability.bed",
        gc_map="FSD/hg38_bin_GC_map.bed",
        filtered_bins="FSD/hg38_filtered_bins.bed",
        fragments=expand("fragments/{sample}_fragments.bed",sample=SAMPLES),
        fsd="FSD/fsd.tsv",
        fsd_annotation="FSD/fsd_annotation.tsv",

        # ====================================================================
        # 8. End motif 分析
        # ====================================================================
        end_motif="end_motif/end_motif_freq.tsv",

        # ====================================================================
        # 10. Consensus peak 分析
        # ====================================================================
        peaks=expand("peaks/{sample}_peaks.narrowPeak",sample=SAMPLES),
        consensus_peak=expand("consensus_peak/{label}_consensus_peak.bed",label=["0","1"]),
        merged_consensus_peak="consensus_peak/merged_consensus_peak.bed",
        peak_counts=expand("consensus_peak/peak_counts/{sample}_counts.bed",sample=SAMPLES),
        peak_count_merged="consensus_peak/peak_count_merged.tsv",
        filtered_peak_counts="consensus_peak/filtered_peak_counts.tsv",

        # ====================================================================
        # 12. ChromHMM 注释 （需要先执行feature_selection）
        # ====================================================================
        enrichment_plot="chromHMM/enrichment_plot.png",
        enrichment_table="chromHMM/enrichment_table.tsv",

        # ====================================================================
        # 13. QC 和可视化分析
        # ====================================================================
        # BigWig 文件
        bw=expand("bw/{sample}.bw",sample=SAMPLES),
        
        # BigWig 矩阵和相关性分析
        bw_matrix="QC_plots/bw_matrix.npz",
        raw_counts="QC_plots/raw_counts.tsv",
        PCA_plot="QC_plots/PCA_plot.png",
        PCA_matrix="QC_plots/PCA_matrix.tsv",
        correlation_plot="QC_plots/correlation_plot.png",
        correlation_matrix="QC_plots/correlation_matrix.tsv",
        
        # GC bias 分析
        GC_bias_plot=expand("QC_plots/{sample}/{sample}_GC_bias_plot.png",sample=SAMPLES),
        GC_bias_freq=expand("QC_plots/{sample}/{sample}_GC_bias_freq.txt",sample=SAMPLES),
        
        # Coverage 图
        coverage_plot=expand("QC_plots/{sample}/{sample}_coverage_plot.png",sample=SAMPLES),
        
        # 每个样本的 Coverage 矩阵
        gz_matrix=expand("coverage_plots/{sample}/{i}.gz",sample=SAMPLES, i=["genebody","enhancer"]),
        raw_matrix=expand("coverage_plots/{sample}/{i}.raw.tsv",sample=SAMPLES, i=["genebody","enhancer"]),
        sorted_regions=expand("coverage_plots/{sample}/sorted_{i}.bed",sample=SAMPLES, i=["genebody","enhancer"]),
        
        # 每个样本的 heatmap 和 lineplot 图
        plot_sample=expand("coverage_plots/{sample}/{i}_{type}.png",sample=SAMPLES, i=["genebody","enhancer"], type=["heatmap","lineplot"]),
        
        # Type-specific 矩阵
        merged_matrix=expand("coverage_plots/type_specific/{i}_merge_{label}.tsv",i=["genebody","enhancer"], label=["0","1"]),
        merged_matrix_all=expand("coverage_plots/type_specific/{i}_merge_all.tsv", i=["genebody", "enhancer"]),
        
        # Type-specific heatmap and lineplot
        plot_groups=expand("coverage_plots/type_specific/{i}_{type}.png",i=["genebody","enhancer"], type=["heatmap","lineplot"]),
        plot_groups_all=expand("coverage_plots/type_specific/{i}_{type}_all.png", i=["genebody", "enhancer"], type=["heatmap", "lineplot"]),
        
        # ====================================================================
        # 15. Jaccard 相似系数分析（需要先执行feature_selection）
        # ====================================================================
        jaccard_matrix="enrichment/jaccard_matrix.tsv",
        jaccard_heatmap="enrichment/jaccard_heatmap.png",
        fisher_ratio_matrix="enrichment/fisher_ratio_matrix.tsv",
        fisher_pvalue_matrix="enrichment/fisher_pvalue_matrix.tsv",
        fisher_heatmap="enrichment/fisher_heatmap.png",

        
        # ====================================================================
        # 14. OCR 分析
        # ====================================================================
        unmapped="ref/hg38/ocr.unmapped.bed",
        filtered="ref/hg38/tumor_ocr_filtered.bed",
        OCR_count="OCR/OCR_count.tsv",
        final_OCR_count="OCR/final_OCR_count.tsv",

        # ====================================================================
        # 17. TOO (Tissue of Origin) 分析
        # ====================================================================
        too_tissue_score="too/tissue_score.tsv",
        too_tissue_cv="too/tissue_CV.tsv",
        too_replicate_plot="too/tissue_replicate_deviation.png",
        too_tissue_percent="too/tissue_percent.tsv",

# ----end all-----

# ----处理前qc -----
rule raw_qc:
    input:
        r1=config["r1_raw"],
        r2=config["r2_raw"]
    output:
        html1="rawqc/{sample}_1_fastqc.html",
        html2="rawqc/{sample}_2_fastqc.html"
    params:
        "rawqc/"
    shell:
        """
        fastqc {input.r1} {input.r2} -o {params} -t 10
        """

# ----end raw_qc-----

# ----质控和修剪接头-----
rule fastp:
    input:
        r1=config["r1_raw"],
        r2=config["r2_raw"]
    output:
        r1="clean/{sample}_1.fastq.gz",
        r2="clean/{sample}_2.fastq.gz",
        html="clean/{sample}.fastp.html",
        json="clean/{sample}.fastp.json"
    log:"clean/{sample}.fastp.log"
    shell:
        """
        fastp \
        -i {input.r1} \
        -I {input.r2} \
        -o {output.r1} \
        -O {output.r2} \
        -h {output.html} \
        -j {output.json} \
        --thread 10 \
        --detect_adapter_for_pe --length_required 50 \
        1>{log} 2>&1
        """   
# ----end fastp-----

# ----处理后qc-----
rule clean_qc:
    input:
        r1="clean/{sample}_1.fastq.gz",
        r2="clean/{sample}_2.fastq.gz"
    output:
        html1="cleanqc/{sample}_1_fastqc.html",
        html2="cleanqc/{sample}_2_fastqc.html"
    params:
        "cleanqc/"
    shell:
        """
        fastqc {input.r1} {input.r2} -o {params} -t 10 
        """

# ----end clean_qc-----

# ----序列比对,比对之前记得准备好参考基因组索引bwa index.fa-----
rule bwa_map:
    input:
        r1="clean/{sample}_1.fastq.gz",
        r2="clean/{sample}_2.fastq.gz",
        genome=config["genome"]
    output:
        bam="map/{sample}.raw.bam",
        index="map/{sample}.raw.bam.bai",
        stat="map/{sample}.raw.stat"
    params:
        "{sample}"
    log:
        "map/{sample}.bwa_map.log"
    shell:
        """
        bwa mem -M -t 20 \
        -R '@RG\\tID:{params}\\tSM:{params}\\tLB:WXS\\tPL:Illumina' \
        {input.genome} {input.r1} {input.r2} | \
        samtools sort -O bam -@ 1 -o {output.bam} \
        1>{log} 2>&1

        samtools index {output.bam} 
        samtools flagstat {output.bam} > {output.stat}
        """

# ----end bwa_map-----

# ----比对完qc-----
rule map_qc:
    input:
        bam="map/{sample}.raw.bam"
    output:
        mapqc="mapqc/{sample}/qualimapReport.html"
    params:
        outdir="mapqc/{sample}/",
        java_mem=QUALIMAP_JAVA_MEM
    shell:
        """
        qualimap bamqc -bam {input.bam} -outdir {params.outdir} --java-mem-size={params.java_mem}
        """

# ----end map_qc-----

# ----去除duplicate-----
rule rm_dup:
    input:
        bam="map/{sample}.raw.bam"
    output:
        rm="rmdup/{sample}.rmdup.bam",
        index="rmdup/{sample}.rmdup.bam.bai",
        stat="rmdup/{sample}.rmdup.stat"
    log:
        "rmdup/{sample}.rmdup.log"
    shell:
        """
        sambamba markdup -t 20 -r {input.bam} {output.rm} 1>{log} 2>&1
        samtools flagstat {output.rm} > {output.stat}
        """

# ----end rm_dup-----

# ----去除没有配对比对的reads，q<30的reads-----
rule rm_low:
    input:
        rm="rmdup/{sample}.rmdup.bam"
    output:
        last="last/{sample}.last.bam",
        index="last/{sample}.last.bam.bai",
        stat="last/{sample}.last.stat"
    shell:
        """
        samtools view -h -f 2 -q 30 {input.rm} | \
        samtools sort -O bam -@ 1 -o {output.last} 

        samtools index {output.last} 
        samtools flagstat {output.last} > {output.stat}
        """

# ----end rm_low-----

# ----最后再qc-----
rule last_qc:
    input:
        bam="last/{sample}.last.bam"
    output:
        mapqc="lastqc/{sample}/qualimapReport.html"
    params:
        outdir="lastqc/{sample}/",
        java_mem=QUALIMAP_JAVA_MEM
    shell:
        """
        qualimap bamqc -bam {input.bam} -outdir {params.outdir} --java-mem-size={params.java_mem}
        """
# ----end last_qc-----

#----make window and filter windows-----
rule make_window:
    input:
        genome_size=config["genome"] + ".genome",
    output:
        window="window/make_window/windows.bed"
    params:
        window_size=config["window_size"]
    log:
        "window/make_window.log"
    shell:
        """
        bedtools makewindows -g {input.genome_size} -w {params.window_size} \
        > {output.window} 2>{log}
        """

rule filter_windows:
    input:
        window="window/make_window/windows.bed",
        mappability="FSD/hg38_mappability/hg38.tmp.genmap.bedgraph",
    output:
        filtered_window="window/make_window/filtered_windows.bed"
    params:
        GC_threshold=config["GC_threshold"],
        mappability_threshold=config["mappability_threshold"],
        length_upper=config["length_upper"],
        length_lower=config["length_lower"],
        genome = config["genome"], 
        blacklist = config["blacklist"],
    log:
        "window/filter_windows.log"
    shell:
        """
        python window/filter_window.py \
        -i {input.window} \
        -m {input.mappability} \
        -gc {params.GC_threshold} \
        -mt {params.mappability_threshold} \
        -lu {params.length_upper} -ll {params.length_lower} \
        -ge {params.genome} -bl {params.blacklist} \
        -o {output.filtered_window} \
        1>{log} 2>&1
        """

# ----make window and filter windows-----

# ----count_to_window-----
rule count_to_window:
    input:
        fragments="fragments/{sample}_fragments.bed",
        filtered_window="window/make_window/filtered_windows.bed",
    output:
        window_count="window/window_count/{sample}_window_count.bed"
    shell:
        """
        bedtools coverage -a {input.filtered_window} -b {input.fragments} -counts > {output.window_count}
        """ 
# ----end count_to_window-----

# ----merge window count-----
rule merge_window_count:
    input:
        expand("window/window_count/{sample}_window_count.bed",sample=SAMPLES)
    output:
        window_count_merge="window/window_count/window_count_merged.tsv"
    shell:
        """
        python window/merge_count.py -i {input} -o {output.window_count_merge}
        """
# ----end merge_window_count-----

# ----get_final_window_count-----
rule get_final_window_count:
    input:
        window_count_merge="window/window_count/window_count_merged.tsv"
    output:
        final_window_count="window/window_count/final_window_count.tsv"
    params:
        variance_threshold=config["variance_threshold"],
        zero_threshold=config["zero_threshold"],
    log:
        "window/get_final_window_count.log"
    shell:
        """
        python window/get_final_window.py \
        -i {input.window_count_merge} \
        -vt {params.variance_threshold} \
        -zt {params.zero_threshold} \
        -o {output.final_window_count} \
        1>{log} 2>&1
        """
# ----end get_final_window_count-----


# ----reads在基因body上的count-----
rule genecounts:
    input:
        bams=expand("last/{sample}.last.bam",sample=SAMPLES),
        gtf=config["genebody"]
    output:
        counts="gene_counts/genebody_counts.txt"
    log:
        "gene_counts/genebody_counts.log"
    shell:
        """
        featureCounts -a {input.gtf} \
        --extraAttributes gene_name \
        -p --countReadPairs \
        -o {output.counts} \
        -t genebody \
        -g gene_id \
        -T 20 \
        {input.bams} \
        1>{log} 2>&1
        """
# ----end genecounts-----

# ----filter_genebody_counts-----
rule filter_genebody_counts:
    input:
        counts="gene_counts/genebody_counts.txt"
    output:
        filtered_counts="gene_counts/filtered_genebody_counts.tsv"
    params:
        zero_threshold=config["zero_threshold"],
        variance_threshold=config["variance_threshold"],
    shell:
        """
        python gene_counts/filter_genebody_counts.py \
        -i {input.counts} \
        -zt {params.zero_threshold} \
        -vt {params.variance_threshold} \
        -o {output.filtered_counts} 
        """
# ----end filter_genebody_counts-----


# ----generate_hg38_tmp_genome-----
rule generate_hg38_tmp_genome:
    input:
        genome=config["genome"]
    output:
        tmp_genome=temp("ref/hg38/hg38.tmp.fa")
    log:
        "ref/hg38/generate_hg38_tmp_genome.log"
    shell:
        """
        awk 'BEGIN{{keep=0}}
             /^>/ {{
                 if ($0 ~ /^>chr([1-9]|1[0-9]|2[0-2]|X|Y|M)$/) {{keep=1}} else {{keep=0}}
             }}
             {{ if (keep) print }}
        ' {input.genome} > {output.tmp_genome} 2> {log}
        """
# ----end generate_hg38_tmp_genome-----

# ----获得hg38 mappability-----
rule get_mappability:
    input:
        genome="ref/hg38/hg38.tmp.fa"
    params:
        index_dir="ref/hg38/mappability",
        error_rate=config["error_rate"],
        k_mer=config["k_mer"],
        output_dir="FSD/hg38_mappability/"
    output:
        output_mappability="FSD/hg38_mappability/hg38.tmp.genmap.bedgraph"
    log:
        "FSD/get_mappability.log"
    shell:
        """
        mkdir -p {params.output_dir}
        
        if [ ! -d "{params.index_dir}" ]; then
            genmap index -F {input.genome} -I {params.index_dir}
        else
            echo "index directory already exists"
        fi

        genmap map -K {params.k_mer} -E {params.error_rate} \
        -I {params.index_dir} -O {params.output_dir} -bg \
        1>{log} 2>&1
        """
# ----end get_mappability-----

# ---- generate_bins -----
rule generate_bins:
    input:
        genome_size=config["genome"] + ".genome"
    params:
        bin_size=config["bin_size"],
        blacklist=config["blacklist"]
    output:
        bins="FSD/hg38_bins.bed"
    log:
        "FSD/generate_bins.log"
    shell:
        """
        bedtools makewindows -g {input.genome_size} -w {params.bin_size} \
        | bedtools intersect -a stdin -b {params.blacklist} -v \
        | sort -k1,1 -k2,2n > {output.bins} 2>{log}
        """
# ----end generate_bins-----

# ----calculate_GC_content-----
rule calculate_GC_content:
    input:
        genome=config["genome"],
        bins="FSD/hg38_bins.bed"
    output:
        GC_content="FSD/hg38_bin_GC.bed"
    log:
        "FSD/calculate_GC.log"
    shell:
        """
       bedtools nuc -fi {input.genome} -bed {input.bins} > {output.GC_content} \
       2>{log}
        """

# ----end calculate_GC_content-----

# ----calculate_mappability-----
rule calculate_mappability:
    input:
        mappability="FSD/hg38_mappability/hg38.tmp.genmap.bedgraph",
        bins="FSD/hg38_bins.bed"
    output:
        map_bins="FSD/hg38_bin_mappability.bed"
    log:
        "FSD/calculate_mappability.log"
    shell:
        """
       bedtools intersect -a {input.bins} -b {input.mappability} -wao \
        | awk 'BEGIN{{OFS="\t"}}
               {{
                 key=$1":"$2"-"$3; chr=$1; s=$2; e=$3;
                 ov=$NF; score=$(NF-1);
                 if (!(key in seen)) {{C[key]=chr; S[key]=s; E[key]=e; seen[key]=1}}
                 if (ov>0 && score!=".") {{ num[key]+=ov*score; den[key]+=ov }}
               }}
               END{{
                 for (k in seen) {{
                   m = (den[k]>0) ? num[k]/den[k] : 0;  
                   print C[k], S[k], E[k], m;
                 }}
               }}' \
        | sort -k1,1 -k2,2n > {output.map_bins} 2>{log}
        """
# ----end calculate_mappability-----

# ----merge_gc_and_map------
rule merge_gc_and_map:
    input:
        GC_content="FSD/hg38_bin_GC.bed",
        map_bins="FSD/hg38_bin_mappability.bed"
    output:
        gc_map="FSD/hg38_bin_GC_map.bed"
    shell:
        """
        awk '!/^#/' {input.GC_content} | \
        paste - {input.map_bins} > {output.gc_map}
        """
# ----end merge_gc_and_map-----

# ----filter_bins-----
rule filter_bins:
    input:
        gc_map="FSD/hg38_bin_GC_map.bed"
    output:
        filtered_bins="FSD/hg38_filtered_bins.bed"
    params:
        GC_threshold=config["GC_threshold"],
        mappability_threshold=config["mappability_threshold"]
    shell:
        """
        awk '$5 >= {params.GC_threshold} && $NF >= {params.mappability_threshold} {{print $1"\\t"$2"\\t"$3"\\t"$5"\\t"$NF}}' {input.gc_map} > {output.filtered_bins}
        """
# ----end filter_bins-----

# ----primary_bam（次级比对和补充比对会被删除，为name sorted而非coordinate sorted）-----
rule paired_bam:
    input:
        bam="last/{sample}.last.bam"
    output:
        paired_bam="paired_bam/{sample}.paired.bam"
    log:
        "paired_bam/{sample}.paired.log"
    threads: 5
    shell:
        """
        samtools view -h -b -f 3 -F 2316 {input.bam} | \
        samtools sort -n -@ {threads} -o {output.paired_bam} 1>{log} 2>&1
        """
# ----end paired_bam-----

# ----paired_bam flagstat-----
rule paired_bam_index_stat:
    input:
        bam="paired_bam/{sample}.paired.bam"
    output:
        stat="paired_bam/{sample}.paired.stat"
    shell:
        """
        samtools flagstat {input.bam} > {output.stat}
        """
# ----end paired_bam_index_stat-----

# ----aggregate sample stats (matrix: sample x metrics)-----
rule aggregate_sample_stats:
    input:
        fastp_jsons=expand("clean/{sample}.fastp.json", sample=SAMPLES),
        map_stats=expand("map/{sample}.raw.stat", sample=SAMPLES),
        last_stats=expand("last/{sample}.last.stat", sample=SAMPLES),
        paired_stats=expand("paired_bam/{sample}.paired.stat", sample=SAMPLES),
    output:
        matrix="statistic/sample_stats_matrix.csv"
    params:
        samples_file=config["samples_file"]
    log:
        "statistic/aggregate_sample_stats.log"
    shell:
        """
        python statistic/aggregate_sample_stats.py \
            --samples-file {params.samples_file} \
            --workdir . \
            -o {output.matrix} \
            1>{log} 2>&1
        """
# ----end aggregate_sample_stats-----

# ----generate_fragments（次级比对和补充比对会被删除）-----
rule generate_fragments:
    input:
        bam="paired_bam/{sample}.paired.bam"
    output:
        fragments="fragments/{sample}_fragments.bed"
    log:
        "fragments/{sample}_fragments.log"
    shell:
        """
        {{
            bedtools bamtobed -i {input.bam} -bedpe | \
                awk 'BEGIN{{OFS="\\t"}} {{
                    if($1==$4) {{
                        start = ($2 < $5 ? $2 : $5)
                        end   = ($3 > $6 ? $3 : $6)
                        print $1, start, end
                    }}
                }}' | \
            sort -k1,1 -k2,2n > {output.fragments}
        }} 2>{log}
        """
# ----end generate_fragments-----


# ----FSD-----
rule FSD:
    input:
        fragments=expand("fragments/{sample}_fragments.bed",sample=SAMPLES),
        bins="FSD/hg38_filtered_bins.bed",
    output:
        fsd="FSD/fsd.tsv",
        fsd_annotation="FSD/fsd_annotation.tsv",
    params:
        n_cores=20,
        length_bin=config["length_bin"],
    log:
        "FSD/generate_fsd.log"
    shell:
        """
        python FSD/generate_fsd.py \
        -i {input.fragments} \
        -b {input.bins} \
        -o {output.fsd} \
        -a {output.fsd_annotation} \
        -l {params.length_bin} \
        -n {params.n_cores} \
        1>{log} 2>&1
        """
# ----end FSD-----


# ----end_motif_analysis-----
rule motif_analysis:
    input:
        expand("last/{sample}.last.bam",sample=SAMPLES)
    output:
        motif_analysis="end_motif/end_motif_freq.tsv"
    params:
        output_dir="end_motif/"
    log:
        "end_motif/motif_analysis.log"
    shell:
        """
        python end_motif/get_end_motif.py -bam {input} -o {params.output_dir} \
        1>{log} 2>&1
        """
# ----end motif_analysis-----


# ----call peaks找5hmc富集区域-----
rule call_peak:
    input:
        fragments="fragments/{sample}_fragments.bed"
    output:
        peaks="peaks/{sample}_peaks.narrowPeak"   ##该文件就是一个bed
    params:
        outdir="./peaks/",
        name="{sample}"
    log:
        "peaks/{sample}.peaks.log"
    shell:
        """
        macs2 callpeak -t {input.fragments} -g hs -f BED \
        --nolambda --nomodel --extsize 167 \
        -n {params.name} --outdir {params.outdir} \
        1>{log} 2>&1
        """
# ----end call_peak-----

# ----generate_consensus_peak-----
rule generate_consensus_peak:
    input:
        peaks=expand("peaks/{sample}_peaks.narrowPeak",sample=SAMPLES),
        metadata=config["metadata"],
    output:
        consensus_peak=expand("consensus_peak/{label}_consensus_peak.bed",label=["0","1"]),
        # filtered_peaks=expand("consensus_peak/filtered_peaks/{sample}.bed",sample=SAMPLES),
    params:
        consensus_peak_dir="consensus_peak/",
        filtered_peaks_dir="consensus_peak/filtered_peaks/",
        genome_file=config["genome"] + ".genome",
        blacklist_file=config["blacklist"],
        label=config["label"],
        percent_threshold=config["percent_threshold"],
        spm_threshold=config["spm_threshold"],
        fc_threshold=config["fc_threshold"],
    shell:
        """
        python consensus_peak/generate_consensus_peak.py \
        -i {input.peaks} -oc {params.consensus_peak_dir} \
        -of {params.filtered_peaks_dir} -m {input.metadata} \
        -l {params.label} -g {params.genome_file} -b {params.blacklist_file} \
        -p {params.percent_threshold} -s {params.spm_threshold} -f {params.fc_threshold}
        """
# ----end generate_consensus_peak-----

# ----merge_consensus_peak-----
rule merge_consensus_peak:
    input:
        consensus_peak=expand("consensus_peak/{label}_consensus_peak.bed",label=["0","1"]),
    output:
        merged_consensus_peak="consensus_peak/merged_consensus_peak.bed"
    shell:
        """
        cat {input.consensus_peak} | bedtools sort | bedtools merge > {output.merged_consensus_peak}
        """
# ----end merge_consensus_peak-----

# ----count_to_consensus_peak-----
rule count_to_consensus_peak:
    input:
        fragments="fragments/{sample}_fragments.bed",
        merged_consensus_peak="consensus_peak/merged_consensus_peak.bed",
    output:
        count_to_consensus_peak="consensus_peak/peak_counts/{sample}_counts.bed"
    shell:
        """
        bedtools coverage -a {input.merged_consensus_peak} -b {input.fragments} -counts > {output.count_to_consensus_peak}
        """
# ----end count_to_consensus_peak-----

# ----merge_peak_count-----
rule merge_peak_count:
    input:
        expand("consensus_peak/peak_counts/{sample}_counts.bed",sample=SAMPLES)
    output:
        peak_count_merge="consensus_peak/peak_count_merged.tsv"
    shell:
        """
        python consensus_peak/merge_peak_count.py -i {input} -o {output.peak_count_merge}
        """
# ----end merge_peak_count-----

# ----filter_peak_counts-----
rule filter_peak_counts:
    input:
        counts="consensus_peak/peak_count_merged.tsv"
    output:
        filtered_counts="consensus_peak/filtered_peak_counts.tsv"
    params:
        zero_threshold=config["zero_threshold"],
        variance_threshold=config["variance_threshold"],
    shell:
        """
        python consensus_peak/filter_peak_counts.py \
        -i {input.counts} \
        -zt {params.zero_threshold} \
        -vt {params.variance_threshold} \
        -o {output.filtered_counts} 
        """
# ----end filter_peak_counts-----


# ----readCounter----
rule readCounter:
    input:
        last="last/{sample}.last.bam",
    output:
        wig="readCounter/{sample}.wig",
    params:
        readCounter_path=config["readCounter_path"],
    shell:
        """
        set +u
        source activate shyR
        {params.readCounter_path} --window 1000000 \
        --quality 30 \
        --chromosome "chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX,chrY" \
        {input.last} > {output.wig}
        source activate py310
        """
# ----end readCounter----

# ----create PoN----
rule create_PoN:
    input:
        PoN_samples_path_file=config["PoN_samples_path_file"],
    output:
        PoN="PoN/normal_median.rds",
    params:
        outfile="PoN/normal",
        Rfile=config["ichorCNA_path"] + "scripts/createPanelOfNormals.R",
        gcWig=config["ichorCNA_path"] + "inst/extdata/gc_hg38_1000kb.wig",
        mapWig=config["ichorCNA_path"] + "inst/extdata/map_hg38_1000kb.wig",
        centromere=config["ichorCNA_path"] + "inst/extdata/GRCh38.GCA_000001405.2_centromere_acen.txt",
    shell:
        """
        set +u
        source activate shyR
        Rscript {params.Rfile} \
        --filelist {input.PoN_samples_path_file} \
        --gcWig {params.gcWig} \
        --mapWig {params.mapWig} \
        --centromere {params.centromere} \
        --outfile {params.outfile}
        source activate py310
        """
# ----end create PoN----

# ----ichorCNA----
rule ichorCNA:
    input:
        wig="readCounter/{sample}.wig",
        normalPanel="PoN/normal_median.rds",
    output:
        ichorCNA="ichorCNA/{sample}/{sample}.params.txt",
    params:
        Rfile=config["ichorCNA_path"] + "scripts/runIchorCNA.R",
        sample_id="{sample}",
        gcWig=config["ichorCNA_path"] + "inst/extdata/gc_hg38_1000kb.wig",
        mapWig=config["ichorCNA_path"] + "inst/extdata/map_hg38_1000kb.wig",
        centromere=config["ichorCNA_path"] + "inst/extdata/GRCh38.GCA_000001405.2_centromere_acen.txt",
        outDir="ichorCNA/{sample}/",
    shell:
        """
        set +u
        source activate shyR
        Rscript {params.Rfile} \
        --genomeBuild hg38 \
        --id {params.sample_id} \
        --WIG {input.wig} \
        --ploidy "c(2)" \
        --normal "c(0.95, 0.99, 0.995, 0.999)" \
        --gcWig {params.gcWig} \
        --mapWig {params.mapWig} \
        --centromere {params.centromere} \
        --normalPanel {input.normalPanel} \
        --maxCN 3 \
        --estimateScPrevalence FALSE --scStates "c()" \
        --outDir {params.outDir}
        source activate py310
        """

# ----end ichorCNA----

# ----get bw----
rule get_bw:
    input:
        last="last/{sample}.last.bam",
    output:
        bw="bw/{sample}.bw",
    params:
        blacklist=config["blacklist"],
        threads=5,
    shell:
        """
        bamCoverage -b {input.last} -o {output.bw} -of bigwig \
        -bl {params.blacklist} --normalizeUsing RPGC \
        --ignoreForNormalization chrY chrM \
        --extendReads --effectiveGenomeSize 2913022398 \
        --numberOfProcessors {params.threads}
        """
# ----end get bw----

# ----get bw matrix----
rule bw_matrix:
    input:
        bw=expand("bw/{sample}.bw",sample=SAMPLES),
    output:
        bw_matrix="QC_plots/bw_matrix.npz",
        raw_counts="QC_plots/raw_counts.tsv",
    params:
        threads=5,
        blacklist=config["blacklist"],
    shell:
        """
        multiBigwigSummary bins -b {input.bw} -o {output.bw_matrix} \
        -p {params.threads} -bl {params.blacklist} --smartLabels \
        --chromosomesToSkip chrY chrM --outRawCounts {output.raw_counts}
        """
# ----end get bw matrix----

# ----plot PCA and correlation----
rule plot_PCA_and_correlation:
    input:
        bw_matrix="QC_plots/bw_matrix.npz",
    output:
        PCA_plot="QC_plots/PCA_plot.png",
        PCA_matrix="QC_plots/PCA_matrix.tsv",
        correlation_plot="QC_plots/correlation_plot.png",
        correlation_matrix="QC_plots/correlation_matrix.tsv",
    shell:
        """
        plotCorrelation -in {input.bw_matrix} -c spearman \
        -o {output.correlation_plot} -p heatmap --skipZeros \
        --plotTitle "Spearman Correlation" --outFileCorMatrix {output.correlation_matrix} \
        --plotNumbers

        plotPCA -in {input.bw_matrix} -o {output.PCA_plot} \
        --outFileNameData {output.PCA_matrix} \
        --plotTitle "PCA Plot" --rowCenter --log2
        """
# ----end plot PCA and correlation----

# ----GC bias plot----
rule GC_bias_plot:
    input:
        last="last/{sample}.last.bam",
    output:
        GC_bias_plot="QC_plots/{sample}/{sample}_GC_bias_plot.png",
        GC_bias_freq="QC_plots/{sample}/{sample}_GC_bias_freq.txt",
    params:
        genome_2bit=config["genome_2bit"],
        blacklist=config["blacklist"],
        threads=5,  
    shell:
        """
        computeGCBias -b {input.last} \
        -g {params.genome_2bit} -bl {params.blacklist} \
        --numberOfProcessors {params.threads} \
        --biasPlot {output.GC_bias_plot} \
        --GCbiasFrequenciesFile {output.GC_bias_freq} \
        --effectiveGenomeSize 2913022398 \
        """
# ----end GC bias plot----


# ----plot coverage plot----
rule plot_coverage_plot:
    input:
        last="last/{sample}.last.bam",
    output:
        coverage_plot="QC_plots/{sample}/{sample}_coverage_plot.png",
    params:
        threads=5,
        blacklist=config["blacklist"],
    shell:
        """
        plotCoverage -b {input.last} -o {output.coverage_plot} \
        --plotTitle "Coverage Plot" --smartLabels \
        --numberOfProcessors {params.threads} \
        --extendReads \
        --blackListFileName {params.blacklist}
        """
# ----end plot coverage plot----


# ----compute matrix----
rule compute_matrix:
    input:
        bw="bw/{sample}.bw",
    output:
        genebody_matrix="coverage_plots/{sample}/genebody.gz",
        genebody_raw_matrix="coverage_plots/{sample}/genebody.raw.tsv",
        genebody_sorted_regions="coverage_plots/{sample}/sorted_genebody.bed",
        enhancer_matrix="coverage_plots/{sample}/enhancer.gz",
        enhancer_raw_matrix="coverage_plots/{sample}/enhancer.raw.tsv",
        enhancer_sorted_regions="coverage_plots/{sample}/sorted_enhancer.bed",
    params:
        genebody=config["genebody_bed"],
        ts_enhancer=config["ts_enhancer"],
        blacklist=config["blacklist"],
        threads=5,
    shell:
        """
        computeMatrix scale-regions -R {params.genebody} \
        -S {input.bw} -o {output.genebody_matrix} \
        --outFileNameMatrix {output.genebody_raw_matrix} \
        --outFileSortedRegions {output.genebody_sorted_regions} \
        --regionBodyLength 5000 \
        -b 2000 -a 2000 \
        --sortRegions descend \
        --skipZeros \
        --blackListFileName {params.blacklist} \
        --smartLabels \
        --numberOfProcessors {params.threads}

        computeMatrix reference-point -R {params.ts_enhancer} \
        -S {input.bw} -o {output.enhancer_matrix} \
        --outFileNameMatrix {output.enhancer_raw_matrix} \
        --outFileSortedRegions {output.enhancer_sorted_regions} \
        --referencePoint "center" \
        -b 2000 -a 2000 \
        --sortRegions descend \
        --skipZeros \
        --blackListFileName {params.blacklist} \
        --smartLabels \
        --numberOfProcessors {params.threads}
        """

# ----end compute matrix----


# ----plot heatmap and lineplot----
rule plot_heatmap_and_lineplot:
    input:
        genebody_raw_matrix="coverage_plots/{sample}/genebody.raw.tsv",
        enhancer_raw_matrix="coverage_plots/{sample}/enhancer.raw.tsv",
    output:
        genebody_heatmap="coverage_plots/{sample}/genebody_heatmap.png",
        genebody_lineplot="coverage_plots/{sample}/genebody_lineplot.png",
        enhancer_heatmap="coverage_plots/{sample}/enhancer_heatmap.png",
        enhancer_lineplot="coverage_plots/{sample}/enhancer_lineplot.png",
    shell:
        """
        python coverage_plots/plot_heatmap.py \
        -i {input.genebody_raw_matrix} \
        -o {output.genebody_heatmap}

        python coverage_plots/plot_lineplot.py \
        -i {input.genebody_raw_matrix} \
        -o {output.genebody_lineplot}

        python coverage_plots/plot_heatmap.py \
        -i {input.enhancer_raw_matrix} \
        -o {output.enhancer_heatmap}

        python coverage_plots/plot_lineplot.py \
        -i {input.enhancer_raw_matrix} \
        -o {output.enhancer_lineplot}
        """
# ----end plot heatmap and lineplot----

# ----merge raw matrix groups----
rule merge_raw_matrix_groups:
    input:
        metadata=config["metadata"],
    output:
        genebody_0="coverage_plots/type_specific/genebody_merge_0.tsv",
        genebody_1="coverage_plots/type_specific/genebody_merge_1.tsv",
        genebody_all="coverage_plots/type_specific/genebody_merge_all.tsv",
        enhancer_0="coverage_plots/type_specific/enhancer_merge_0.tsv",
        enhancer_1="coverage_plots/type_specific/enhancer_merge_1.tsv",
        enhancer_all="coverage_plots/type_specific/enhancer_merge_all.tsv",
    params:
        label_col=config["label"],
        output_dir="coverage_plots/type_specific",
        coverage_dir="coverage_plots",
    shell:
        """
        python coverage_plots/merge_raw_matrix.py \
        -m {input.metadata} \
        -l 0 \
        --label_col {params.label_col} \
        --output_dir {params.output_dir} \
        --coverage_dir {params.coverage_dir}

        python coverage_plots/merge_raw_matrix.py \
        -m {input.metadata} \
        -l 1 \
        --label_col {params.label_col} \
        --output_dir {params.output_dir} \
        --coverage_dir {params.coverage_dir}

        python coverage_plots/merge_raw_matrix.py \
        -m {input.metadata} \
        --label_col {params.label_col} \
        --output_dir {params.output_dir} \
        --coverage_dir {params.coverage_dir}
        """
# ----end merge raw matrix groups----

# ----plot heatmap and lineplot groups----
rule plot_heatmap_and_lineplot_groups:
    input:
        genebody_0="coverage_plots/type_specific/genebody_merge_0.tsv",
        genebody_1="coverage_plots/type_specific/genebody_merge_1.tsv",
        genebody_all="coverage_plots/type_specific/genebody_merge_all.tsv",
        enhancer_0="coverage_plots/type_specific/enhancer_merge_0.tsv",
        enhancer_1="coverage_plots/type_specific/enhancer_merge_1.tsv",
        enhancer_all="coverage_plots/type_specific/enhancer_merge_all.tsv",
    output:
        genebody_heatmap="coverage_plots/type_specific/genebody_heatmap.png",
        genebody_lineplot="coverage_plots/type_specific/genebody_lineplot.png",
        genebody_heatmap_all="coverage_plots/type_specific/genebody_heatmap_all.png",
        genebody_lineplot_all="coverage_plots/type_specific/genebody_lineplot_all.png",
        enhancer_heatmap="coverage_plots/type_specific/enhancer_heatmap.png",
        enhancer_lineplot="coverage_plots/type_specific/enhancer_lineplot.png",
        enhancer_heatmap_all="coverage_plots/type_specific/enhancer_heatmap_all.png",
        enhancer_lineplot_all="coverage_plots/type_specific/enhancer_lineplot_all.png",
    shell:
        """
        python coverage_plots/plot_heatmap_group.py \
        -i {input.genebody_0} {input.genebody_1} \
        -o {output.genebody_heatmap}

        python coverage_plots/plot_lineplot_group.py \
        -i {input.genebody_0} {input.genebody_1} \
        -o {output.genebody_lineplot}

        python coverage_plots/plot_heatmap_group.py \
        -i {input.enhancer_0} {input.enhancer_1} \
        -o {output.enhancer_heatmap}

        python coverage_plots/plot_lineplot_group.py \
        -i {input.enhancer_0} {input.enhancer_1} \
        -o {output.enhancer_lineplot}

        python coverage_plots/plot_heatmap_group.py \
        -i {input.genebody_all} \
        -o {output.genebody_heatmap_all} \
        -l label_0:all

        python coverage_plots/plot_lineplot_group.py \
        -i {input.genebody_all} \
        -o {output.genebody_lineplot_all} \
        -l label_0:all

        python coverage_plots/plot_heatmap_group.py \
        -i {input.enhancer_all} \
        -o {output.enhancer_heatmap_all} \
        -l label_0:all

        python coverage_plots/plot_lineplot_group.py \
        -i {input.enhancer_all} \
        -o {output.enhancer_lineplot_all} \
        -l label_0:all
        """
# ----end plot heatmap and lineplot groups----


# --ChromHMM state annatation----
rule ChromHMM_state_annatation:
    input:
        chrom_state=config["chrom_state"],
        consensus_peak_0="consensus_peak/0_consensus_peak.bed",
        consensus_peak_1="consensus_peak/1_consensus_peak.bed",
        diff_peak="feature_selection/consensus_peak/significant_peaks.bed",
        diff_window="feature_selection/window/significant_windows.bed",
        diff_OCR="feature_selection/OCR/significant_OCR.bed",
    output:
        annotation_0="annotation/0_consensus_peak.txt",
        annotation_1="annotation/1_consensus_peak.txt",
        annotation_diff_peak="annotation/significant_peaks.txt",
        annotation_diff_window="annotation/significant_windows.txt",
        annotation_diff_OCR="annotation/significant_OCR.txt",
    params:
        output_dir="annotation/",
    shell:
        """
        python chromHMM/overlap_enrichment.py \
            -i {input.consensus_peak_0} \
               {input.consensus_peak_1} \
               {input.diff_peak} \
               {input.diff_window} \
               {input.diff_OCR} \
            -c {input.chrom_state} \
            --output-dir {params.output_dir} \
            --java-memory 2G
        """
# ----end ChromHMM state annatation----

# ----plot enrichment----
rule plot_enrichment:
    input:
        annotation=expand("annotation/{type}.txt",type=["0_consensus_peak", "1_consensus_peak", "significant_peaks", "significant_windows", "significant_OCR"]),
    output:
        enrichment_plot="chromHMM/enrichment_plot.png",
        enrichment_table="chromHMM/enrichment_table.tsv",
    params:
        chrom_state_anno=config["chrom_state_anno"],
    shell:
        """
        python chromHMM/plot_enrichment.py \
        -i {input.annotation} \
        -o {output.enrichment_plot} \
        -t {output.enrichment_table} \
        -a {params.chrom_state_anno}
        """
# ----end plot enrichment----



# ----plot jaccard and fisher score----
rule plot_jaccard_and_fisher_score:
    input:
        diff_peak="feature_selection/consensus_peak/significant_peaks.bed",
        diff_window="feature_selection/window/significant_windows.bed",
        diff_OCR="feature_selection/OCR/significant_OCR.bed",
        genome_size=config["genome"] + ".genome",
    output:
        jaccard_matrix="enrichment/jaccard_matrix.tsv",
        jaccard_heatmap="enrichment/jaccard_heatmap.png",
        fisher_ratio_matrix="enrichment/fisher_ratio_matrix.tsv",
        fisher_pvalue_matrix="enrichment/fisher_pvalue_matrix.tsv",
        fisher_heatmap="enrichment/fisher_heatmap.png",
    params:
        ts_enhancer=config["ts_enhancer"],
        ts_promoter=config["ts_promoter"],
    shell:
        """
        python enrichment/plot_jaccard_score.py \
            --diff_peak {input.diff_peak} \
            --diff_window {input.diff_window} \
            --diff_OCR {input.diff_OCR} \
            --ts_promoter {params.ts_promoter} \
            --ts_enhancer {params.ts_enhancer} \
            --genome_size {input.genome_size} \
            --output_plot {output.jaccard_heatmap} \
            --output_matrix {output.jaccard_matrix} \
            --output_fisher_ratio {output.fisher_ratio_matrix} \
            --output_fisher_pvalue {output.fisher_pvalue_matrix} \
            --output_fisher_plot {output.fisher_heatmap}
        """
# ----end plot jaccard score----

# ----filter OCR----
rule filter_OCR:
    input:
        OCR=config["OCR"],                       
        liftover_chain=config["liftover_chain"], 
        blacklist=config["blacklist"],      
    output:
        lifted=temp("ref/hg38/ocr.hg38.bed"),
        unmapped="ref/hg38/ocr.unmapped.bed",
        filtered="ref/hg38/tumor_ocr_filtered.bed",
    shell:
        """
        liftOver \
          -minMatch=0.95 \
          {input.OCR} \
          {input.liftover_chain} \
          {output.lifted} \
          {output.unmapped}

        bedtools intersect \
          -v \
          -a {output.lifted} \
          -b {input.blacklist} \
          > {output.filtered}
        """

# ----end filter OCR----


# ----OCR count----
rule OCR_count:
    input:
        OCR="ref/hg38/tumor_ocr_filtered.bed",
        fragment=expand("fragments/{sample}_fragments.bed",sample=SAMPLES),
    output:
        OCR_count="OCR/OCR_count.tsv",
    shell:
        """
        python OCR/OCR_count.py --ocr {input.OCR} -i {input.fragment} -o {output.OCR_count}
        """
# ----end OCR count----

# ----filter OCR count----
rule filter_OCR_count:
    input:
        OCR_count="OCR/OCR_count.tsv"
    output:
        final_OCR_count="OCR/final_OCR_count.tsv"
    params:
        variance_threshold=config["variance_threshold"],
        zero_threshold=config["zero_threshold"],
    log:
        "OCR/filter_OCR_count.log"
    shell:
        """
        python OCR/filter_OCR.py \
        -i {input.OCR_count} \
        -vt {params.variance_threshold} \
        -zt {params.zero_threshold} \
        -o {output.final_OCR_count} \
        1>{log} 2>&1
        """
# ----end filter OCR count----

# ----evaluate tumor fraction-----
rule tumor_fraction:
    input:
        exp_specific_peak=config["exp_specific_peak"],
        fragment=expand("fragments/{sample}_fragments.bed",sample=SAMPLES),
    output:
        exp_coverage="tumor_fraction/exp_coverage.tsv",
        size_factor="tumor_fraction/size_factor.tsv",
        tumor_fraction="tumor_fraction/tumor_fraction.tsv",
    shell:
        """
        python tumor_fraction/merge_coverage.py \
        -p {input.exp_specific_peak} \
        -f {input.fragment} \
        -o {output.exp_coverage} \
        -s {output.size_factor}

        python tumor_fraction/calculate_tumor_fraction.py \
        -e {output.exp_coverage} \
        -s {output.size_factor} \
        -o {output.tumor_fraction}

        """
# ----end tumor fraction-----

# ----CNA----
rule CNA:
    input:
        seg=expand("ichorCNA/{sample}/{sample}.cna.seg",sample=SAMPLES),
        ploidy=expand("ichorCNA/{sample}/{sample}.params.txt",sample=SAMPLES),
    output:
        CNA="CNA/CNA.tsv",
        CNA_tumor_fraction="CNA/CNA_tumor_fraction.tsv",
    shell:
        """
        python CNA/CNA.py \
        -i {input.seg} \
        -o {output.CNA}

        python CNA/tumor_fraction.py \
        -i {input.ploidy} \
        -o {output.CNA_tumor_fraction}
        """
# ----end CNA----

# ----CNA bed----
rule CNA_bed:
    input:
        CNA="CNA/CNA.tsv",
    output:
        CNA_bed="CNA/CNA.bed",
    shell:
        """
        awk 'BEGIN {{OFS="\\t"}} NR>1 {{print "chr"$1, $2-1, $3}}' {input.CNA} > {output.CNA_bed}
        """
# ----end CNA bed----

# ----TOO (Tissue of Origin) 分析-----
rule run_too:
    input:
        fragments=expand("fragments/{sample}_fragments.bed", sample=SAMPLES),
        peaks=get_ref_peak_files,
    output:
        tissue_score="too/tissue_score.tsv",
        tissue_cv="too/tissue_CV.tsv",
        replicate_plot="too/tissue_replicate_deviation.png",
        tissue_percent="too/tissue_percent.tsv",
    threads: 4,
    shell:
        """
        python too/run_too.py \
            --fragments {input.fragments} \
            --peaks {input.peaks} \
            --out-score {output.tissue_score} \
            --out-cv {output.tissue_cv} \
            --out-plot {output.replicate_plot} \
            --out-percent {output.tissue_percent} \
            --n-jobs {threads}
        """
# ----end TOO-----




