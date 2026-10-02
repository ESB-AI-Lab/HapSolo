#!/bin/bash
# Generate per-species MD5 manifest files
# Columns: complete_hash\tlast8hash\tfile_location
# Saves incrementally — each file is hashed and appended immediately

BASEDIR="/dpool/github/HapSolo"
OUTDIR="${BASEDIR}/md5_manifests"
mkdir -p "$OUTDIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

process_species() {
    local species="$1"
    local outfile="$2"
    shift 2

    log "START: ${species}"
    > "$outfile"
    local count=0

    # Collect all FASTA files from all provided directories
    local tmplist=$(mktemp)
    for d in "$@"; do
        [ ! -d "$d" ] && [ ! -f "$d" ] && continue
        if [ -f "$d" ]; then
            echo "$d" >> "$tmplist"
            continue
        fi
        find "$d" -name "*.fasta" \
            -not -name "*secondary*" \
            -not -name "*_a_ctg*" \
            -not -path "*/contigs/*" \
            >> "$tmplist"
    done

    local total=$(wc -l < "$tmplist")
    log "  ${species}: found ${total} FASTA files"

    # Process each file
    sort "$tmplist" | while IFS= read -r f; do
        count=$((count + 1))
        local full=$(md5sum "$f" | cut -d' ' -f1)
        local last8="${full: -8}"
        printf "%s\t%s\t%s\n" "$full" "$last8" "$f" >> "$outfile"
        if (( count % 10 == 0 )); then
            log "  ${species}: ${count}/${total} done"
        fi
    done

    rm -f "$tmplist"
    local final=$(wc -l < "$outfile")
    log "DONE: ${species} — ${final} entries → ${outfile}"
}

START=$(date +%s)

# === Shark (smallest — 5 files) ===
process_species "squalus_acanthias" "${OUTDIR}/squalus_acanthias_md5.txt" \
    "${BASEDIR}/example_data/shark" \
    "${BASEDIR}/asms/shark"

# === Chardonnay (21 files) ===
process_species "vitis_vinifera_chardonnay" "${OUTDIR}/vitis_vinifera_chardonnay_md5.txt" \
    "${BASEDIR}/example_data/chardonnay" \
    "${BASEDIR}/asms/chardonnay"

# === Mosquito (28 files) ===
process_species "anopheles_funestus" "${OUTDIR}/anopheles_funestus_md5.txt" \
    "${BASEDIR}/example_data/mosquito" \
    "${BASEDIR}/asms/mosquito"

# === Gwen (101 files) ===
process_species "persea_americana_gwen" "${OUTDIR}/persea_americana_gwen_md5.txt" \
    "${BASEDIR}/example_data/gwen" \
    "${BASEDIR}/asms/gwen"

# === Culex tarsalis (largest — 372 files) ===
process_species "culex_tarsalis" "${OUTDIR}/culex_tarsalis_md5.txt" \
    "${BASEDIR}/example_data/culex_tarsalis" \
    "${BASEDIR}/asms/culex_tarsalis_all" \
    "${BASEDIR}/asms/culex_tarsalis" \
    "${BASEDIR}/asms/culex_tarsalis_d2" \
    "${BASEDIR}/asms/culex_tarsalis_d2m2" \
    "${BASEDIR}/asms/culex_tarsalis_d2m3" \
    "${BASEDIR}/asms/culex_tarsalis_d05m2" \
    "${BASEDIR}/asms/culex_tarsalis_s2d3" \
    "${BASEDIR}/asms/culex_tarsalis_s3d2" \
    "${BASEDIR}/asms/culex_tarsalis_filtered_d2m2" \
    "${BASEDIR}/asms/culex_tarsalis_filtered_d2m2_gpu" \
    "${BASEDIR}/asms/culex_tarsalis_unfiltered_d2m2" \
    "${BASEDIR}/asms/culex_m3_bp" \
    "${BASEDIR}/asms/culex_m3_bp_hap1" \
    "${BASEDIR}/asms/culex_m3_bp_hap1_rw200k" \
    "${BASEDIR}/asms/culex_m3_bp_hap1_sa200k" \
    "${BASEDIR}/asms/culex_m3_bp_hap2" \
    "${BASEDIR}/asms/culex_m3_bp_hap2_rw200k" \
    "${BASEDIR}/asms/culex_m3_bp_hap2_sa200k" \
    "${BASEDIR}/asms/culex_m3_bp_rw200k" \
    "${BASEDIR}/asms/culex_m3_bp_sa200k" \
    "${BASEDIR}/asms/culex_m3_flye" \
    "${BASEDIR}/asms/culex_m3_flye_rw200k" \
    "${BASEDIR}/asms/culex_m3_flye_sa200k" \
    "${BASEDIR}/asms/culex_m3_hifi_bp" \
    "${BASEDIR}/asms/culex_m3_hifi_bp_hap1" \
    "${BASEDIR}/asms/culex_m3_hifi_bp_hap1_rw200k" \
    "${BASEDIR}/asms/culex_m3_hifi_bp_hap1_sa200k" \
    "${BASEDIR}/asms/culex_m3_hifi_bp_hap2" \
    "${BASEDIR}/asms/culex_m3_hifi_bp_hap2_rw200k" \
    "${BASEDIR}/asms/culex_m3_hifi_bp_hap2_sa200k" \
    "${BASEDIR}/asms/culex_m3_hifi_bp_rw200k" \
    "${BASEDIR}/asms/culex_m3_hifi_bp_sa200k" \
    "${BASEDIR}/asms/culex_m3_hifi_p_ctg" \
    "${BASEDIR}/asms/culex_m3_ont_homcov" \
    "${BASEDIR}/asms/culex_m3_p_ctg" \
    "${BASEDIR}/asms/culex_m3_p_ctg_rw200k" \
    "${BASEDIR}/asms/culex_m3_p_ctg_sa200k" \
    "${BASEDIR}/asms/culex_pooled_bp" \
    "${BASEDIR}/asms/culex_pooled_bp_hap1" \
    "${BASEDIR}/asms/culex_pooled_bp_hap1_rw200k" \
    "${BASEDIR}/asms/culex_pooled_bp_hap1_sa200k" \
    "${BASEDIR}/asms/culex_pooled_bp_hap2" \
    "${BASEDIR}/asms/culex_pooled_bp_hap2_rw200k" \
    "${BASEDIR}/asms/culex_pooled_bp_hap2_sa200k" \
    "${BASEDIR}/asms/culex_pooled_bp_rw200k" \
    "${BASEDIR}/asms/culex_pooled_bp_sa200k" \
    "${BASEDIR}/asms/culex_pooled_p_ctg" \
    "${BASEDIR}/asms/culex_m23_rw200k" \
    "${BASEDIR}/asms/culex_m23_sa100k" \
    "${BASEDIR}/asms/culex_m23_sa200k" \
    "${BASEDIR}/busco_gt_fastas"

END=$(date +%s)
ELAPSED=$(( (END - START) / 60 ))

echo ""
log "============================================"
log "All manifests generated in ${ELAPSED} minutes"
log "============================================"
echo ""
for f in "${OUTDIR}"/*.txt; do
    echo "$(basename $f): $(wc -l < $f) entries"
done
