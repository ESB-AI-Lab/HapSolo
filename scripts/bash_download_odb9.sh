#!/usr/bin/env bash
# bash_download_odb9.sh
#
# Download OrthoDB v9 lineage datasets for BUSCO v3 / HapSolo.
# 
# Usage:
#   # Download a single lineage:
#   bash bash_download_odb9.sh vertebrata
#
#   # Download all lineages:
#   bash bash_download_odb9.sh all
#
#   # List available lineages:
#   bash bash_download_odb9.sh list

set -euo pipefail

# ── Update these URLs after uploading ─────────────────────────────────────
# Internet Archive (primary):
IA_BASE="https://archive.org/download/odb9-busco-lineages"
# Zenodo (backup — DOI: 10.5281/zenodo.20987183):
ZEN_BASE="https://zenodo.org/records/20987183/files"

BASE_URL="${IA_BASE}"

LINEAGES=(
    actinobacteria_odb9
    actinopterygii_odb9
    alveolata_stramenophiles_ensembl
    arthropoda_odb9
    ascomycota_odb9
    aves_odb9
    bacillales_odb9
    bacteria_odb9
    bacteroidetes_odb9
    basidiomycota_odb9
    betaproteobacteria_odb9
    clostridia_odb9
    cyanobacteria_odb9
    deltaepsilonsub_odb9
    dikarya_odb9
    diptera_odb9
    embryophyta_odb9
    endopterygota_odb9
    enterobacteriales_odb9
    euarchontoglires_odb9
    eukaryota_odb9
    eurotiomycetes_odb9
    firmicutes_odb9
    fungi_odb9
    gammaproteobacteria_odb9
    hymenoptera_odb9
    insecta_odb9
    lactobacillales_odb9
    laurasiatheria_odb9
    mammalia_odb9
    metazoa_odb9
    microsporidia_odb9
    nematoda_odb9
    pezizomycotina_odb9
    proteobacteria_odb9
    protists_ensembl
    rhizobiales_odb9
    saccharomyceta_odb9
    saccharomycetales_odb9
    sordariomyceta_odb9
    spirochaetes_odb9
    tenericutes_odb9
    tetrapoda_odb9
    vertebrata_odb9
)

case "${1:-list}" in
    list)
        echo "Available odb9 lineages (${#LINEAGES[@]} total):"
        echo ""
        printf '  %s\n' "${LINEAGES[@]}"
        echo ""
        echo "Usage:  bash $0 <lineage_name>    # download one"
        echo "        bash $0 all               # download all (~3.6 GB)"
        ;;
    all)
        echo "Downloading all ${#LINEAGES[@]} odb9 lineages (~3.6 GB)..."
        echo ""
        for lineage in "${LINEAGES[@]}"; do
            echo "── ${lineage} ──"
            wget -c "${BASE_URL}/${lineage}.tar.gz"
            echo ""
        done
        echo "Downloading checksums..."
        wget -c "${BASE_URL}/SHA256SUMS.txt"
        echo ""
        echo "Verifying integrity..."
        sha256sum -c SHA256SUMS.txt
        echo ""
        echo "Done! Extract a lineage with: tar -xzf <lineage>.tar.gz"
        ;;
    *)
        LINEAGE="${1}"
        # Strip trailing .tar.gz if the user included it
        LINEAGE="${LINEAGE%.tar.gz}"

        # Validate
        FOUND=0
        for l in "${LINEAGES[@]}"; do
            if [[ "${l}" == "${LINEAGE}" ]]; then
                FOUND=1
                break
            fi
        done

        if [[ ${FOUND} -eq 0 ]]; then
            echo "ERROR: Unknown lineage '${LINEAGE}'"
            echo "Run '$0 list' to see available lineages."
            exit 1
        fi

        echo "Downloading ${LINEAGE}..."
        wget -c "${BASE_URL}/${LINEAGE}.tar.gz"
        echo ""
        echo "Extract with: tar -xzf ${LINEAGE}.tar.gz"
        echo "Then use with BUSCO v3:"
        echo "  run_BUSCO.py -i genome.fasta -o output -l ${LINEAGE} -m genome"
        ;;
esac
