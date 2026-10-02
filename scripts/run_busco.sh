#!/bin/bash
# Run BUSCO on a single genome assembly
# Usage: bash scripts/run_busco.sh <assembly.fasta>

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <assembly.fasta>"
    exit 1
fi

FASTA=$(realpath "$1")
OUTDIR=$(dirname "$FASTA")
BASENAME=$(basename "$FASTA" _primary.fasta)
OUTNAME="busco_${BASENAME}"

CONTAINER=/dpool/singularity/busco/busco6.1.sif
DOWNLOADS=/dpool/github/HapSolo/busco_downloads
LINEAGE=diptera_odb12.2
CORES=16

if [ ! -f "$FASTA" ]; then
    echo "Error: $FASTA not found"
    exit 1
fi

if [ -f "$OUTDIR/$OUTNAME/short_summary.specific.${LINEAGE}.${OUTNAME}.txt" ]; then
    echo "SKIP: already complete at $OUTDIR/$OUTNAME"
    cat "$OUTDIR/$OUTNAME/short_summary.specific.${LINEAGE}.${OUTNAME}.txt" | grep -E "C:|Complete|Single|Duplicated|Fragmented|Missing"
    exit 0
fi

echo "Running BUSCO on: $FASTA"
echo "Output: $OUTDIR/$OUTNAME"

singularity exec --bind /dpool:/dpool "$CONTAINER" \
    busco -i "$FASTA" \
        -l $LINEAGE \
        -o "$OUTNAME" \
        --out_path "$OUTDIR" \
        --download_path "$DOWNLOADS" \
        --offline \
        -m genome \
        -c $CORES \
        -f
