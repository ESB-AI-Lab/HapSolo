#!/bin/bash
#$ -N hapsolo_preprocess
#$ -o hapsolo_preprocess.o$JOB_ID
#$ -e hapsolo_preprocess.e$JOB_ID
#$ -pe openmp 1
#$ -l h_rt=02:00:00
###$ -q YOUR_QUEUE

# HapSolo step 1: clean FASTA headers and split contigs into individual files.
#
# Usage:
#   1. Edit REF, HAPSOLO_DIR, and PYTHON below.
#   2. Submit:  qsub qsub_preprocess.sh
#
# Output:
#   ${REF%.fasta}_new.fasta   - sanitized assembly
#   contigs/                  - per-contig FASTA files
#   contigs/name_mapping.tsv  - original-to-sanitized name lookup

set -euo pipefail

# === EDIT THESE ===
REF=myassembly.fasta
HAPSOLO_DIR=/path/to/HapSolo
PYTHON=python3
# ==================

cd "$SGE_O_WORKDIR"

$PYTHON $HAPSOLO_DIR/hapsolo_cli.py preprocess -i ${REF}

echo "Preprocess complete."
echo "  Sanitized assembly: ${REF%.fasta}_new.fasta"
echo "  Per-contig files:   contigs/"
