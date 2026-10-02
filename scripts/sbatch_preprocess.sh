#!/bin/bash
#SBATCH -J hapsolo_preprocess
#SBATCH -o hapsolo_preprocess.o%j
#SBATCH -e hapsolo_preprocess.e%j
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 02:00:00
###SBATCH -p YOUR_PARTITION

# HapSolo step 1: clean FASTA headers and split contigs into individual files.
#
# Usage:
#   1. Edit REF, HAPSOLO_DIR, and PYTHON below.
#   2. Submit:  sbatch sbatch_preprocess.sh
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

cd "$SLURM_SUBMIT_DIR"

$PYTHON $HAPSOLO_DIR/hapsolo_cli.py preprocess -i ${REF}

echo "Preprocess complete."
echo "  Sanitized assembly: ${REF%.fasta}_new.fasta"
echo "  Per-contig files:   contigs/"
