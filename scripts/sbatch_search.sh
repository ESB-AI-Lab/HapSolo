#!/bin/bash
#SBATCH -J hapsolo_search
#SBATCH -o hapsolo_search.o%j
#SBATCH -e hapsolo_search.e%j
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 12:00:00
###SBATCH -p YOUR_PARTITION

# HapSolo step 3: ortholog gene search with miniprot against an OrthoDB lineage.
#
# Single multi-threaded job. Replaces the legacy BUSCO V3 array-job approach.
#
# Usage:
#   1. Edit REF, LINEAGE, OUTPUT_DIR, HAPSOLO_DIR, and PYTHON below.
#   2. Make sure miniprot is on PATH.
#   3. Submit:  sbatch sbatch_search.sh
#
# Output:
#   ${OUTPUT_DIR}/   - per-contig ortholog classification TSV files

set -euo pipefail

# === EDIT THESE ===
REF=myassembly_new.fasta
LINEAGE=/path/to/diptera_odb10
OUTPUT_DIR=ortholog_output
HAPSOLO_DIR=/path/to/HapSolo
PYTHON=python3
# ==================

cd "$SLURM_SUBMIT_DIR"

# Optional: load miniprot from your HPC's module system
###module load miniprot

$PYTHON $HAPSOLO_DIR/hapsolo_cli.py search \
    -i ${REF} \
    -l ${LINEAGE} \
    -o ${OUTPUT_DIR} \
    -t ${SLURM_CPUS_PER_TASK}

echo "Ortholog search complete."
echo "  Output: ${OUTPUT_DIR}/"
