#!/bin/bash
#$ -N hapsolo_search
#$ -o hapsolo_search.o$JOB_ID
#$ -e hapsolo_search.e$JOB_ID
#$ -pe openmp 16
#$ -l h_rt=12:00:00
###$ -q YOUR_QUEUE

# HapSolo step 3: ortholog gene search with miniprot against an OrthoDB lineage.
#
# Single multi-threaded job. Replaces the legacy BUSCO V3 array-job approach.
#
# Usage:
#   1. Edit REF, LINEAGE, OUTPUT_DIR, HAPSOLO_DIR, and PYTHON below.
#   2. Make sure miniprot is on PATH.
#   3. Submit:  qsub qsub_search.sh
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

cd "$SGE_O_WORKDIR"

# Optional: load miniprot from your HPC's module system
###module load miniprot

$PYTHON $HAPSOLO_DIR/hapsolo_cli.py search \
    -i ${REF} \
    -l ${LINEAGE} \
    -o ${OUTPUT_DIR} \
    -t ${NSLOTS}

echo "Ortholog search complete."
echo "  Output: ${OUTPUT_DIR}/"
