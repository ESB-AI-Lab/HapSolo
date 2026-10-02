#!/bin/bash
#$ -N hapsolo_align
#$ -o hapsolo_align.o$JOB_ID
#$ -e hapsolo_align.e$JOB_ID
#$ -pe openmp 36
#$ -l h_rt=24:00:00
###$ -q YOUR_QUEUE

# HapSolo step 2: minimap2 self-alignment using the published HapSolo parameters.
#
# Single multi-threaded job (no array). Output is auto-gzipped at the end.
#
# Usage:
#   1. Edit REF, HAPSOLO_DIR, and PYTHON below.
#   2. Make sure minimap2 is on PATH (load module or install in /usr/local/bin).
#   3. Submit:  qsub qsub_align.sh
#
# Output:
#   ${REF%.fasta}_self_align.paf.gz

set -euo pipefail

# === EDIT THESE ===
REF=myassembly_new.fasta
HAPSOLO_DIR=/path/to/HapSolo
PYTHON=python3
# ==================

cd "$SGE_O_WORKDIR"

# Optional: load minimap2 from your HPC's module system
###module load minimap2

$PYTHON $HAPSOLO_DIR/hapsolo_cli.py align \
    -i ${REF} \
    -t ${NSLOTS}

echo "Self-alignment complete."
echo "  Output: ${REF%.fasta}_self_align.paf.gz"
