#!/bin/bash
#SBATCH -J hapsolo_train
#SBATCH -o hapsolo_train.o%j
#SBATCH -e hapsolo_train.e%j
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 48:00:00
###SBATCH -p YOUR_PARTITION

# HapSolo step 4: hill-climbing optimization to find best filter thresholds
# and write the primary/secondary assemblies.
#
# Each thread runs an independent random walk. Total iterations = -t * -n.
#
# Usage:
#   1. Edit REF, PAF, BUSCOS, ITERATIONS, HAPSOLO_DIR, and PYTHON below.
#   2. Submit:  sbatch sbatch_train.sh
#
# Output:
#   asms/                       - primary and secondary assembly FASTAs
#   ${REF%.fasta}_*.scores      - cost trajectory (one row per thread)
#   ${REF%.fasta}_*.deltascores - cost delta trajectory

set -euo pipefail

# === EDIT THESE ===
REF=myassembly_new.fasta
PAF=myassembly_new_self_align.paf.gz
BUSCOS=ortholog_output
ITERATIONS=2000
HAPSOLO_DIR=/path/to/HapSolo
PYTHON=python3
# ==================

cd "$SLURM_SUBMIT_DIR"

$PYTHON $HAPSOLO_DIR/hapsolo_cli.py train \
    -i ${REF} \
    --paf ${PAF} \
    -b ${BUSCOS} \
    -t ${SLURM_CPUS_PER_TASK} \
    -n ${ITERATIONS}

echo "Training complete."
echo "  Primary/secondary assemblies: asms/"
