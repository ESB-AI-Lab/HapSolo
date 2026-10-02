#!/bin/bash
#SBATCH -J hapsolo_pipeline
#SBATCH -o hapsolo_pipeline.o%j
#SBATCH -e hapsolo_pipeline.e%j
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 72:00:00
###SBATCH -p YOUR_PARTITION

# HapSolo full pipeline: preprocess -> align -> search -> train, in one job.
#
# This is convenient for small-to-medium assemblies. For large assemblies,
# consider running each step as a separate job (sbatch_preprocess.sh,
# sbatch_align.sh, sbatch_search.sh, sbatch_train.sh) so failures only
# require re-running the affected step.
#
# Usage:
#   1. Edit REF, LINEAGE, ITERATIONS, HAPSOLO_DIR, and PYTHON below.
#   2. Make sure minimap2 and miniprot are on PATH.
#   3. Submit:  sbatch sbatch_pipeline.sh

set -euo pipefail

# === EDIT THESE ===
REF=myassembly.fasta
LINEAGE=/path/to/diptera_odb10
ITERATIONS=2000
HAPSOLO_DIR=/path/to/HapSolo
PYTHON=python3
# ==================

cd "$SLURM_SUBMIT_DIR"

# Optional: load tools from your HPC's module system
###module load minimap2 miniprot

NEW_REF=${REF%.fasta}_new.fasta
PAF=${REF%.fasta}_new_self_align.paf.gz
ORTHOLOG_DIR=ortholog_output

echo "=== Step 1: Preprocess ==="
$PYTHON $HAPSOLO_DIR/hapsolo_cli.py preprocess -i ${REF}

echo "=== Step 2: Self-alignment ==="
$PYTHON $HAPSOLO_DIR/hapsolo_cli.py align \
    -i ${NEW_REF} \
    -t ${SLURM_CPUS_PER_TASK}

echo "=== Step 3: Ortholog search ==="
$PYTHON $HAPSOLO_DIR/hapsolo_cli.py search \
    -i ${NEW_REF} \
    -l ${LINEAGE} \
    -o ${ORTHOLOG_DIR} \
    -t ${SLURM_CPUS_PER_TASK}

echo "=== Step 4: Hill-climbing optimization ==="
$PYTHON $HAPSOLO_DIR/hapsolo_cli.py train \
    -i ${NEW_REF} \
    --paf ${PAF} \
    -b ${ORTHOLOG_DIR} \
    -t ${SLURM_CPUS_PER_TASK} \
    -n ${ITERATIONS}

echo "=== Pipeline complete ==="
echo "  Primary/secondary assemblies: asms/"
