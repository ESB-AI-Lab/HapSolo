#!/bin/bash
#$ -N hapsolo_pipeline
#$ -o hapsolo_pipeline.o$JOB_ID
#$ -e hapsolo_pipeline.e$JOB_ID
#$ -pe openmp 32
#$ -l h_rt=72:00:00
###$ -q YOUR_QUEUE

# HapSolo full pipeline: preprocess -> align -> search -> train, in one job.
#
# This is convenient for small-to-medium assemblies. For large assemblies,
# consider running each step as a separate job (qsub_preprocess.sh,
# qsub_align.sh, qsub_search.sh, qsub_train.sh) so failures only
# require re-running the affected step.
#
# Usage:
#   1. Edit REF, LINEAGE, ITERATIONS, HAPSOLO_DIR, and PYTHON below.
#   2. Make sure minimap2 and miniprot are on PATH.
#   3. Submit:  qsub qsub_pipeline.sh

set -euo pipefail

# === EDIT THESE ===
REF=myassembly.fasta
LINEAGE=/path/to/diptera_odb10
ITERATIONS=2000
HAPSOLO_DIR=/path/to/HapSolo
PYTHON=python3
# ==================

cd "$SGE_O_WORKDIR"

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
    -t ${NSLOTS}

echo "=== Step 3: Ortholog search ==="
$PYTHON $HAPSOLO_DIR/hapsolo_cli.py search \
    -i ${NEW_REF} \
    -l ${LINEAGE} \
    -o ${ORTHOLOG_DIR} \
    -t ${NSLOTS}

echo "=== Step 4: Hill-climbing optimization ==="
$PYTHON $HAPSOLO_DIR/hapsolo_cli.py train \
    -i ${NEW_REF} \
    --paf ${PAF} \
    -b ${ORTHOLOG_DIR} \
    -t ${NSLOTS} \
    -n ${ITERATIONS}

echo "=== Pipeline complete ==="
echo "  Primary/secondary assemblies: asms/"
