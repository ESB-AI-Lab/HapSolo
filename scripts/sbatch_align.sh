#!/bin/bash
#SBATCH -J hapsolo_align
#SBATCH -o hapsolo_align.o%j
#SBATCH -e hapsolo_align.e%j
#SBATCH -c 36
#SBATCH --mem=64G
#SBATCH -t 24:00:00
###SBATCH -p YOUR_PARTITION

# HapSolo step 2: minimap2 self-alignment using the published HapSolo parameters.
#
# Single multi-threaded job (no array). Output is auto-gzipped at the end.
#
# Usage:
#   1. Edit REF, HAPSOLO_DIR, and PYTHON below.
#   2. Make sure minimap2 is on PATH (load module or install in /usr/local/bin).
#   3. Submit:  sbatch sbatch_align.sh
#
# Output:
#   ${REF%.fasta}_self_align.paf.gz

set -euo pipefail

# === EDIT THESE ===
REF=myassembly_new.fasta
HAPSOLO_DIR=/path/to/HapSolo
PYTHON=python3
# ==================

cd "$SLURM_SUBMIT_DIR"

# Optional: load minimap2 from your HPC's module system
###module load minimap2

$PYTHON $HAPSOLO_DIR/hapsolo_cli.py align \
    -i ${REF} \
    -t ${SLURM_CPUS_PER_TASK}

echo "Self-alignment complete."
echo "  Output: ${REF%.fasta}_self_align.paf.gz"
