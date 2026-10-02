#!/bin/bash
#SBATCH -J hapsolo_train                    # jobname
#SBATCH -o hapsolo/train.o%A.%a             # stdout: %A=jobid, %a=arraytaskid
#SBATCH -e hapsolo/train.e%A.%a             # stderr
#SBATCH --array=0-2                         # UPDATE: 0 to (number of assemblies - 1)
#SBATCH -A TG-MCB180035
#SBATCH -N 1
#SBATCH -p gpu-shared
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=4G
#SBATCH -t 12:00:00
#SBATCH --mail-user=esolares80@gmail.com
#SBATCH --mail-type=begin,end,fail
#SBATCH --export=ALL

# HapSolo GPU training on SDSC Expanse
#
# Run AFTER sbatch_sdsc_hapsolo.sh completes (needs preprocessed FASTA,
# self-alignment PAF, and ortholog search output).
#
# For CPU-only training: change to -p shared, remove --gres, set USE_GPU=0.
#
# Usage:
#   1. Edit ASSEMBLIES array and parameters below (must match CPU script)
#   2. Update --array to match: 0 to (${#ASSEMBLIES[@]} - 1)
#   3. mkdir -p hapsolo   (for log files)
#   4. sbatch sbatch_sdsc_hapsolo_train.sh

set -euo pipefail

CPUS=$SLURM_CPUS_ON_NODE

module purge
module load gpu
module load singularitypro

# ── Singularity image ─────────────────────────────────────────
SINGULARITY_IMAGE=~/esolares/singularity_images/hapsolo2.sif
export SINGULARITYENV_TINI_SUBREAPER=1

# ── Assembly array (must match CPU script) ────────────────────
ASSEMBLIES=(
    "assemblies/species1.fasta"
    "assemblies/species2.fasta"
    "assemblies/species3.fasta"
)

# ── Training parameters ──────────────────────────────────────
MODE=3                          # 0=random walk, 1=fixed, 2=steepest descent, 3=SA
TOTAL_ITERS=100000              # total iterations across all agents/threads
THETA_S=1.0
THETA_D=1.0
THETA_F=0.0
THETA_M=1.0
USE_GPU=1                       # 1=GPU (needs gpu-shared + --gres), 0=CPU only

# ── Resolve paths (must match CPU script layout) ──────────────
ASM="${ASSEMBLIES[$SLURM_ARRAY_TASK_ID]}"
ASM_DIR=$(dirname "$ASM")
ASM_BASE=$(basename "$ASM" .fasta)
ASM_NEW="${ASM_DIR}/${ASM_BASE}_new.fasta"
PAF="${ASM_DIR}/${ASM_BASE}_new_self_align.paf"
ORTHO_DIR="${ASM_DIR}/orthologs_${ASM_BASE}"
OUTDIR="asms/${ASM_BASE}"

mkdir -p "$OUTDIR"

EXEC="singularity exec --nv --bind $(pwd):$(pwd) --pwd $(pwd) $SINGULARITY_IMAGE"

echo "================================================================"
echo "$(date) START: HapSolo training for ${ASM_BASE}"
echo "  Assembly:   $ASM_NEW"
echo "  PAF:        $PAF"
echo "  Orthologs:  $ORTHO_DIR"
echo "  Mode:       $MODE"
echo "  Iterations: $TOTAL_ITERS"
echo "  Thetas:     S=$THETA_S D=$THETA_D F=$THETA_F M=$THETA_M"
echo "  GPU:        $USE_GPU"
echo "  Task ID:    $SLURM_ARRAY_TASK_ID / Job: $SLURM_ARRAY_JOB_ID"
echo "================================================================"

if [[ $USE_GPU -eq 1 ]]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

GPU_FLAG=""
THREAD_FLAG="-t $CPUS"
if [[ $USE_GPU -eq 1 ]]; then
    GPU_FLAG="--gpu"
    THREAD_FLAG=""
fi

$EXEC hapsolo train \
    -i "$ASM_NEW" \
    --paf "$PAF" \
    -b "$ORTHO_DIR" \
    --mode "$MODE" \
    -n "$TOTAL_ITERS" \
    -S "$THETA_S" -D "$THETA_D" -F "$THETA_F" -M "$THETA_M" \
    --outdir "$OUTDIR" \
    $GPU_FLAG $THREAD_FLAG

echo ""
echo "================================================================"
echo "$(date) DONE: ${ASM_BASE}"
echo "  Output: $OUTDIR"
echo "================================================================"
