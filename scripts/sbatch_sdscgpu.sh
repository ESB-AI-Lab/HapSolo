#!/bin/bash
#SBATCH -J hapsolo_gpu
#SBATCH -o hapsolo_gpu.o%j
#SBATCH -e hapsolo_gpu.e%j
#SBATCH -A TG-MCB180035
#SBATCH -N 1
#SBATCH -p gpu-shared
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --mail-user=esolares80@gmail.com
#SBATCH --mail-type=fail
#SBATCH --export=ALL

# HapSolo GPU benchmark on SDSC Expanse
#
# GPU options: --gres=gpu:1 (default v100), --gres=gpu:a100:1, --gres=gpu:h100:1
# Adjust --gres above to select GPU type.
#
# This script benchmarks GPU hill-climbing with --totaliters and auto-detected
# max concurrent agents. It also runs a CPU comparison on the same node.
#
# Usage:
#   1. Edit REF, PAF, BUSCOS, HAPSOLO_DIR below.
#   2. Submit:  sbatch sbatch_sdscgpu.sh
#
# Output:
#   benchmark_sdsc_${SLURM_JOB_ID}.csv  — timing + score results
#   asms/                                — primary/secondary FASTAs (from last run)
#   *.scores / *.deltascores             — cost trajectories

set -euo pipefail

module purge
module load gpu
module load singularitypro

# === EDIT THESE ===
REF=example_data/pamer.contigs.c21.consensus.consensus_pilon_pilon_new.fasta
PAF=example_data/pamer_self_align.paf
BUSCOS=example_data/odbaln_output
HAPSOLO_DIR=${SLURM_SUBMIT_DIR}/..
PYTHON=python3
SINGULARITY_IMAGE=""
# ==================

cd "$HAPSOLO_DIR"
mkdir -p asms

OUTCSV="benchmark_sdsc_${SLURM_JOB_ID}.csv"
echo "platform,mode,total_iters,agents,iters_per_agent,wall_seconds,best_score,gpu_name" > "$OUTCSV"

hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ',' '_')

RUN_CMD="$PYTHON -m hapsolo"
if [ -n "$SINGULARITY_IMAGE" ]; then
    RUN_CMD="singularity exec --nv $SINGULARITY_IMAGE $PYTHON -m hapsolo"
fi

for MODE in 0 2; do
    MODE_NAME="random_walk"
    [ "$MODE" -eq 2 ] && MODE_NAME="steepest_descent"

    for TOTAL in 1000 5000 10000 50000; do
        echo ""
        echo "=== Mode ${MODE} (${MODE_NAME}), total_iters=${TOTAL} ==="

        # GPU run (auto-detect agents from CUDA)
        echo "  GPU run..."
        START=$(date +%s%N)
        GPU_OUT=$($RUN_CMD \
            -i "$REF" -a "$PAF" -b "$BUSCOS" \
            --mode "$MODE" \
            --totaliters "$TOTAL" \
            --gpu 2>&1)
        END=$(date +%s%N)
        GPU_SECS=$(echo "scale=3; ($END - $START) / 1000000000" | bc)
        GPU_SCORE=$(echo "$GPU_OUT" | grep -oP 'with score: \K[\d.e+-]+' | tail -1)
        GPU_AGENTS=$(echo "$GPU_OUT" | grep -oP '(\d+) agents' | head -1 | grep -oP '\d+')
        GPU_IPA=$(echo "$GPU_OUT" | grep -oP '(\d+) iters/agent' | head -1 | grep -oP '\d+')
        echo "    Time: ${GPU_SECS}s, Score: ${GPU_SCORE}, Agents: ${GPU_AGENTS}"
        echo "gpu,${MODE},${TOTAL},${GPU_AGENTS:-0},${GPU_IPA:-0},${GPU_SECS},${GPU_SCORE:-NA},${GPU_NAME}" >> "$OUTCSV"

        # CPU run
        CPU_THREADS=${SLURM_CPUS_PER_TASK:-10}
        [ "$CPU_THREADS" -gt 16 ] && CPU_THREADS=16
        echo "  CPU run (${CPU_THREADS} threads)..."
        START=$(date +%s%N)
        CPU_OUT=$($RUN_CMD \
            -i "$REF" -a "$PAF" -b "$BUSCOS" \
            --mode "$MODE" \
            --totaliters "$TOTAL" \
            -t "$CPU_THREADS" 2>&1)
        END=$(date +%s%N)
        CPU_SECS=$(echo "scale=3; ($END - $START) / 1000000000" | bc)
        CPU_SCORE=$(echo "$CPU_OUT" | grep -oP 'with score: \K[\d.e+-]+' | tail -1)
        CPU_IPA=$(echo "$CPU_OUT" | grep -oP '(\d+) iters/thread' | head -1 | grep -oP '\d+')
        echo "    Time: ${CPU_SECS}s, Score: ${CPU_SCORE}, Threads: ${CPU_THREADS}"
        echo "cpu,${MODE},${TOTAL},${CPU_THREADS},${CPU_IPA:-0},${CPU_SECS},${CPU_SCORE:-NA},${GPU_NAME}" >> "$OUTCSV"

        SPEEDUP=$(echo "scale=2; $CPU_SECS / $GPU_SECS" | bc 2>/dev/null || echo "NA")
        echo "  Speedup: ${SPEEDUP}x"
    done
done

echo ""
echo "=== BENCHMARK RESULTS ==="
column -t -s',' "$OUTCSV"
echo ""
echo "Results saved to: $OUTCSV"
echo "Done."
