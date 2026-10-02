#!/bin/bash
#SBATCH -J hapsolo                      # jobname
#SBATCH -o hapsolo/hapsolo.o%A.%a       # stdout: %A=jobid, %a=arraytaskid
#SBATCH -e hapsolo/hapsolo.e%A.%a       # stderr
#SBATCH --array=0-2                     # UPDATE: 0 to (number of assemblies - 1)
#SBATCH -A TG-MCB180035
#SBATCH -N 1
#SBATCH -p shared
#SBATCH --ntasks-per-node=32
#SBATCH --mem-per-cpu=4G
#SBATCH -t 48:00:00
#SBATCH --mail-user=esolares80@gmail.com
#SBATCH --mail-type=begin,end,fail
#SBATCH --export=ALL

# HapSolo CPU pipeline on SDSC Expanse (preprocess + align + search)
#
# Each SLURM array task processes one assembly through the CPU steps:
#   preprocess → self-align → ortholog search
# Toggle individual steps with DO_* flags below.
#
# Training is a separate job — use sbatch_sdsc_hapsolo_train.sh
# (submits to gpu-shared when USE_GPU=1).
#
# Usage:
#   1. Edit ASSEMBLIES array and LINEAGE below
#   2. Update --array to match: 0 to (${#ASSEMBLIES[@]} - 1)
#   3. mkdir -p hapsolo   (for log files)
#   4. sbatch sbatch_sdsc_hapsolo.sh

set -euo pipefail

CPUS=$SLURM_CPUS_ON_NODE

module purge
module load singularitypro

# ── Singularity image ─────────────────────────────────────────
SINGULARITY_IMAGE=~/esolares/singularity_images/hapsolo2.sif
export SINGULARITYENV_TINI_SUBREAPER=1

# ── Assembly array (one per array task) ───────────────────────
ASSEMBLIES=(
    "assemblies/species1.fasta"
    "assemblies/species2.fasta"
    "assemblies/species3.fasta"
)

# ── Ortholog lineage dataset ──────────────────────────────────
LINEAGE="diptera_odb10"

# ── Pipeline steps (1=run, 0=skip) ────────────────────────────
DO_PREPROCESS=1
DO_ALIGN=1
DO_SEARCH=1

# ── Resolve paths ─────────────────────────────────────────────
ASM="${ASSEMBLIES[$SLURM_ARRAY_TASK_ID]}"
ASM_DIR=$(dirname "$ASM")
ASM_BASE=$(basename "$ASM" .fasta)
ASM_NEW="${ASM_DIR}/${ASM_BASE}_new.fasta"
PAF="${ASM_DIR}/${ASM_BASE}_new_self_align.paf"
ORTHO_DIR="${ASM_DIR}/orthologs_${ASM_BASE}"

EXEC="singularity exec --bind $(pwd):$(pwd) --pwd $(pwd) $SINGULARITY_IMAGE"

echo "================================================================"
echo "$(date) START: HapSolo pipeline (CPU) for ${ASM_BASE}"
echo "  Assembly:   $ASM"
echo "  Lineage:    $LINEAGE"
echo "  CPUs:       $CPUS"
echo "  Task ID:    $SLURM_ARRAY_TASK_ID / Job: $SLURM_ARRAY_JOB_ID"
echo "================================================================"

# ── Step 1: Preprocess ────────────────────────────────────────
if [[ $DO_PREPROCESS -eq 1 ]]; then
    echo ""
    echo "$(date) Step 1/3: Preprocessing"
    $EXEC hapsolo preprocess -i "$ASM"
    echo "  Output: $ASM_NEW"
fi

# ── Step 2: Self-alignment (minimap2) ─────────────────────────
if [[ $DO_ALIGN -eq 1 ]]; then
    echo ""
    echo "$(date) Step 2/3: Self-alignment (minimap2, $CPUS threads)"
    $EXEC hapsolo align -i "$ASM_NEW" -t "$CPUS"
    echo "  Output: $PAF"
fi

# ── Step 3: Ortholog search (miniprot) ────────────────────────
if [[ $DO_SEARCH -eq 1 ]]; then
    echo ""
    echo "$(date) Step 3/3: Ortholog search ($LINEAGE)"
    mkdir -p "$ORTHO_DIR"
    $EXEC hapsolo search \
        -i "$ASM_NEW" \
        -l "$LINEAGE" \
        -o "$ORTHO_DIR" \
        -t "$CPUS"
    echo "  Output: $ORTHO_DIR"
fi

echo ""
echo "================================================================"
echo "$(date) DONE: ${ASM_BASE}"
echo "  Next: sbatch sbatch_sdsc_hapsolo_train.sh"
echo "================================================================"
