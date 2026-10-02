#!/bin/bash
#SBATCH -J busco                        # jobname
#SBATCH -o busco/busco.o%A.%a           # stdout: %A=jobid, %a=arraytaskid
#SBATCH -e busco/busco.e%A.%a           # stderr
#SBATCH --array=0-3                     # UPDATE: 0 to (number of jobs - 1)
#SBATCH -A TG-MCB180035
#SBATCH -N 1
#SBATCH -p shared
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=4G
#SBATCH -t 12:00:00
#SBATCH --mail-user=solarese@uci.edu
#SBATCH --mail-type=begin,end
#SBATCH --export=ALL

# BUSCO validation on SDSC Expanse
#
# Each SLURM array task runs BUSCO on one assembly+lineage combination.
# Define all combinations in the JOBS array below.
#
# Usage:
#   1. Edit JOBS array (one "assembly lineage" pair per line)
#   2. Update --array to match: 0 to (${#JOBS[@]} - 1)
#   3. mkdir -p busco   (for log files)
#   4. sbatch sbatch_sdsc_busco.sh

set -euo pipefail

CPUS=$SLURM_CPUS_ON_NODE

module purge
module load singularitypro

# ── Singularity image ─────────────────────────────────────────
SINGULARITY_IMAGE=~/esolares/singularity_images/busco6.1.sif
export SINGULARITYENV_TINI_SUBREAPER=1

# ── BUSCO download path (pre-downloaded lineage datasets) ─────
# Set to a shared location with lineage data, or leave empty to
# let BUSCO download on the fly (requires internet on compute node).
DOWNLOAD_PATH=""

# ── Job array: "assembly_path lineage_name" per entry ─────────
# Paths relative to submission directory.
# Lineage examples: diptera_odb12.2, diptera_odb10, metazoa_odb12.2,
#                   embryophyta_odb10, actinopterygii_odb12.2
JOBS=(
    "assemblies/species1.fasta diptera_odb12.2"
    "assemblies/species1_reduced.fasta diptera_odb12.2"
    "assemblies/species2.fasta metazoa_odb12.2"
    "assemblies/species2_reduced.fasta metazoa_odb12.2"
)

# ── Parse job entry ───────────────────────────────────────────
JOB="${JOBS[$SLURM_ARRAY_TASK_ID]}"
ASM=$(echo "$JOB" | awk '{print $1}')
LINEAGE=$(echo "$JOB" | awk '{print $2}')

ASM_BASE=$(basename "$ASM" .fasta)
OUTNAME="${ASM_BASE}_${LINEAGE}"
OUTDIR="busco/${OUTNAME}"

EXEC="singularity exec --bind $(pwd):/home/$USER $SINGULARITY_IMAGE"

echo "================================================================"
echo "$(date) START: BUSCO validation"
echo "  Assembly: $ASM"
echo "  Lineage:  $LINEAGE"
echo "  Output:   $OUTDIR"
echo "  CPUs:     $CPUS"
echo "  Task ID:  $SLURM_ARRAY_TASK_ID / Job: $SLURM_ARRAY_JOB_ID"
echo "================================================================"

DOWNLOAD_FLAG=""
if [[ -n "$DOWNLOAD_PATH" ]]; then
    DOWNLOAD_FLAG="--download_path $DOWNLOAD_PATH --offline"
fi

$EXEC busco \
    -i "$ASM" \
    -l "$LINEAGE" \
    -o "$OUTNAME" \
    --out_path busco \
    -m genome \
    -c "$CPUS" \
    $DOWNLOAD_FLAG

echo ""
echo "================================================================"
echo "$(date) DONE: $OUTNAME"

if [[ -f "${OUTDIR}/short_summary.specific.${LINEAGE}.${OUTNAME}.txt" ]]; then
    echo ""
    grep -E "C:|S:|D:|F:|M:" "${OUTDIR}/short_summary.specific.${LINEAGE}.${OUTNAME}.txt" || true
fi
echo "================================================================"
