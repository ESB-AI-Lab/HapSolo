# HapSolo HPC Job Scripts

Job submission templates for SLURM and SGE/UGE schedulers.

## Layout

```
scripts/
├── sbatch_preprocess.sh   SLURM: clean FASTA headers, split contigs
├── sbatch_align.sh        SLURM: minimap2 self-alignment
├── sbatch_search.sh       SLURM: ortholog search with miniprot
├── sbatch_train.sh        SLURM: hill-climbing optimization
├── sbatch_pipeline.sh     SLURM: full pipeline in one job
│
├── qsub_preprocess.sh     SGE: clean FASTA headers, split contigs
├── qsub_align.sh          SGE: minimap2 self-alignment
├── qsub_search.sh         SGE: ortholog search with miniprot
├── qsub_train.sh          SGE: hill-climbing optimization
├── qsub_pipeline.sh       SGE: full pipeline in one job
│
└── legacy/                Legacy BLAT array-job scripts and the old BUSCO V3
                           pipeline. Kept for reproducing published results.
                           See legacy/README.md for details.
```

## Quick Start

Each script has an `=== EDIT THESE ===` block at the top. Change the variables for your data and submit.

```
# Edit the variables in each script first
vi sbatch_preprocess.sh sbatch_align.sh sbatch_search.sh sbatch_train.sh

# Submit them in order, chaining with --dependency=afterok:
JID1=$(sbatch --parsable sbatch_preprocess.sh)
JID2=$(sbatch --parsable --dependency=afterok:$JID1 sbatch_align.sh)
JID3=$(sbatch --parsable --dependency=afterok:$JID2 sbatch_search.sh)
JID4=$(sbatch --parsable --dependency=afterok:$JID3 sbatch_train.sh)
echo "Submitted: $JID1 $JID2 $JID3 $JID4"
```

Or run all four steps in a single job using the pipeline scripts:

```
sbatch sbatch_pipeline.sh
# or
qsub qsub_pipeline.sh
```

## Resource Recommendations

| Step       | Cores | Memory | Wall time | Notes |
|------------|-------|--------|-----------|-------|
| preprocess | 1     | 8 GB   | 2 h       | Single-threaded, IO-bound |
| align      | 32+   | 64 GB  | 24 h      | minimap2 scales linearly with threads |
| search     | 16    | 32 GB  | 12 h      | miniprot is fast; CPU-bound |
| train      | 32+   | 128 GB | 48 h      | Each thread = independent random walk; total iterations = `-t * -n` |

These are starting points for a ~1 Gb diploid genome. Adjust for your dataset size.

## Common Edits

Every script has the same `=== EDIT THESE ===` block:

```
REF=myassembly.fasta            # input assembly (or _new.fasta for steps 2-4)
HAPSOLO_DIR=/path/to/HapSolo    # where you cloned the repo
PYTHON=python3                  # python3.10, python3.11, or full path
```

For step 3 (`search`):
```
LINEAGE=/path/to/diptera_odb10  # OrthoDB lineage dataset for your taxon
OUTPUT_DIR=ortholog_output      # where to write classification TSVs
```

For step 4 (`train`):
```
PAF=myassembly_new_self_align.paf.gz  # output from align step
BUSCOS=ortholog_output                 # output from search step
ITERATIONS=2000                        # iterations per thread
```

## Loading Tools from HPC Modules

If your HPC uses environment modules, uncomment the `module load` lines at the top of each script:

```
module load python/3.11
module load minimap2
module load miniprot
```

Or use Singularity (see `../singularity/README.md` for the image build instructions):

```
singularity run --bind $PWD:/data /path/to/hapsolo.sif preprocess -i /data/assembly.fasta
```

## Legacy Scripts

The `legacy/` subdirectory contains the original BLAT array-job scripts and the BUSCO V3 pipeline. These are no longer the recommended workflow but are kept for users reproducing published results that depend on those exact tools. See `legacy/README.md` for usage.
