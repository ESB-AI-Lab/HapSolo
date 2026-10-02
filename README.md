<img src=./hapsolo_logo.png width=50%/>

An optimization approach for removing secondary haplotigs during diploid genome assembly and scaffolding.

HapSolo runs a hill-climbing search over alignment filter thresholds (PID, query coverage, query/reference length ratio) to minimize a cost function based on conserved single-copy ortholog completeness scores. Three optimizer modes are available: a **random forward walk** (mode 0, default) that takes stochastic steps through the threshold space, a **steepest descent** (mode 2) that evaluates all 8 diagonal neighbors at each step and moves to the lowest-cost one, and **simulated annealing** (mode 3) with Metropolis acceptance, geometric cooling and an auto-calibrated initial temperature. Every mode can run on the CPU or, with `--gpu`, on an NVIDIA GPU. The result is a primary assembly with reduced haplotype duplication and a secondary assembly containing the purged haplotigs.

- **Methods** (algorithms, cost function, GPU implementation): [methods.md](methods.md)
- **Results** on *A. funestus*, *P. americana*, *V. vinifera* and *A. radiata*: [analysis.md](analysis.md)
- **Assembly statistics and plots**: [assembly_stats.py](#assembly-statistics-and-plots)

# Installation

## pip install (recommended)

```
git clone https://github.com/esolares/HapSolo.git
cd HapSolo
pip install .
```

This installs the `hapsolo` and `hapsolo-cli` commands system-wide (or in your virtualenv). You can then run:

```
hapsolo-cli preprocess -i assembly.fasta
hapsolo-cli train ...
hapsolo --help        # direct optimizer entry point
```

## Without installing

```
git clone https://github.com/esolares/HapSolo.git
cd HapSolo
pip install -r requirements.txt
python3 hapsolo_cli.py preprocess -i assembly.fasta
```

## Dependencies

**Python packages:** `pandas`, `numpy`, `tqdm` (installed automatically by `pip install .`, or manually via `pip install -r requirements.txt`).

**Optional:** `cupy` for GPU acceleration (`pip install cupy-cuda12x` — match your CUDA version, or `pip install .[gpu]`).

**External tools** (must be on `$PATH`):

Build from source (small, no external dependencies):

```
# minimap2 (required for self-alignment)
git clone https://github.com/lh3/minimap2
cd minimap2 && make
sudo cp minimap2 /usr/local/bin/
cd ..

# miniprot (required for ortholog gene search)
git clone https://github.com/lh3/miniprot
cd miniprot && make
sudo cp miniprot /usr/local/bin/
cd ..
```

Or download precompiled binaries:

```
# minimap2 release binaries
wget https://github.com/lh3/minimap2/releases/download/v2.28/minimap2-2.28_x64-linux.tar.bz2
tar xf minimap2-2.28_x64-linux.tar.bz2
sudo cp minimap2-2.28_x64-linux/minimap2 /usr/local/bin/
```

BLAT is supported as an alternative aligner if you prefer it. Download the precompiled binary from UCSC:

```
wget https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/blat/blat
chmod +x blat
sudo mv blat /usr/local/bin/
```

## Singularity / Apptainer

The HapSolo2 image bundles HapSolo2, minimap2 2.31, miniprot 0.18, BLAT, `assembly_stats.py` and the CUDA 12 runtime for GPU mode, so only the NVIDIA driver is needed on the host. Build it from the clean source tree in `dev/hapsolo2/`:

```
cd dev/hapsolo2
singularity build --fakeroot hapsolo2.sif hapsolo2.def
singularity test --nv hapsolo2.sif          # self-test, including a CuPy GPU check
```

Run it from the directory that holds your data (HapSolo does not accept absolute input paths). Add `--nv` for GPU runs:

```
singularity run hapsolo2.sif preprocess -i assembly.fasta
singularity run --nv hapsolo2.sif train -i assembly_new.fasta --paf assembly_new_self_align.paf.gz -b ortholog_output/ --mode 3 --gpu --totaliters 100000
singularity exec hapsolo2.sif assembly_stats.py -i assembly.fasta --genome-size 900m -n 12
singularity run-help hapsolo2.sif           # full usage
```

The image ignores Python packages installed in your home directory, so host packages cannot shadow its pinned versions. The exact tool versions are listed in `/opt/HapSolo/VERSIONS.txt` inside the image. The older image on the Sylabs library (`library://esolares/default/hapsolo:latest`) predates HapSolo2.

## OrthoDB Lineage Datasets

HapSolo classifies conserved orthologs against per-lineage datasets curated by the [BUSCO](https://busco.ezlab.org/) team from [OrthoDB](https://www.orthodb.org/) data. OrthoDB provides the underlying orthologous group definitions and protein sequences; the BUSCO team packages them into per-lineage subsets with the `scores_cutoff` and `lengths_cutoff` files needed for gene completeness classification. Choose the lineage that best matches your taxon.

Supported versions: **ODB10**, **ODB12.2** (recommended), and legacy **ODB9** (no longer available for download).

**Download** from the BUSCO data archive on S3:

```
# ODB10 example (Diptera)
wget https://busco-data.s3.amazonaws.com/lineages/diptera_odb10.2024-01-08.tar.gz
tar xzf diptera_odb10.2024-01-08.tar.gz

# ODB12.2 example (Embryophyta)
wget https://busco-data.s3.amazonaws.com/lineages/embryophyta_odb12.2.2026-05-13.tar.gz
tar xzf embryophyta_odb12.2.2026-05-13.tar.gz
```

Available lineages include `insecta`, `diptera`, `vertebrata`, `mammalia`, `embryophyta`, `fungi`, `metazoa`, `actinopterygii`, and many others. To find the exact filename for your lineage, check the BUSCO lineage list:

```
wget https://busco-data.s3.amazonaws.com/information/lineages_list.2026-05-26.txt.tar.gz
tar xzf lineages_list.2026-05-26.txt.tar.gz
```

**Version notes:**
- **ODB12.2** — latest version with updated ortholog groups and protein sequences. Recommended for new analyses.
- **ODB12** (without `.2`) — supported but **not recommended**. These datasets (`OrthoDB_version=12.1`) ship with a raw protein database containing many more ortholog groups than the `scores_cutoff` file covers, and they lack `lengths_cutoff` entirely. HapSolo automatically excludes groups without score cutoffs and warns about them, but ODB12.2 is the better choice.
- **ODB10** — well-tested, widely used in published studies. Fully supported.
- **ODB9** — legacy format with different file structure (`ancestral_variants/` instead of `refseq_db.faa`). Still supported if you have local copies. The original downloads have been taken offline by the BUSCO team, but are preserved on the [Internet Archive](https://archive.org/details/odb9-busco-lineages).

**ODB12 vs ODB12.2:** Despite similar names, these are completely different datasets with **zero overlapping ortholog group IDs**, different group counts, and different protein databases. Although HapSolo supports both, **we strongly recommend using ODB12.2 over ODB12**. ODB12 (`OrthoDB_version=12.1`) ships with a raw protein database containing many more ortholog groups than its `scores_cutoff` file covers, and it lacks `lengths_cutoff` entirely — meaning Complete vs Fragmented classification falls back to a less precise heuristic. ODB12.2 (`OrthoDB_version=12.2`) is fully curated with consistent protein, score, and length data. When downloading lineage datasets, look for filenames containing `_odb12.2.` (not just `_odb12.`).

# Quick Start

The unified CLI (`hapsolo_cli.py` or `hapsolo-cli` if installed) drives the entire pipeline through six subcommands:

```
hapsolo-cli preprocess -i assembly.fasta
hapsolo-cli align      -i assembly_new.fasta -t 8
hapsolo-cli search     -i assembly_new.fasta -l diptera_odb10/ -o ortholog_output/ -t 8
hapsolo-cli cache      -i assembly_new.fasta --paf assembly_new_self_align.paf.gz -b ortholog_output/
hapsolo-cli train      -i assembly_new.fasta --hap assembly_new_self_align.hap -b ortholog_output/ -t 32 -n 2000
```

The optimized primary and secondary assemblies, along with assembly statistics reports, are written to the `asms/` directory.

Run `hapsolo-cli` (no arguments) to see the full help, or `hapsolo-cli <subcommand> --help` for details on any step.

# Pipeline Steps

## 1. Preprocess

Clean FASTA headers (remove special characters, ensure uniqueness) and split contigs into individual files.

```
hapsolo-cli preprocess -i assembly.fasta [-m MAXCONTIG_MB]
```

| Flag | Description |
|---|---|
| `-i` | Input assembly FASTA |
| `-m` | Maximum contig size in Mb for individual file output (default: 10). Contigs larger than this are still in the main output FASTA but skipped for per-contig processing. |

**Output:**
- `assembly_new.fasta` — sanitized headers
- `contigs/` — individual per-contig FASTA files
- `contigs/name_mapping.tsv` — original-to-sanitized name lookup table

## 2. Align

Run all-by-all self-alignment to identify candidate haplotig pairs.

```
hapsolo-cli align -i assembly_new.fasta -t 8 [--aligner minimap2|blat] [--no-gzip]
```

| Flag | Description |
|---|---|
| `-i` | Preprocessed assembly FASTA |
| `-t` | Number of threads (default: 1) |
| `--aligner` | `minimap2` (default) or `blat` |
| `-o` | Output filename (default: auto-named) |
| `--no-gzip` | Skip the gzip compression step |

**Output:** A gzipped PAF (or PSL) file alongside the input assembly. HapSolo reads `.gz` files directly, so no decompression is needed for downstream steps.

The default minimap2 parameters are tuned for sensitive self-alignment with up to 50 secondary hits per query. These parameters reproduce the published HapSolo results and work well for most diploid assemblies.

### Tuning alignment parameters

Advanced users can run minimap2 manually with custom parameters and feed the result directly to `train`:

```
minimap2 [your custom params] assembly_new.fasta assembly_new.fasta | gzip > my_align.paf.gz
hapsolo-cli train -i assembly_new.fasta --paf my_align.paf.gz -b ortholog_output/
```

For reference, the default parameters used by `hapsolo align` are:

```
minimap2 -t <threads> -P -G 500k -k19 -w2 -A1 -B2 -O2,4 -E2,1 \
    -s200 -z200 -N50 --max-qlen 10000000 --min-occ-floor=100 --paf-no-hit \
    assembly_new.fasta assembly_new.fasta | gzip > assembly_new_self_align.paf.gz
```

Avoid the `-x asm5` preset for self-alignment — it is tuned for high-identity assembly-to-reference alignment and discards most haplotig candidates.

**About `-N`:** with `-P` in effect, minimap2 retains all chains and ignores `-N` (minimap2 2.31 manual: "Options -p and -N have no effect when this option is in use"), so lowering `-N` does not reduce run time or PAF size with these parameters. To shorten alignment on large assemblies, change the parameters that do apply, and compare the resulting primary assembly statistics (size, ortholog completeness, contig count) against a run with the default parameters before adopting them.

## 3. Search (Ortholog Classification)

Align OrthoDB protein profiles against each contig with miniprot, then classify each ortholog as Complete, Fragmented, or Missing per contig. This produces the ortholog completeness metrics consumed by the optimizer. Supports ODB9, ODB10, ODB12, and ODB12.2 lineage datasets (auto-detects format, handles gzipped protein files).

```
hapsolo-cli search -i assembly_new.fasta -l diptera_odb10/ -o ortholog_output/ -t 8
```

| Flag | Description |
|---|---|
| `-i` | Preprocessed assembly FASTA |
| `-l` | Path to the OrthoDB lineage dataset directory |
| `-o` | Output directory for per-contig classification files (default: `busco_output`) |
| `-t` | Number of threads for miniprot |
| `-j` / `--splice-model` | miniprot splice model: `0`=none, `1`=general, `2`=vertebrate/insect (default: 2) |
| `-B` / `--end-bonus` | miniprot end-of-alignment bonus; higher values reduce protein-end clipping (default: 25) |
| `--classify-params` | Classification filter parameters as comma-separated `key=value` pairs. Allowed: `sr` (min score/cutoff ratio, default 2.5), `cov` (min alignment coverage, default 0.7), `gap` (2nd contig score gap vs best, default 0.9). Example: `--classify-params "sr=2.5,cov=0.7,gap=0.9"` |

**Output:** Per-contig TSV files in the output directory, in the format expected by the optimizer.

## 4. Cache (optional)

Pre-parse alignment files into the HapSolo `.hap` format and verify ortholog data. This step is optional but saves time when running multiple training experiments on the same data — the expensive PAF/PSL parsing happens once.

```
hapsolo-cli cache -i assembly_new.fasta --paf self_align.paf.gz -b ortholog_output/
```

| Flag | Description |
|---|---|
| `-i` | Preprocessed assembly FASTA |
| `--paf` / `--psl` | Alignment file to cache as `.hap` |
| `-b` | Ortholog search output directory to verify |
| `--align-only` | Only cache alignment data (skip ortholog verification) |
| `--search-only` | Only verify ortholog data (skip alignment caching) |
| `--min-contig` | Minimum contig size filter (default: 1000) |

**Output:** A `.hap` file alongside the alignment file, plus verification that ortholog TSV files are present and parseable. Subsequent `train` and `classify` runs can use `--hap` instead of `--paf`/`--psl` to skip parsing.

## 5. Train

Run hill-climbing optimization to find the alignment filter thresholds that minimize the cost function. Each thread starts from a random initial point and explores independently.

```
hapsolo-cli train -i assembly_new.fasta --paf assembly_new_self_align.paf.gz -b ortholog_output/ -t 32 -n 2000
```

| Flag | Description |
|---|---|
| `-i` | Preprocessed assembly FASTA |
| `--paf` / `--psl` | Self-alignment file (gzipped or uncompressed) |
| `--hap` | Pre-cached HAP alignment file (from `cache` step) |
| `-b` | Ortholog classification directory (output of `search`) |
| `-t` | Number of parallel threads (default: 1). Total iterations = `-t × -n` |
| `-n` | Iterations per thread (default: 1000) |
| `-B` | Number of best candidate assemblies to return (default: 1) |
| `--min-contig` | Minimum contig size for primary assembly (default: 1000 bp) |
| `--mode` | Optimizer mode: `0` = random walk (default), `2` = steepest descent, `3` = simulated annealing |
| `-S` / `-D` / `-F` / `-M` | Cost function weights for Single, Duplicate, Fragmented, Missing orthologs |
| `--formula` | Custom cost formula (see `python -m hapsolo --help` for syntax) |
| `--outdir` | Output directory for assembly files (default: `asms`) |

**Simulated annealing flags** (mode 3 only):

| Flag | Description |
|---|---|
| `--sa-temp` | Initial temperature (default: auto-calibrate from data) |
| `--sa-cooling` | Cooling rate alpha for geometric schedule (default: auto from iterations) |
| `--sa-min-temp` | Minimum temperature floor (default: 1e-6) |

**Cost function:** `(F·θF + D·θD + M·θM) / (S·θS)`

Defaults: θS=1.0, θD=1.0, θF=0.0, θM=1.0

Lower scores are better. The optimizer minimizes duplicates and missing orthologs while maximizing single-copy orthologs.

**Recommended weights:** For most assemblies, increasing the missing and duplicate penalty may improve results. We recommend `-M 2 -D 1.5` as a starting point — this penalizes missing orthologs 2× and duplicates 1.5× relative to the default, which encourages the optimizer to retain contigs carrying unique orthologs while still filtering redundant haplotigs:

```
hapsolo-cli train -i assembly_new.fasta --paf self_align.paf.gz -b ortholog_output/ -t 16 -n 2000 -M 2 -D 1.5
```

### Optimizer modes

**Mode 0: random forward walk (default).** Each iteration adds 0, 1 or 2 steps (step = 0.0001) to each threshold and always moves to the new point, so each walker moves from permissive to stringent thresholds; it restarts at a random point when it stalls on a plateau or leaves the search space. With enough iterations (1,000 to 2,000 per thread) the walkers cover the space broadly. On the GPU the step is drawn from -2 to +2 steps, so the walk is not forward-only; plateau restarts work as on the CPU.

```
hapsolo-cli train -i assembly_new.fasta --paf self_align.paf.gz -b ortholog_output/ -t 8 -n 2000
```

**Mode 2: steepest descent.** Each iteration evaluates the 8 diagonal neighbors (one step up or down in all three thresholds at once) and moves to the lowest-cost neighbor, even if it is worse than the current point, so it behaves as a greedy best-neighbor walk. It costs up to 8 evaluations per iteration against 1 for the other modes, and uses the same plateau and boundary restarts as mode 0 on both CPU and GPU.

```
hapsolo-cli train -i assembly_new.fasta --paf self_align.paf.gz -b ortholog_output/ --mode 2 -t 8 -n 2000
```

**Mode 3: simulated annealing.** Uses a Metropolis acceptance criterion to accept worse solutions with a probability that falls as the temperature cools on a geometric schedule, reaching the minimum temperature (1e-6) at the last iteration. If `--sa-temp` is not given, the initial temperature is calibrated from the data so that a typical worsening is accepted with probability 0.8. On a plateau the walker restarts at a random point and is reheated to T0 × 0.5^k at its k-th plateau, on both CPU and GPU (see [methods.md](methods.md)).

```
hapsolo-cli train -i assembly_new.fasta --paf self_align.paf.gz -b ortholog_output/ --mode 3 -t 8 -n 2000
```

All three modes produce the same output format and can be compared directly. In our benchmarks (100,000 iterations on the GPU, four species, three OrthoDB versions), the best costs of the three modes were within 1.8% of each other in all 15 comparisons, while mode 2 took 2.7 to 6.5 times longer than mode 0 ([analysis.md](analysis.md)).

### Performance

Both CPU and GPU paths avoid pandas in the optimization hot loop. At startup, alignment columns are extracted into flat arrays (numpy for CPU, cupy for GPU) and ortholog data is encoded as integer-indexed numpy arrays (`ScoringData`). The hot path — `evaluate_thresholds()`, called thousands of times per optimization — does boolean array filtering and vectorized scoring with zero pandas overhead.

On the pamer dataset (1,440 orthologs, 5,122 contigs, 49,285 alignments), compute-only benchmarks (mean ± stdev, n=5):

| Config | Time | Evals/s | Speedup |
|--------|------|---------|---------|
| Mode 0 CPU 100k (10 threads) | 148.28 ± 3.93s | 683 | — |
| Mode 0 GPU 100k (84 agents) | 26.52 ± 0.16s | 3,775 | 5.6x |
| Mode 2 CPU 100k (10 threads) | 1,441.68 ± 272.71s | 531 | — |
| Mode 2 GPU 100k (84 agents) | 189.56 ± 0.54s | 4,220 | 7.6x |

Mode 2 benefits most from GPU because each steepest-descent iteration evaluates 8 neighbors in a single batched operation. GPU timing variance is also dramatically lower (CV 0.3% vs 19% for CPU mode 2) because all walkers execute synchronously rather than gated by the slowest `pool.map()` worker.

### GPU acceleration

HapSolo supports GPU-accelerated hill climbing via [CuPy](https://cupy.dev/). The `--gpu` flag runs all walkers in a single CUDA context, evaluating threshold combinations simultaneously via batched cupy array operations. No cuDF is needed — alignment data is extracted as cupy arrays at startup and stays GPU-resident throughout optimization.

GPU flags are available through both `hapsolo train` and `python -m hapsolo` (the direct optimizer entry point):

```
pip install cupy-cuda12x   # match your CUDA version

# GPU with auto-detected agent count (default: SM count)
hapsolo train -i assembly_new.fasta --paf self_align.paf.gz -b ortholog_output/ \
    --mode 3 --gpu --totaliters 100000

# Same run through the optimizer directly
python3 -m hapsolo -i assembly_new.fasta --paf self_align.paf.gz -b ortholog_output/ \
    --gpu --totaliters 50000

# GPU with explicit agent count
python3 -m hapsolo -i assembly_new.fasta --paf self_align.paf.gz -b ortholog_output/ \
    --gpu --totaliters 50000 --agents 128
```

| Flag | Description |
|---|---|
| `--gpu` | Enable GPU acceleration (requires cupy) |
| `--totaliters` | Total iterations across all agents/threads. Auto-divides by agent count. Works with both GPU and CPU. |
| `--agents` | Number of GPU walkers (default: SM count from CUDA query). Capped at GPU memory limit based on dataset size. |
| `-n` | Iterations per agent/thread. `--totaliters` takes precedence if both are specified. |

**Agent count:** The default number of GPU walkers equals the GPU's SM (Streaming Multiprocessor) count (e.g., 84 on an RTX A6000, 108 on an A100). Each walker runs an independent optimization path. A user-supplied `--agents` is capped at 60% of GPU memory divided by 5 bytes × orthologs × contigs (the size of the dense scoring tensor used by mode 2), and HapSolo reports the cap if `--agents` exceeds it; the default is not capped. Modes 0 and 3 use a sync-free GPU loop that scores candidates with sparse and dense matrix products and never forms that tensor (see [methods.md](methods.md), section 8).

**`--totaliters` for fair comparison:** Use `--totaliters` to specify the total iteration budget. HapSolo auto-divides by the agent/thread count:
- GPU: `--totaliters 50000` with 84 SMs → 596 iters/agent × 84 agents = 50,064 total
- CPU: `--totaliters 50000` with `-t 10` → 5000 iters/thread × 10 threads = 50,000 total

**Output:** Primary and secondary assembly FASTAs in the `asms/` directory (or `--outdir`), with contigs in the same order as the input assembly (so identical runs give byte-identical files), a `_report.txt` with assembly statistics and ortholog completeness for each result, plus `.scores` and `.deltascores` files containing the cost trajectory of every iteration.

### Progress display

During training, each thread shows a live progress bar with its current parameters and score:

```
JOBID: 0  [██████████████░░░░░░░░░░░░░░░░] 460/1000   PID: 0.7521 QPMin: 0.6843 QRPMin: 0.5912 CostΔ +0.0023 Score: 0.4156
JOBID: 1  [████████████░░░░░░░░░░░░░░░░░░] 392/1000   PID: 0.6234 QPMin: 0.7102 QRPMin: 0.4587 CostΔ +0.0000 Score: 0.4892
JOBID: 2  [█████████████░░░░░░░░░░░░░░░░░] 437/1000   PID: 0.8104 QPMin: 0.5621 QRPMin: 0.6342 CostΔ +0.0011 Score: 0.4321
...
```

## 6. Classify

Apply fixed thresholds and write primary/secondary assemblies without running the hill-climbing optimization. Useful for reproducing prior results, applying known-good thresholds from a previous training run, or running quick what-if analyses.

```
hapsolo-cli classify -i assembly_new.fasta --paf assembly_new_self_align.paf.gz -b ortholog_output/
```

| Flag | Description |
|---|---|
| `-i` | Preprocessed assembly FASTA |
| `--paf` / `--psl` | Self-alignment file (gzipped or uncompressed) |
| `--hap` | Pre-cached HAP alignment file (from `cache` step) |
| `-b` | Ortholog classification directory (output of `search`) |
| `-P` / `--pid` | Fixed PID threshold (default: 0.7) |
| `-Q` / `--qpct` | Fixed query coverage threshold (default: 0.7) |
| `-R` / `--qrpct` | Fixed query/reference alignment length ratio threshold (default: 0.7) |
| `--min-contig` | Minimum contig size for primary assembly (default: 1000 bp) |
| `--outdir` | Output directory for assembly files (default: `asms`) |

**Default behavior** (no thresholds specified) — uses 0.7/0.7/0.7, matching the original HapSolo mode 1 defaults:

```
hapsolo-cli classify -i assembly_new.fasta --paf assembly_new_self_align.paf.gz -b ortholog_output/
```

**Custom thresholds** — use the exact values found by a previous `train` run, or any values you want to test:

```
hapsolo-cli classify -i assembly_new.fasta --paf assembly_new_self_align.paf.gz -b ortholog_output/ \
    -P 0.85 -Q 0.60 -R 0.50
```

The threshold values are encoded in the output filename so you can run `classify` repeatedly with different settings without overwriting earlier results.

# HPC Usage

## Choosing a Python interpreter

`hapsolo_cli.py` invokes sub-scripts using the same Python interpreter that launches it. On HPC systems with multiple Python versions installed (e.g., `python3`, `python3.9`, `python3.10`), choose your version in either of two ways:

**Option 1 — invoke directly:**
```
python3.10 hapsolo_cli.py train ...
```
All sub-scripts inherit `python3.10` automatically.

**Option 2 — use the `--python` flag:**
```
python3 hapsolo_cli.py --python python3.10 train ...
python3 hapsolo_cli.py --python /opt/python/3.11/bin/python3 train ...
```

This is useful when the parent script must use one interpreter but the workers need another.

## Example SLURM batch scripts

**CPU:**
```
#!/bin/bash
#SBATCH --job-name=hapsolo
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00

module load python/3.10 minimap2 miniprot

HAPSOLO=/path/to/HapSolo
ASM=my_assembly.fasta
LINEAGE=/path/to/diptera_odb10

python3 $HAPSOLO/hapsolo_cli.py preprocess -i $ASM
python3 $HAPSOLO/hapsolo_cli.py align      -i ${ASM%.fasta}_new.fasta -t $SLURM_CPUS_PER_TASK
python3 $HAPSOLO/hapsolo_cli.py search     -i ${ASM%.fasta}_new.fasta -l $LINEAGE -o ortholog_output -t $SLURM_CPUS_PER_TASK
python3 $HAPSOLO/hapsolo_cli.py train      -i ${ASM%.fasta}_new.fasta \
                                            --paf ${ASM%.fasta}_new_self_align.paf.gz \
                                            -b ortholog_output \
                                            -t $SLURM_CPUS_PER_TASK --totaliters 50000 --mode 2
```

**GPU:** (see also `scripts/sbatch_bridges2gpu.sh` and `scripts/sbatch_sdscgpu.sh` for PSC and SDSC examples)
```
#!/bin/bash
#SBATCH --job-name=hapsolo_gpu
#SBATCH -p gpu-shared
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

python3 -m hapsolo -i assembly_new.fasta \
    --paf assembly_new_self_align.paf.gz \
    -b ortholog_output/ \
    --gpu --totaliters 50000 --mode 2
```

# Running Steps Individually

The CLI is a thin wrapper. Each step can be invoked directly as Python modules:

```
# 1. Preprocess
python3 -m hapsolo.preprocess -i assembly.fasta

# 2. Self-alignment (use the parameters above, not -x asm5)
minimap2 -t 36 -P -G 500k -k19 -w2 -A1 -B2 -O2,4 -E2,1 -s200 -z200 -N50 \
    --max-qlen 10000000 --min-occ-floor=100 --paf-no-hit \
    assembly_new.fasta assembly_new.fasta | gzip > self_align.paf.gz

# 3. Ortholog classification
python3 -m hapsolo.search -i assembly_new.fasta -l diptera_odb10/ -o ortholog_output/ -t 8

# 4. Optimization
# mode 0 = random walk (default), mode 2 = steepest descent,
# mode 3 = simulated annealing, mode 1 = fixed thresholds
python3 -m hapsolo -i assembly_new.fasta --paf self_align.paf.gz -b ortholog_output/ \
    --mode 0 -t 32 -n 2000

# Steepest descent optimizer
python3 -m hapsolo -i assembly_new.fasta --paf self_align.paf.gz -b ortholog_output/ \
    --mode 2 -t 8 -n 2000

# Simulated annealing
python3 -m hapsolo -i assembly_new.fasta --paf self_align.paf.gz -b ortholog_output/ \
    --mode 3 -t 8 -n 2000

# GPU-accelerated (requires cupy)
python3 -m hapsolo -i assembly_new.fasta --paf self_align.paf.gz -b ortholog_output/ \
    --gpu --totaliters 50000
```

The legacy root-level scripts (`preprocessfasta.py`, `search_orthologs.py`) still work as backward-compatible stubs that delegate to the package modules.

# BLAT Alternative

HapSolo also accepts BLAT PSL alignment files. HPC batch scripts for BLAT are provided in the `scripts/` directory for SLURM (`sbatch_blat.sh`) and SGE (`qsub_blat.sh`). After BLAT array jobs complete, concatenate individual PSL files:

```
bash_andreaconcatpsl.sh myoutput_selfaln.PSL
hapsolo-cli train -i assembly_new.fasta --psl myoutput_selfaln.PSL.gz -b ortholog_output/
```

For the parallel BLAT implementation, see https://github.com/icebert/pblat

# Output Files

After a successful run, you will have:

```
asms/
  <assembly>_<minContig>_<PID>_<QRPctMin>to<QRPctMax>_<QPctMin>_primary.fasta
  <assembly>_<minContig>_<PID>_<QRPctMin>to<QRPctMax>_<QPctMin>_secondary.fasta
  <assembly>_<minContig>_<PID>_<QRPctMin>to<QRPctMax>_<QPctMin>_report.txt
<assembly>_<timestamp>.scores
<assembly>_<timestamp>.deltascores
```

The filenames encode the threshold values selected by the optimizer. The `_report.txt` contains assembly statistics at multiple size thresholds (contig counts, total lengths, N50/N75/L50/L75, GC content, N density) and ortholog completeness scores — comparable to QUAST + BUSCO output in a single file. The `.scores` and `.deltascores` files contain comma-separated cost values for every iteration of every thread, useful for plotting convergence curves. With `--gpu` the GPU loop records the cost of every walker at every iteration, so these files hold real trajectories in all modes.

# Contig Name Conversion

Contig names can mismatch between FASTA, alignment, and ortholog classification files when users skip preprocessing. The `build_conversion_dict()` function in `hapsolo/names.py` handles this with a 3-tier matching strategy:
1. **Exact match** — names are identical
2. **Sanitized match** — names match after replacing special characters with `_` (same regex as `hapsolo/preprocess.py`)
3. **Prefix match** — one sanitized name is a prefix of another (handles truncated names from preprocessing)

`hapsolo preprocess` writes `contigs/name_mapping.tsv` (original → sanitized name) for reference.

# Assembly statistics and plots

`assembly_stats.py` reports contiguity statistics for one or more FASTA files and draws a cumulative length plot, for example to compare contigs with the Hi-C scaffolds built from them:

```
python3 assembly_stats.py -i contigs.fasta yahs.out_scaffolds_final.fa \
    --name "Contigs" "HiC Scaffolds" --title "Persea americana (Thille) Assembly" \
    --prefix persea_americana_thille --genome-size 900m -n 12 -o plots/
```

| Option | Description |
|---|---|
| `-i/--input` | One or more FASTA files; several are compared side by side |
| `-o/--outdir` | Output directory (default: directory of the first input) |
| `--genome-size` | Expected genome size (`900m`, `2g`, `890000000`), used for NG50/NG75 and LG50/LG75 |
| `-n/--nchroms` | Expected chromosome number; reports the size and genome fraction of the top *n* sequences (Nxsome) |
| `--name` | Display name per input for the legend and table |
| `--prefix` | Output file prefix (required with several inputs) |
| `--title` | Plot title |
| `--no-plot` | Skip the plot |

Outputs are `<prefix>_assembly_stats.txt` (contig counts at size thresholds, total length, largest contig, N50/N75, L50/L75, NG50/NG75, GC%, N density and Nxsome) and `<prefix>_cumulative_length.png` and `.pdf`. The script imports the `hapsolo` package, so run it from the repository root, after `pip install`, or from the container (`singularity exec hapsolo2.sif assembly_stats.py ...`). `scripts/bash_thille_assembly_stats.sh` is a worked example.

# Limitations

- HapSolo does not accept absolute paths for input files. All file paths must be relative to the current working directory.
- Versions before 2026-10-01 overwrote an input assembly not named `*.fasta` (e.g. `*.fa`) with the `.scores`/`.deltascores` dumps. This is fixed; with older versions, name inputs `*.fasta`.
- The `.hap` alignment cache is reused whenever it exists, whatever prefilter (`--min`, `-P`, `-Q`, `-R`) the current run requests. Delete or rename the `.hap` when changing the prefilter. Since 2026-10-01 the cache is written atomically, so an interrupted run no longer leaves a partial cache behind.

# License

Apache License 2.0. See [LICENSE](LICENSE) for details.

# Bug Reports

Please submit issues at https://github.com/esolares/HapSolo/issues

## Citation

If you use HapSolo2, please cite:

Yin J\*, Agrawal M\*, Goel P\*, Solares EA. HapSolo2: GPU-accelerated optimization for removing secondary haplotigs from diploid genome assemblies. *bioRxiv*. 2026. Preprint. (\*These authors contributed equally.)

```bibtex
@misc{yin2026hapsolo2,
  author       = {Yin, Juan and Agrawal, Mansi and Goel, Prashansa and Solares, Edwin A.},
  title        = {{HapSolo2}: {GPU}-accelerated optimization for removing secondary haplotigs from diploid genome assemblies},
  howpublished = {bioRxiv},
  year         = {2026},
  note         = {Preprint}
}
```

For the original method, please also cite:

Solares EA, Tao Y, Long AD, Gaut BS. HapSolo: an optimization approach for removing secondary haplotigs during diploid genome assembly and scaffolding. *BMC Bioinformatics*. 2021;22:9. https://doi.org/10.1186/s12859-020-03939-y
