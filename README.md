<img src=./hapsolo_logo.png width=50%/>

An optimization approach for removing secondary haplotigs during diploid genome assembly and scaffolding.

HapSolo runs a hill-climbing search over alignment filter thresholds (PID, query coverage, query/reference length ratio) to minimize a cost function based on conserved single-copy ortholog completeness scores. Two optimizer modes are available: a **random walk** (mode 0, default) that takes stochastic steps through the threshold space, and a **steepest descent** (mode 2) that evaluates all 8 octant neighbors at each step and moves to the lowest-cost position. Both use plateau detection and boundary resets to escape local minima. The result is a primary assembly with reduced haplotype duplication and a secondary assembly containing the purged haplotigs.<br><br>
***Note: OrthoDB v9 BUSCO lineages have been taken offline but I have preserved them on the Internet Archive*** [https://archive.org/details/odb9-busco-lineages](https://archive.org/details/odb9-busco-lineages) ***or Zenodo*** [https://doi.org/10.5281/zenodo.20987183](https://doi.org/10.5281/zenodo.20987183)

# Singularity Installation (Preferred)
```
singularity pull --arch amd64 library://esolares/default/hapsolo:latest
singularity exec --bind $(pwd):/home/$USER hapsolo_latest.sif hapsolo
```

# Installation

```
git clone https://github.com/esolares/HapSolo.git
cd HapSolo
```

## Dependencies

HapSolo requires Python 3 and the following:

```
# Python packages
pip install -r requirements.txt
```

Build the alignment tools from source (small, no external dependencies):

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

## OrthoDB Lineage Datasets

HapSolo classifies conserved orthologs against an OrthoDB lineage dataset. Choose the lineage that best matches your taxon. Supported versions: ODB9, ODB10, ODB11.

```
# ODB10 example (Diptera)
wget https://busco-data.ezlab.org/v5/data/lineages/diptera_odb10.2024-01-08.tar.gz
tar xzf diptera_odb10.2024-01-08.tar.gz
```

Browse other lineages (insecta, vertebrata, mammalia, embryophyta, etc.):
- ODB9: https://busco-data.ezlab.org/v3/data/lineages/
- ODB10: https://busco-data.ezlab.org/v5/data/lineages/

# Quick Start

The unified CLI (`hapsolo_cli.py`) drives the entire pipeline through five subcommands:

```
python3 hapsolo_cli.py preprocess -i assembly.fasta
python3 hapsolo_cli.py align      -i assembly_new.fasta -t 8
python3 hapsolo_cli.py search     -i assembly_new.fasta -l diptera_odb10/ -o ortholog_output/ -t 8
python3 hapsolo_cli.py train      -i assembly_new.fasta --paf assembly_new_self_align.paf.gz -b ortholog_output/ -t 32 -n 2000
```

The optimized primary and secondary assemblies, along with assembly statistics reports, are written to the `asms/` directory.

Run `python3 hapsolo_cli.py` (no arguments) to see the full help, or `python3 hapsolo_cli.py <subcommand> --help` for details on any step.

# Pipeline Steps

## 1. Preprocess

Clean FASTA headers (remove special characters, ensure uniqueness) and split contigs into individual files.

```
python3 hapsolo_cli.py preprocess -i assembly.fasta [-m MAXCONTIG_MB]
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
python3 hapsolo_cli.py align -i assembly_new.fasta -t 8 [--aligner minimap2|blat] [--no-gzip]
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
python3 hapsolo_cli.py train -i assembly_new.fasta --paf my_align.paf.gz -b ortholog_output/
```

For reference, the default parameters used by `hapsolo align` are:

```
minimap2 -t <threads> -P -G 500k -k19 -w2 -A1 -B2 -O2,4 -E2,1 \
    -s200 -z200 -N50 --max-qlen 10000000 --min-occ-floor=100 --paf-no-hit \
    assembly_new.fasta assembly_new.fasta | gzip > assembly_new_self_align.paf.gz
```

Avoid the `-x asm5` preset for self-alignment — it is tuned for high-identity assembly-to-reference alignment and discards most haplotig candidates.

**Reducing compute time with `-N`:** The `-N50` flag asks minimap2 to report up to 50 secondary alignments per query, which is the most expensive part of the run on large assemblies. Lowering this value (for example `-N20` or `-N10`) can substantially reduce alignment time and PAF file size and may still produce a workable result for HapSolo. However, this is a tunable knob — the optimal `-N` for your assembly depends on its repeat content, ploidy, and haplotig structure. We recommend testing different values on a representative dataset and comparing the resulting primary assembly statistics (size, BUSCO/ortholog completeness, contig count) before adopting a lower value for production runs.

## 3. Search (Ortholog Classification)

Align OrthoDB protein profiles against each contig with miniprot, then classify each ortholog as Complete, Fragmented, or Missing per contig. This produces the BUSCO-like gene completeness metrics consumed by the optimizer.

```
python3 hapsolo_cli.py search -i assembly_new.fasta -l diptera_odb10/ -o ortholog_output/ -t 8
```

| Flag | Description |
|---|---|
| `-i` | Preprocessed assembly FASTA |
| `-l` | Path to the OrthoDB lineage dataset directory |
| `-o` | Output directory for per-contig classification files (default: `busco_output`) |
| `-t` | Number of threads for miniprot |

**Output:** Per-contig TSV files in the output directory, in the format expected by the optimizer.

## 4. Train

Run hill-climbing optimization to find the alignment filter thresholds that minimize the cost function. Each thread starts from a random initial point and explores independently.

```
python3 hapsolo_cli.py train -i assembly_new.fasta --paf assembly_new_self_align.paf.gz -b ortholog_output/ -t 32 -n 2000
```

| Flag | Description |
|---|---|
| `-i` | Preprocessed assembly FASTA |
| `--paf` / `--psl` | Self-alignment file (gzipped or uncompressed) |
| `-b` | Ortholog classification directory (output of `search`) |
| `-t` | Number of parallel threads (default: 1). Total iterations = `-t × -n` |
| `-n` | Iterations per thread (default: 1000) |
| `-B` | Number of best candidate assemblies to return (default: 1) |
| `--min-contig` | Minimum contig size for primary assembly (default: 1000 bp) |
| `--mode` | Optimizer mode: `0` = random walk (default), `2` = steepest descent (see below) |
| `-S` / `-D` / `-F` / `-M` | Cost function weights for Single, Duplicate, Fragmented, Missing orthologs |

**Cost function:** `(F·θF + D·θD + M·θM) / (S·θS)`

Defaults: θS=1.0, θD=1.0, θF=0.0, θM=1.0

Lower scores are better. The optimizer minimizes duplicates and missing orthologs while maximizing single-copy orthologs.

### Optimizer modes

**Mode 0 — Random walk (default).** Each iteration perturbs the current thresholds by a random step in each dimension. Simple and effective; with enough iterations (1000–2000 per thread) it explores the search space broadly.

```
python3 hapsolo_cli.py train -i assembly_new.fasta --paf self_align.paf.gz -b ortholog_output/ -t 8 -n 2000
```

**Mode 2 — Steepest descent.** Each iteration evaluates all 8 octant neighbors (±step on each of the 3 threshold dimensions) and moves to the lowest-cost neighbor. More computationally intensive per iteration (8 evaluations per step vs 1), but converges faster when starting near a good solution. Uses the same plateau detection and boundary restart logic as mode 0.

```
python3 hapsolo_cli.py train -i assembly_new.fasta --paf self_align.paf.gz -b ortholog_output/ --mode 2 -t 8 -n 2000
```

Both modes produce the same output format and can be compared directly. In practice, mode 2 with fewer iterations can match or outperform mode 0 with more iterations, at the cost of higher per-iteration compute.

**Output:** Primary and secondary assembly FASTAs in the `asms/` directory, a `_report.txt` with assembly statistics and ortholog completeness for each result, plus `.scores` and `.deltascores` files containing the cost trajectory of every iteration.

### Progress display

During training, each thread shows a live progress bar with its current parameters and score:

```
JOBID: 0  [██████████████░░░░░░░░░░░░░░░░] 460/1000   PID: 0.7521 QPMin: 0.6843 QRPMin: 0.5912 CostΔ +0.0023 Score: 0.4156
JOBID: 1  [████████████░░░░░░░░░░░░░░░░░░] 392/1000   PID: 0.6234 QPMin: 0.7102 QRPMin: 0.4587 CostΔ +0.0000 Score: 0.4892
JOBID: 2  [█████████████░░░░░░░░░░░░░░░░░] 437/1000   PID: 0.8104 QPMin: 0.5621 QRPMin: 0.6342 CostΔ +0.0011 Score: 0.4321
...
```

## 5. Classify

Apply fixed thresholds and write primary/secondary assemblies without running the hill-climbing optimization. Useful for reproducing prior results, applying known-good thresholds from a previous training run, or running quick what-if analyses.

```
python3 hapsolo_cli.py classify -i assembly_new.fasta --paf assembly_new_self_align.paf.gz -b ortholog_output/
```

| Flag | Description |
|---|---|
| `-i` | Preprocessed assembly FASTA |
| `--paf` / `--psl` | Self-alignment file (gzipped or uncompressed) |
| `-b` | Ortholog classification directory (output of `search`) |
| `-P` / `--pid` | Fixed PID threshold (default: 0.7) |
| `-Q` / `--qpct` | Fixed query coverage threshold (default: 0.7) |
| `-R` / `--qrpct` | Fixed query/reference alignment length ratio threshold (default: 0.7) |
| `--min-contig` | Minimum contig size for primary assembly (default: 1000 bp) |

**Default behavior** (no thresholds specified) — uses 0.7/0.7/0.7, matching the original HapSolo mode 1 defaults:

```
python3 hapsolo_cli.py classify -i assembly_new.fasta --paf assembly_new_self_align.paf.gz -b ortholog_output/
```

**Custom thresholds** — use the exact values found by a previous `train` run, or any values you want to test:

```
python3 hapsolo_cli.py classify -i assembly_new.fasta --paf assembly_new_self_align.paf.gz -b ortholog_output/ \
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

## Example SLURM batch script

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
                                            -t $SLURM_CPUS_PER_TASK -n 2000 --mode 2
```

# Running Steps Individually

The CLI is a thin wrapper. Each step can be invoked directly:

```
# 1. Preprocess
python3 preprocessfasta.py -i assembly.fasta

# 2. Self-alignment (use the parameters above, not -x asm5)
minimap2 -t 36 -P -G 500k -k19 -w2 -A1 -B2 -O2,4 -E2,1 -s200 -z200 -N50 \
    --max-qlen 10000000 --min-occ-floor=100 --paf-no-hit \
    assembly_new.fasta assembly_new.fasta | gzip > self_align.paf.gz

# 3. Ortholog classification
python3 search_orthologs.py -i assembly_new.fasta -l diptera_odb10/ -o ortholog_output/ -t 8

# 4. Optimization
# mode 0 = random walk (default), mode 2 = steepest descent, mode 1 = fixed thresholds
python3 hapsolo.py -i assembly_new.fasta --paf self_align.paf.gz -b ortholog_output/ \
    --mode 0 -t 32 -n 2000

# Steepest descent optimizer
python3 hapsolo.py -i assembly_new.fasta --paf self_align.paf.gz -b ortholog_output/ \
    --mode 2 -t 8 -n 2000
```

# BLAT Alternative

HapSolo also accepts BLAT PSL alignment files. HPC batch scripts for BLAT are provided in the `scripts/` directory for SLURM (`sbatch_blat.sh`) and SGE (`qsub_blat.sh`). After BLAT array jobs complete, concatenate individual PSL files:

```
bash_andreaconcatpsl.sh myoutput_selfaln.PSL
python3 hapsolo_cli.py train -i assembly_new.fasta --psl myoutput_selfaln.PSL.gz -b ortholog_output/
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

The filenames encode the threshold values selected by the optimizer. The `_report.txt` contains assembly statistics at multiple size thresholds (contig counts, total lengths, N50/N75/L50/L75, GC content, N density) and ortholog completeness scores — comparable to QUAST + BUSCO output in a single file. The `.scores` and `.deltascores` files contain comma-separated cost values for every iteration of every thread, useful for plotting convergence curves.

# Limitations

- HapSolo does not accept absolute paths for input files. All file paths must be relative to the current working directory.

# Bug Reports

Please submit issues at https://github.com/esolares/HapSolo/issues
