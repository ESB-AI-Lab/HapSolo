#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hapsolo_cli.py — Unified command-line interface for the HapSolo pipeline.

Subcommands:
  preprocess  Clean FASTA headers and split contigs
  align       Self-alignment with minimap2 or BLAT
  search      Ortholog gene search with miniprot + OrthoDB
  cache       Cache alignment and ortholog data for fast repeated training
  train       Hill-climbing optimization to find best filter thresholds
  classify    Apply fixed thresholds and write primary/secondary assemblies

Typical workflow:
  python3 hapsolo_cli.py preprocess -i assembly.fasta
  python3 hapsolo_cli.py align -i assembly_new.fasta -t 8
  python3 hapsolo_cli.py search -i assembly_new.fasta -l diptera_odb10/ -t 8
  python3 hapsolo_cli.py cache -i assembly_new.fasta --paf self_align.paf -b odb_output/
  python3 hapsolo_cli.py train -i assembly_new.fasta --hap self_align.hap -b odb_output/ -t 4
  python3 hapsolo_cli.py classify -i assembly_new.fasta --hap self_align.hap -b odb_output/ --pid 0.7 --qpct 0.7 --qrpct 0.7
"""
import argparse
import os
import shutil
import subprocess
import sys

# Directory containing this script (and the hapsolo/ package)
_CLI_DIR = os.path.dirname(os.path.abspath(__file__))





def resolve_python(args):
    """Resolve which Python interpreter to use for sub-scripts.

    If --python was passed, validate and return its absolute path.
    Otherwise return sys.executable (the interpreter running this CLI).
    """
    if getattr(args, 'python', None):
        # Resolve to absolute path so subprocess uses the exact interpreter
        path = shutil.which(args.python)
        if path is None:
            print('Error: --python "' + args.python + '" not found on PATH '
                  'or as a file. Try `which python3.10` or use an absolute path.')
            sys.exit(1)
        return path
    return sys.executable


def check_tool(name):
    """Check if an external tool is available on PATH."""
    return shutil.which(name) is not None


def _make_env():
    """Build subprocess env with PYTHONPATH including the HapSolo root."""
    env = os.environ.copy()
    env['PYTHONPATH'] = _CLI_DIR + os.pathsep + env.get('PYTHONPATH', '')
    return env


def run_cmd(cmd, description=None):
    """Run a shell command, print it, and exit on failure."""
    if description:
        print('\n=== ' + description + ' ===')
    print('$ ' + ' '.join(cmd))
    proc = subprocess.run(cmd, env=_make_env())
    if proc.returncode != 0:
        print('Error: command failed with exit code ' + str(proc.returncode))
        sys.exit(proc.returncode)
    return proc


# ── preprocess ──────────────────────────────────────────────────────────────

def cmd_preprocess(args):
    """Clean FASTA headers, split contigs into individual files."""
    cmd = [resolve_python(args), '-m', 'hapsolo.preprocess', '-i', args.input]
    if args.maxcontig is not None:
        cmd.extend(['-m', str(args.maxcontig)])
    if args.output is not None:
        cmd.extend(['-o', args.output])
    run_cmd(cmd, 'Preprocessing FASTA')

    basename = os.path.basename(args.input)
    base, ext = os.path.splitext(basename)
    if args.output:
        new_fasta = os.path.join(args.output, base + '_new' + ext)
        contigs_dir = os.path.join(args.output, 'contigs/')
    else:
        indir = os.path.dirname(args.input) or '.'
        new_fasta = os.path.join(indir, base + '_new' + ext)
        contigs_dir = os.path.join(indir, 'contigs/')
    print('\nOutput: ' + new_fasta)
    print('Contigs: ' + contigs_dir)


# ── align ───────────────────────────────────────────────────────────────────

def cmd_align(args):
    """Run self-alignment with minimap2 or BLAT.

    Note: HapSolo uses sensitive minimap2 parameters from the published paper.
    These defaults work well for most genomes, but advanced users are welcome
    to tune parameters further to match their assembly characteristics
    (e.g., adjust -k, -w for read accuracy, -N for max secondary alignments,
    or -s/-z for chaining sensitivity). See `minimap2 --help` for details.
    """
    if args.aligner == 'minimap2':
        if not check_tool('minimap2'):
            print('Error: minimap2 not found on PATH.')
            print('Install from https://github.com/lh3/minimap2 or download a precompiled release.')
            sys.exit(1)

        output = args.output if args.output else args.input.replace('.fasta', '_self_align.paf')
        cmd = [
            'minimap2',
            '-t', str(args.threads),
            '-P',
            '-G', '500k',
            '-k', '19',
            '-w', '2',
            '-A', '1',
            '-B', '2',
            '-O', '2,4',
            '-E', '2,1',
            '-s', '200',
            '-z', '200',
            '-N', '50',
            '--max-qlen', '10000000',
            '--min-occ-floor=100',
            '--paf-no-hit',
            args.input,
            args.input,
        ]
        run_cmd(cmd + ['-o', output], 'Self-alignment with minimap2 (HapSolo paper params)')
        print('\nOutput: ' + output)

    elif args.aligner == 'blat':
        if not check_tool('blat'):
            print('Error: blat not found on PATH.')
            print('Download from https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/blat/blat')
            sys.exit(1)

        output = args.output if args.output else args.input.replace('.fasta', '_self_align.psl')
        cmd = [
            'blat',
            args.input,
            args.input,
            output,
            '-noHead',
        ]
        run_cmd(cmd, 'Self-alignment with BLAT')
        print('\nOutput: ' + output)

    # Compress the alignment file with gzip (hapsolo.py reads .gz directly)
    if not args.no_gzip:
        if not check_tool('gzip'):
            print('Warning: gzip not found, leaving alignment uncompressed.')
        else:
            run_cmd(['gzip', '-f', output], 'Compressing alignment file')
            print('\nFinal output: ' + output + '.gz')
            print('Note: hapsolo.py / hapsolo_cli.py read .gz directly — no need to decompress.')
    print('\nTip: HapSolo uses sensitive minimap2 parameters from the published paper.')
    print('     Advanced users are welcome to tune parameters further by running')
    print('     minimap2 manually and passing the result to `hapsolo train --paf <file>`.')


# ── search ──────────────────────────────────────────────────────────────────

def cmd_search(args):
    """Ortholog gene search using miniprot against OrthoDB lineage database."""
    if not check_tool('miniprot'):
        print('Error: miniprot not found on PATH.')
        print('Install from https://github.com/lh3/miniprot (build with `make`)')
        sys.exit(1)

    cmd = [
        resolve_python(args), '-m', 'hapsolo.search',
        '-i', args.input,
        '-l', args.lineage,
        '-o', args.output,
        '-t', str(args.threads),
        '-B', str(args.end_bonus),
        '-j', str(args.splice_model),
    ]
    if args.contig_dir:
        cmd.extend(['--contig-dir', args.contig_dir])
    if args.classify_params:
        cmd.extend(['--classify-params', args.classify_params])
    run_cmd(cmd, 'Ortholog search with miniprot')


# ── cache ──────────────────────────────────────────────────────────────────

def cmd_cache(args):
    """Cache alignment and/or ortholog data for faster repeated training runs.

    Produces a .hap file from PAF/PSL alignment and verifies the consolidated
    ortholog TSV exists. These cached files skip the expensive parsing step
    in subsequent train runs.
    """
    python = resolve_python(args)
    did_align = False
    did_search = False
    missing = []

    # Alignment caching (PAF/PSL → HAP)
    if not args.search_only:
        if args.paf or args.psl:
            cmd = [
                python, '-m', 'hapsolo',
                '-i', args.input,
                '-b', args.orthologs if args.orthologs else 'dummy',
                '--generate-hap',
            ]
            if args.paf:
                cmd.extend(['-a', args.paf])
            else:
                cmd.extend(['-p', args.psl])
            if args.min_contig is not None:
                cmd.extend(['--min', str(args.min_contig)])

            # generate-hap needs -b but won't use it; provide orthologs if available
            # or use a minimal invocation. We run it directly instead.
            from hapsolo.alignment import create_paf_alignment, create_psl_alignment
            import time as _time

            print('\n=== Caching alignment data ===')
            _t0 = _time.time()
            min_contig = args.min_contig if args.min_contig is not None else 1000
            if args.paf:
                create_paf_alignment(args.paf, min_contig, 0.2, 0.2, 0.2)
            else:
                create_psl_alignment(args.psl, min_contig, 0.2, 0.2, 0.2)
            _t1 = _time.time()
            print('Alignment caching complete (%.1fs)' % (_t1 - _t0))

            # Report the HAP file location
            aln_file = args.paf or args.psl
            base = aln_file
            for ext in ['.gz', '.paf', '.psl']:
                if base.endswith(ext):
                    base = base[:-len(ext)]
            hap_path = base + '.hap'
            print('HAP file: ' + hap_path)
            did_align = True
        else:
            missing.append('align')

    # Ortholog caching (verify consolidated TSV exists)
    if not args.align_only:
        if args.orthologs:
            import os
            odb_dir = args.orthologs
            if os.path.isdir(odb_dir):
                tsv_path = os.path.join(odb_dir, 'full_table_results.tsv')
                if os.path.isfile(tsv_path):
                    # Verify it's readable
                    from hapsolo.scoring import import_orthologs_tsv
                    import time as _time
                    print('\n=== Caching ortholog data ===')
                    _t0 = _time.time()
                    b2c, c2b = import_orthologs_tsv(tsv_path)
                    _t1 = _time.time()
                    print('Loaded %d orthologs across %d contigs from consolidated TSV (%.1fs)' %
                          (len(b2c), len(c2b), _t1 - _t0))
                    print('Ortholog TSV: ' + tsv_path)
                    did_search = True
                else:
                    print('\nWarning: no full_table_results.tsv found in ' + odb_dir)
                    print('Run the search step first to generate ortholog classifications.')
                    missing.append('search')
            elif os.path.isfile(odb_dir):
                print('\n=== Ortholog TSV already cached: ' + odb_dir + ' ===')
                did_search = True
            else:
                print('\nWarning: ortholog path not found: ' + odb_dir)
                missing.append('search')
        else:
            missing.append('search')

    # Summary
    print('')
    if did_align and did_search:
        print('Cache complete. Both alignment and ortholog data are ready for training.')
    elif missing:
        for m in missing:
            if m == 'align':
                print('Warning: alignment data was not cached. Run `hapsolo align` first,')
                print('  then re-run `hapsolo cache` with --paf or --psl to generate the HAP file.')
            elif m == 'search':
                print('Warning: ortholog data was not cached. Run `hapsolo search` first,')
                print('  then re-run `hapsolo cache` with -b to verify the ortholog TSV.')


# ── train ───────────────────────────────────────────────────────────────────

def cmd_train(args):
    """Run hill-climbing optimization to find best filter thresholds."""
    cmd = [
        resolve_python(args), '-m', 'hapsolo',
        '-i', args.input,
        '-b', args.orthologs,
        '--mode', str(args.mode),
        '-t', str(args.threads),
        '-n', str(args.iterations),
    ]

    # Alignment file
    if args.hap:
        cmd.extend(['--hap', args.hap])
    elif args.paf:
        cmd.extend(['-a', args.paf])
    elif args.psl:
        cmd.extend(['-p', args.psl])
    else:
        print('Error: provide --hap, --paf, or --psl alignment file')
        sys.exit(1)

    if args.bestn is not None:
        cmd.extend(['-B', str(args.bestn)])
    if args.min_contig is not None:
        cmd.extend(['--min', str(args.min_contig)])
    if args.thetaS is not None:
        cmd.extend(['-S', str(args.thetaS)])
    if args.thetaD is not None:
        cmd.extend(['-D', str(args.thetaD)])
    if args.thetaF is not None:
        cmd.extend(['-F', str(args.thetaF)])
    if args.thetaM is not None:
        cmd.extend(['-M', str(args.thetaM)])
    if getattr(args, 'sa_temp', None) is not None:
        cmd.extend(['--sa-temp', str(args.sa_temp)])
    if getattr(args, 'sa_cooling', None) is not None:
        cmd.extend(['--sa-cooling', str(args.sa_cooling)])
    if getattr(args, 'sa_min_temp', None) is not None and args.sa_min_temp != 1e-6:
        cmd.extend(['--sa-min-temp', str(args.sa_min_temp)])
    if getattr(args, 'outdir', None) is not None and args.outdir != 'asms':
        cmd.extend(['--outdir', args.outdir])
    if getattr(args, 'formula', None) is not None:
        cmd.extend(['--formula', args.formula])
    if getattr(args, 'gpu', False):
        cmd.append('--gpu')
    if getattr(args, 'totaliters', None) is not None:
        cmd.extend(['--totaliters', str(args.totaliters)])
    if getattr(args, 'agents', None) is not None:
        cmd.extend(['--agents', str(args.agents)])

    run_cmd(cmd, 'Hill-climbing optimization')


# ── classify ────────────────────────────────────────────────────────────────

def cmd_classify(args):
    """Apply user-supplied (or default) thresholds and write primary/secondary
    assemblies without optimization (mode 1)."""
    cmd = [
        resolve_python(args), '-m', 'hapsolo',
        '-i', args.input,
        '-b', args.orthologs,
        '--mode', '1',
    ]

    # Alignment file
    if args.hap:
        cmd.extend(['--hap', args.hap])
    elif args.paf:
        cmd.extend(['-a', args.paf])
    elif args.psl:
        cmd.extend(['-p', args.psl])
    else:
        print('Error: provide --hap, --paf, or --psl alignment file')
        sys.exit(1)

    if args.min_contig is not None:
        cmd.extend(['--min', str(args.min_contig)])
    if args.pid is not None:
        cmd.extend(['-P', str(args.pid)])
    if args.qpct is not None:
        cmd.extend(['-Q', str(args.qpct)])
    if args.qrpct is not None:
        cmd.extend(['-R', str(args.qrpct)])
    if getattr(args, 'outdir', None) is not None and args.outdir != 'asms':
        cmd.extend(['--outdir', args.outdir])

    run_cmd(cmd, 'Classifying assembly (mode 1, fixed thresholds)')


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog='hapsolo',
        description='HapSolo — Haplotype reduction for diploid genome assemblies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Typical workflow:
  hapsolo preprocess -i assembly.fasta
  hapsolo align      -i assembly_new.fasta -t 8
  hapsolo search     -i assembly_new.fasta -l diptera_odb10/ -o odb_output/ -t 8
  hapsolo cache      -i assembly_new.fasta --paf self_align.paf -b odb_output/
  hapsolo train      -i assembly_new.fasta --hap self_align.hap -b odb_output/ -t 4
  hapsolo classify   -i assembly_new.fasta --hap self_align.hap -b odb_output/

Python interpreter:
  By default, sub-scripts run under the same interpreter that launches
  this CLI (sys.executable). On HPCs with multiple Python versions, you
  can either:

    a) Invoke the CLI with the desired interpreter:
         python3.10 hapsolo_cli.py train ...
       Sub-scripts inherit the same interpreter automatically.

    b) Use --python to override which interpreter runs sub-scripts:
         hapsolo_cli.py --python python3.10 train ...
         hapsolo_cli.py --python /opt/python3.11/bin/python3 train ...
""")
    parser.add_argument('--python', default=None,
        help='Python interpreter to use for sub-scripts (default: same as the '
             'interpreter running this CLI). Useful on HPCs with multiple '
             'Python versions installed (python3, python3.9, python3.10, etc.). '
             'Accepts a name like "python3.10" or an absolute path.')

    subparsers = parser.add_subparsers(dest='command', help='Pipeline step to run')

    # ── preprocess ──
    p_pre = subparsers.add_parser('preprocess',
        help='Clean FASTA headers and split contigs')
    p_pre.add_argument('-i', '--input', required=True,
        help='Input FASTA file')
    p_pre.add_argument('-o', '--output', type=str, default=None,
        help='Output directory for _new.fasta and contigs/ (default: same as input)')
    p_pre.add_argument('-m', '--maxcontig', type=int, default=None,
        help='Max contig size in Mb for individual files (default: 10)')
    p_pre.set_defaults(func=cmd_preprocess)

    # ── align ──
    p_aln = subparsers.add_parser('align',
        help='Self-alignment with minimap2 or BLAT')
    p_aln.add_argument('-i', '--input', required=True,
        help='Input FASTA file (preprocessed)')
    p_aln.add_argument('-t', '--threads', type=int, default=1,
        help='Number of threads (default: 1)')
    p_aln.add_argument('--aligner', choices=['minimap2', 'blat'], default='minimap2',
        help='Alignment tool (default: minimap2)')
    p_aln.add_argument('-o', '--output', default=None,
        help='Output alignment file (default: auto-named)')
    p_aln.add_argument('--no-gzip', action='store_true',
        help='Do not gzip the alignment output (default: compresses with gzip)')
    p_aln.set_defaults(func=cmd_align)

    # ── search ──
    p_search = subparsers.add_parser('search',
        help='Ortholog search and classification with miniprot + OrthoDB')
    p_search.add_argument('-i', '--input', required=True,
        help='Input genome FASTA file')
    p_search.add_argument('-l', '--lineage', required=True,
        help='OrthoDB lineage dataset directory (ODB9/10/11)')
    p_search.add_argument('-o', '--output', default='odbaln_output',
        help='Output directory for ortholog search results (default: odbaln_output)')
    p_search.add_argument('-t', '--threads', type=int, default=1,
        help='Number of threads (default: 1)')
    p_search.add_argument('--contig-dir', default='contigs',
        help='Contig directory name for output headers (default: contigs)')
    p_search.add_argument('-B', '--end-bonus', type=int, default=25,
        help='miniprot end-of-alignment bonus; higher values reduce '
             'protein-end clipping (default: 25)')
    p_search.add_argument('-j', '--splice-model', type=int, default=2,
        choices=[0, 1, 2],
        help='miniprot splice model: 0=none, 1=general, '
             '2=vertebrate/insect (default: 2)')
    p_search.add_argument('--classify-params', type=str, default=None,
        help='Classification filter parameters as comma-separated key=value '
             'pairs. Allowed: sr (min score/cutoff ratio, default 2.5), '
             'cov (min alignment coverage, default 0.7), gap (2nd contig '
             'score gap vs best, default 0.9). '
             'Example: --classify-params "sr=2.5,cov=0.7,gap=0.9"')
    p_search.set_defaults(func=cmd_search)

    # ── cache ──
    p_cache = subparsers.add_parser('cache',
        help='Cache alignment and ortholog data for faster repeated training (CPU only, one-time)')
    p_cache.add_argument('-i', '--input', required=True,
        help='Input FASTA file (preprocessed)')
    p_cache.add_argument('--paf', default=None,
        help='Minimap2 PAF alignment file to cache as HAP')
    p_cache.add_argument('--psl', default=None,
        help='BLAT PSL alignment file to cache as HAP')
    p_cache.add_argument('-b', '--orthologs', default=None,
        help='Ortholog search output directory (odb_output/) to verify cached TSV')
    p_cache.add_argument('--min-contig', type=int, default=None,
        help='Minimum contig size filter (default: 1000)')
    p_cache.add_argument('--align-only', action='store_true',
        help='Only cache alignment data (skip ortholog verification)')
    p_cache.add_argument('--search-only', action='store_true',
        help='Only verify ortholog data (skip alignment caching)')
    p_cache.set_defaults(func=cmd_cache)

    # ── train ──
    p_train = subparsers.add_parser('train',
        help='Hill-climbing optimization for filter thresholds')
    p_train.add_argument('-i', '--input', required=True,
        help='Input FASTA file (preprocessed)')
    p_train.add_argument('-b', '--orthologs', required=True,
        help='Ortholog data: search output directory (odb_output/) or cached TSV file')
    p_train.add_argument('--hap', default=None,
        help='Pre-cached HAP alignment file (from cache step)')
    p_train.add_argument('--paf', default=None,
        help='Minimap2 PAF alignment file')
    p_train.add_argument('--psl', default=None,
        help='BLAT PSL alignment file')
    p_train.add_argument('-t', '--threads', type=int, default=1,
        help='Number of parallel hill-climbing threads (default: 1)')
    p_train.add_argument('-n', '--iterations', type=int, default=1000,
        help='Number of iterations per thread (default: 1000)')
    p_train.add_argument('-B', '--bestn', type=int, default=None,
        help='Number of best assemblies to return (default: 1)')
    p_train.add_argument('--min-contig', type=int, default=None,
        help='Minimum contig size for primary assembly (default: 1000)')
    p_train.add_argument('-S', '--thetaS', type=float, default=None,
        help='Weight for single orthologs (default: 1.0)')
    p_train.add_argument('-D', '--thetaD', type=float, default=None,
        help='Weight for duplicate orthologs (default: 1.0)')
    p_train.add_argument('-F', '--thetaF', type=float, default=None,
        help='Weight for fragmented orthologs (default: 0.0)')
    p_train.add_argument('-M', '--thetaM', type=float, default=None,
        help='Weight for missing orthologs (default: 1.0)')
    p_train.add_argument('--mode', type=int, choices=[0, 2, 3], default=0,
        help='Optimizer mode: 0 = random walk (default), 2 = steepest descent, 3 = simulated annealing')
    p_train.add_argument('--sa-temp', type=float, default=None,
        help='SA initial temperature (default: auto-calibrate)')
    p_train.add_argument('--sa-cooling', type=float, default=None,
        help='SA cooling rate alpha (default: auto from iterations)')
    p_train.add_argument('--sa-min-temp', type=float, default=1e-6,
        help='SA minimum temperature floor (default: 1e-6)')
    p_train.add_argument('--outdir', type=str, default='asms',
        help='Output directory for assembly files (default: asms)')
    p_train.add_argument('--formula', type=str, default=None,
        help='Custom cost formula (see python -m hapsolo --help for syntax)')
    p_train.add_argument('--gpu', action='store_true',
        help='Use GPU acceleration via cupy (-t is then ignored)')
    p_train.add_argument('--totaliters', type=int, default=None,
        help='Total iterations across all agents/threads; auto-divided by the '
             'agent/thread count and takes precedence over -n (default: -t x -n)')
    p_train.add_argument('--agents', type=int, default=None,
        help='Number of GPU walkers (default: auto-detected from the GPU; '
             'see python -m hapsolo --help)')
    p_train.set_defaults(func=cmd_train)

    # ── classify ──
    p_cls = subparsers.add_parser('classify',
        help='Apply user-supplied fixed thresholds and write assemblies (no optimization)')
    p_cls.add_argument('-i', '--input', required=True,
        help='Input FASTA file (preprocessed)')
    p_cls.add_argument('-b', '--orthologs', required=True,
        help='Ortholog data: search output directory (odb_output/) or cached TSV file')
    p_cls.add_argument('--hap', default=None,
        help='Pre-cached HAP alignment file (from cache step)')
    p_cls.add_argument('--paf', default=None,
        help='Minimap2 PAF alignment file')
    p_cls.add_argument('--psl', default=None,
        help='BLAT PSL alignment file')
    p_cls.add_argument('--min-contig', type=int, default=None,
        help='Minimum contig size for primary assembly (default: 1000)')
    p_cls.add_argument('-P', '--pid', type=float, default=None,
        help='Fixed PID threshold (default: 0.7)')
    p_cls.add_argument('-Q', '--qpct', type=float, default=None,
        help='Fixed query coverage threshold (default: 0.7)')
    p_cls.add_argument('-R', '--qrpct', type=float, default=None,
        help='Fixed query/reference alignment length ratio threshold (default: 0.7)')
    p_cls.add_argument('--outdir', type=str, default='asms',
        help='Output directory for assembly files (default: asms)')
    p_cls.set_defaults(func=cmd_classify)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == '__main__':
    main()
