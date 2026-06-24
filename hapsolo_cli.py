#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hapsolo_cli.py — Unified command-line interface for the HapSolo pipeline.

Subcommands:
  preprocess  Clean FASTA headers and split contigs
  align       Self-alignment with minimap2 or BLAT
  search      Ortholog gene search with miniprot + OrthoDB
  train       Hill-climbing optimization to find best filter thresholds
  classify    Apply fixed thresholds and write primary/secondary assemblies

Typical workflow:
  python3 hapsolo_cli.py preprocess -i assembly.fasta
  python3 hapsolo_cli.py align -i assembly_new.fasta -t 8
  python3 hapsolo_cli.py search -i assembly_new.fasta -l diptera_odb10/ -t 8
  python3 hapsolo_cli.py train -i assembly_new.fasta --paf self_align.paf -b busco_output/ -t 4
  python3 hapsolo_cli.py classify -i assembly_new.fasta --paf self_align.paf -b busco_output/ --pid 0.7 --qpct 0.7 --qrpct 0.7
"""
import argparse
import os
import shutil
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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


def run_cmd(cmd, description=None):
    """Run a shell command, print it, and exit on failure."""
    if description:
        print('\n=== ' + description + ' ===')
    print('$ ' + ' '.join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print('Error: command failed with exit code ' + str(proc.returncode))
        sys.exit(proc.returncode)
    return proc


# ── preprocess ──────────────────────────────────────────────────────────────

def cmd_preprocess(args):
    """Clean FASTA headers, split contigs into individual files."""
    script = os.path.join(SCRIPT_DIR, 'preprocessfasta.py')
    cmd = [resolve_python(args), script, '-i', args.input]
    if args.maxcontig is not None:
        cmd.extend(['-m', str(args.maxcontig)])
    run_cmd(cmd, 'Preprocessing FASTA')

    # Determine output name
    base, ext = os.path.splitext(args.input)
    new_fasta = base + '_new' + ext
    print('\nOutput: ' + new_fasta)
    print('Contigs: contigs/')


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

    script = os.path.join(SCRIPT_DIR, 'search_orthologs.py')
    cmd = [
        resolve_python(args), script,
        '-i', args.input,
        '-l', args.lineage,
        '-o', args.output,
        '-t', str(args.threads),
    ]
    if args.contig_dir:
        cmd.extend(['--contig-dir', args.contig_dir])
    run_cmd(cmd, 'Ortholog search with miniprot')


# ── train ───────────────────────────────────────────────────────────────────

def cmd_train(args):
    """Run hill-climbing optimization to find best filter thresholds."""
    script = os.path.join(SCRIPT_DIR, 'hapsolo.py')
    cmd = [
        resolve_python(args), script,
        '-i', args.input,
        '-b', args.buscos,
        '--mode', str(args.mode),
        '-t', str(args.threads),
        '-n', str(args.iterations),
    ]

    # Alignment file (mutually exclusive)
    if args.paf:
        cmd.extend(['-a', args.paf])
    elif args.psl:
        cmd.extend(['-p', args.psl])
    else:
        print('Error: provide either --paf or --psl alignment file')
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

    run_cmd(cmd, 'Hill-climbing optimization')


# ── classify ────────────────────────────────────────────────────────────────

def cmd_classify(args):
    """Apply user-supplied (or default) thresholds and write primary/secondary
    assemblies without optimization (mode 1)."""
    script = os.path.join(SCRIPT_DIR, 'hapsolo.py')
    cmd = [
        resolve_python(args), script,
        '-i', args.input,
        '-b', args.buscos,
        '--mode', '1',
    ]

    # Alignment file
    if args.paf:
        cmd.extend(['-a', args.paf])
    elif args.psl:
        cmd.extend(['-p', args.psl])
    else:
        print('Error: provide either --paf or --psl alignment file')
        sys.exit(1)

    if args.min_contig is not None:
        cmd.extend(['--min', str(args.min_contig)])
    if args.pid is not None:
        cmd.extend(['-P', str(args.pid)])
    if args.qpct is not None:
        cmd.extend(['-Q', str(args.qpct)])
    if args.qrpct is not None:
        cmd.extend(['-R', str(args.qrpct)])

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
  hapsolo search     -i assembly_new.fasta -l diptera_odb10/ -t 8
  hapsolo train      -i assembly_new.fasta --paf self_align.paf -b busco_output/ -t 4
  hapsolo classify   -i assembly_new.fasta --paf self_align.paf -b busco_output/

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
    p_search.set_defaults(func=cmd_search)

    # ── train ──
    p_train = subparsers.add_parser('train',
        help='Hill-climbing optimization for filter thresholds')
    p_train.add_argument('-i', '--input', required=True,
        help='Input FASTA file (preprocessed)')
    p_train.add_argument('-b', '--buscos', required=True,
        help='Ortholog search output directory (odbaln_output/ or busco_output/)')
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
        help='Weight for single BUSCOs (default: 1.0)')
    p_train.add_argument('-D', '--thetaD', type=float, default=None,
        help='Weight for duplicate BUSCOs (default: 1.0)')
    p_train.add_argument('-F', '--thetaF', type=float, default=None,
        help='Weight for fragmented BUSCOs (default: 0.0)')
    p_train.add_argument('-M', '--thetaM', type=float, default=None,
        help='Weight for missing BUSCOs (default: 1.0)')
    p_train.add_argument('--mode', type=int, choices=[0, 2], default=0,
        help='Optimizer mode: 0 = random walk (default), 2 = steepest descent')
    p_train.set_defaults(func=cmd_train)

    # ── classify ──
    p_cls = subparsers.add_parser('classify',
        help='Apply user-supplied fixed thresholds and write assemblies (no optimization)')
    p_cls.add_argument('-i', '--input', required=True,
        help='Input FASTA file (preprocessed)')
    p_cls.add_argument('-b', '--buscos', required=True,
        help='Ortholog search output directory (odbaln_output/ or busco_output/)')
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
    p_cls.set_defaults(func=cmd_classify)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == '__main__':
    main()
