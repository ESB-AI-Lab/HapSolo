#!/usr/bin/env python3
"""Compute assembly statistics for a FASTA file or compare multiple assemblies."""

import argparse
import os
import sys

import numpy as np

from hapsolo.assembly import CalculateContigSizes
from hapsolo.report import _compute_detailed_asm_stats, _compute_gc_and_ns

PLOT_COLORS = ['#1f77b4', '#e377c2', '#2ca02c', '#ff7f0e', '#9467bd',
               '#8c564b', '#17becf', '#bcbd22']


def parse_genome_size(value):
    """Parse genome size with optional suffix: 900m, 2g, 2.5g, 890000000."""
    value = value.strip().lower()
    multipliers = {'k': 1_000, 'm': 1_000_000, 'g': 1_000_000_000}
    if value[-1] in multipliers:
        return int(float(value[:-1]) * multipliers[value[-1]])
    return int(float(value))


def compute_assembly(fasta_path, genome_size=None, nchroms=None):
    """Load a FASTA and compute all stats. Returns a dict."""
    contig_sizes, _ = CalculateContigSizes(fasta_path)
    all_contigs = set(contig_sizes.keys())
    stats = _compute_detailed_asm_stats(all_contigs, contig_sizes)
    if stats is None:
        return None

    gc_pct, ns_per_100k = _compute_gc_and_ns(fasta_path)

    sizes_sorted = sorted(
        [contig_sizes[c][0] for c in all_contigs], reverse=True
    )

    ng50 = ng75 = 0
    lg50 = lg75 = 0
    if genome_size and genome_size > 0:
        cumsum = 0
        for i, s in enumerate(sizes_sorted):
            cumsum += s
            if ng50 == 0 and cumsum > genome_size / 2.0:
                ng50 = s
                lg50 = i + 1
            if ng75 == 0 and cumsum > genome_size * 0.75:
                ng75 = s
                lg75 = i + 1
                break

    nxsome = None
    if nchroms and nchroms > 0:
        n = min(nchroms, len(sizes_sorted))
        top_n = sizes_sorted[:n]
        top_n_total = sum(top_n)
        nxsome = {
            'n': n, 'expected': nchroms,
            'total': top_n_total, 'smallest': top_n[-1], 'largest': top_n[0],
            'pct_asm': 100.0 * top_n_total / stats['total_length'],
            'pct_genome': 100.0 * top_n_total / genome_size if genome_size else None,
        }

    return {
        'stats': stats, 'gc_pct': gc_pct, 'ns_per_100k': ns_per_100k,
        'sizes_sorted': sizes_sorted,
        'ng50': ng50, 'ng75': ng75, 'lg50': lg50, 'lg75': lg75,
        'nxsome': nxsome,
    }


def format_single_report(display_name, data, genome_size=None):
    """Format stats report for a single assembly."""
    stats = data['stats']
    w = 32
    vw = 16
    lines = []
    lines.append(f'{"Assembly":<{w}}{display_name}')
    lines.append('')
    for t in [0, 1000, 5000, 10000, 25000, 50000]:
        lines.append(f'{"# contigs (>= " + str(t) + " bp)":<{w}}{stats["counts"][t]:<{vw},}')
    lines.append('')
    for t in [0, 1000, 5000, 10000, 25000, 50000]:
        lines.append(f'{"Total length (>= " + str(t) + " bp)":<{w}}{stats["lengths"][t]:<{vw},}')
    lines.append('')
    lines.append(f'{"# contigs":<{w}}{stats["total_contigs"]:<{vw},}')
    lines.append(f'{"Largest contig":<{w}}{stats["largest"]:<{vw},}')
    lines.append(f'{"Total length":<{w}}{stats["total_length"]:<{vw},}')
    lines.append(f'{"GC (%)":<{w}}{data["gc_pct"]:<{vw}.2f}')
    lines.append(f'{"N50":<{w}}{stats["n50"]:<{vw},}')
    lines.append(f'{"N75":<{w}}{stats["n75"]:<{vw},}')
    lines.append(f'{"L50":<{w}}{stats["l50"]:<{vw},}')
    lines.append(f'{"L75":<{w}}{stats["l75"]:<{vw},}')
    if genome_size:
        lines.append(f'{"NG50":<{w}}{data["ng50"]:<{vw},}')
        lines.append(f'{"NG75":<{w}}{data["ng75"]:<{vw},}')
        lines.append(f'{"LG50":<{w}}{data["lg50"]:<{vw},}')
        lines.append(f'{"LG75":<{w}}{data["lg75"]:<{vw},}')
        lines.append(f'{"Expected genome size":<{w}}{genome_size:<{vw},}')
    lines.append(f'{"# N\'s per 100 kbp":<{w}}{data["ns_per_100k"]:<{vw}.2f}')
    nxsome = data['nxsome']
    if nxsome:
        lines.append('')
        lines.append(f'# Nxsome (top {nxsome["expected"]} contigs/scaffolds)')
        lines.append(f'{"Nxsome contigs used":<{w}}{nxsome["n"]:<{vw},}')
        lines.append(f'{"Nxsome total length":<{w}}{nxsome["total"]:<{vw},}')
        lines.append(f'{"Nxsome largest":<{w}}{nxsome["largest"]:<{vw},}')
        lines.append(f'{"Nxsome smallest":<{w}}{nxsome["smallest"]:<{vw},}')
        lines.append(f'{"Nxsome % of assembly":<{w}}{nxsome["pct_asm"]:<{vw}.2f}')
        if nxsome['pct_genome'] is not None:
            lines.append(f'{"Nxsome % of expected genome":<{w}}{nxsome["pct_genome"]:<{vw}.2f}')
    return '\n'.join(lines) + '\n'


def format_multi_report(names, data_list, genome_size=None):
    """Format a side-by-side comparison report for multiple assemblies."""
    w = 32
    vw = max(22, max(len(n) for n in names) + 4)
    n = len(names)

    def row(label, values):
        line = f'{label:<{w}}'
        for v in values:
            line += f'{v:<{vw}}'
        return line

    def row_int(label, key_fn):
        return row(label, [f'{key_fn(d):>,}' for d in data_list])

    def row_float(label, key_fn, fmt='.2f'):
        return row(label, [f'{key_fn(d):{fmt}}' for d in data_list])

    lines = []
    lines.append(row('Assembly', names))
    lines.append('')
    for t in [0, 1000, 5000, 10000, 25000, 50000]:
        lines.append(row_int(
            f'# contigs (>= {t} bp)',
            lambda d, t=t: d['stats']['counts'][t]))
    lines.append('')
    for t in [0, 1000, 5000, 10000, 25000, 50000]:
        lines.append(row_int(
            f'Total length (>= {t} bp)',
            lambda d, t=t: d['stats']['lengths'][t]))
    lines.append('')
    lines.append(row_int('# contigs', lambda d: d['stats']['total_contigs']))
    lines.append(row_int('Largest contig', lambda d: d['stats']['largest']))
    lines.append(row_int('Total length', lambda d: d['stats']['total_length']))
    lines.append(row_float('GC (%)', lambda d: d['gc_pct']))
    lines.append(row_int('N50', lambda d: d['stats']['n50']))
    lines.append(row_int('N75', lambda d: d['stats']['n75']))
    lines.append(row_int('L50', lambda d: d['stats']['l50']))
    lines.append(row_int('L75', lambda d: d['stats']['l75']))
    if genome_size:
        lines.append(row_int('NG50', lambda d: d['ng50']))
        lines.append(row_int('NG75', lambda d: d['ng75']))
        lines.append(row_int('LG50', lambda d: d['lg50']))
        lines.append(row_int('LG75', lambda d: d['lg75']))
        lines.append(row_int('Expected genome size', lambda d: genome_size))
    lines.append(row_float("# N's per 100 kbp", lambda d: d['ns_per_100k']))

    has_nxsome = any(d['nxsome'] for d in data_list)
    if has_nxsome:
        nchroms = next(d['nxsome']['expected'] for d in data_list if d['nxsome'])
        lines.append('')
        lines.append(f'# Nxsome (top {nchroms} contigs/scaffolds)')
        lines.append(row_int('Nxsome contigs used',
                             lambda d: d['nxsome']['n'] if d['nxsome'] else 0))
        lines.append(row_int('Nxsome total length',
                             lambda d: d['nxsome']['total'] if d['nxsome'] else 0))
        lines.append(row_int('Nxsome largest',
                             lambda d: d['nxsome']['largest'] if d['nxsome'] else 0))
        lines.append(row_int('Nxsome smallest',
                             lambda d: d['nxsome']['smallest'] if d['nxsome'] else 0))
        lines.append(row_float('Nxsome % of assembly',
                               lambda d: d['nxsome']['pct_asm'] if d['nxsome'] else 0.0))
        if genome_size:
            lines.append(row_float('Nxsome % of expected genome',
                                   lambda d: d['nxsome']['pct_genome'] if d['nxsome'] and d['nxsome']['pct_genome'] is not None else 0.0))

    return '\n'.join(lines) + '\n'


def plot_cumulative_length(assemblies, title, outpath,
                           genome_size=None, nchroms=None):
    """Plot cumulative length for one or more assemblies.

    assemblies: list of (display_name, sizes_sorted, nxsome_or_None)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (name, sizes, nxsome) in enumerate(assemblies):
        cumulative = np.cumsum(sizes)
        x = np.arange(1, len(cumulative) + 1)
        color = PLOT_COLORS[idx % len(PLOT_COLORS)]
        ax.plot(x, cumulative / 1e6, color=color, linewidth=2, label=name)

        if nchroms and nxsome:
            nxsome_mb = nxsome['total'] / 1e6
            ax.plot(nchroms, nxsome_mb, 'o', color=color, markersize=7,
                    zorder=5, markeredgecolor='black', markeredgewidth=0.8)

    if genome_size:
        genome_mb = genome_size / 1e6
        ax.axhline(y=genome_mb, color='#555555', linestyle='--', linewidth=1.2,
                    label=f'Expected genome ({genome_mb:,.0f} Mb)')

    if nchroms:
        ax.axvline(x=nchroms, color='black', linestyle='--', linewidth=1.5,
                    label=f'n = {nchroms} chromosomes')

    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel('# of Fragments (log)', fontsize=12)
    ax.set_ylabel('Assembly Size (Mb)', fontsize=12)
    ax.set_title(f'{title} Cumulative Length Graph', fontsize=13)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(True, which='major', linestyle='-', alpha=0.3)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Compute assembly statistics (N50, L50, GC%%, etc.) for one or more FASTA files.'
    )
    parser.add_argument('-i', '--input', nargs='+', required=True,
                        help='Input FASTA assembly file(s). Multiple files are compared side-by-side.')
    parser.add_argument('-o', '--outdir', default=None,
                        help='Output directory for stats file (default: same directory as first input)')
    parser.add_argument('--genome-size', type=parse_genome_size, default=None,
                        help='Expected genome size (e.g., 900m, 2g, 2.5g, 890000000). '
                             'Used to compute NG50/NG75.')
    parser.add_argument('-n', '--nchroms', type=int, default=None,
                        help='Expected number of chromosomes. Computes Nxsome: total size '
                             'and genome fraction in the top N contigs/scaffolds.')
    parser.add_argument('--name', nargs='+', default=None,
                        help='Display name(s) for plot title and legend. One per input file. '
                             'Defaults to input filenames.')
    parser.add_argument('--prefix', default=None,
                        help='Output filename prefix (required when comparing multiple assemblies).')
    parser.add_argument('--title', default=None,
                        help='Plot title (e.g., "Anopheles funestus Assembly"). '
                             'Defaults to --name (single) or --prefix (multi).')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating the cumulative length plot.')
    args = parser.parse_args()

    inputs = args.input
    multi = len(inputs) > 1

    if multi and not args.prefix:
        print('Error: --prefix is required when comparing multiple assemblies', file=sys.stderr)
        sys.exit(1)

    if args.name and len(args.name) != len(inputs):
        print(f'Error: --name count ({len(args.name)}) must match --input count ({len(inputs)})',
              file=sys.stderr)
        sys.exit(1)

    for f in inputs:
        if not os.path.isfile(f):
            print(f'Error: {f} not found', file=sys.stderr)
            sys.exit(1)

    outdir = args.outdir if args.outdir else os.path.dirname(os.path.abspath(inputs[0]))
    os.makedirs(outdir, exist_ok=True)

    names = []
    data_list = []
    for idx, fasta in enumerate(inputs):
        basename = os.path.splitext(os.path.basename(fasta))[0]
        name = args.name[idx] if args.name else basename
        names.append(name)

        print(f'Processing: {name} ({fasta})')
        data = compute_assembly(fasta, genome_size=args.genome_size, nchroms=args.nchroms)
        if data is None:
            print(f'Error: no contigs found in {fasta}', file=sys.stderr)
            sys.exit(1)
        data_list.append(data)

    prefix = args.prefix if args.prefix else os.path.splitext(os.path.basename(inputs[0]))[0]
    outfile = os.path.join(outdir, f'{prefix}_assembly_stats.txt')

    if multi:
        report = format_multi_report(names, data_list, genome_size=args.genome_size)
    else:
        report = format_single_report(names[0], data_list[0], genome_size=args.genome_size)

    with open(outfile, 'w') as f:
        f.write(report)

    print('')
    print(report)
    print(f'Stats written to {outfile}')

    if not args.no_plot:
        title = args.title if args.title else (names[0] if not multi else args.prefix)
        assemblies = [
            (names[i], data_list[i]['sizes_sorted'], data_list[i]['nxsome'])
            for i in range(len(inputs))
        ]
        for ext in ['png', 'pdf']:
            plot_path = os.path.join(outdir, f'{prefix}_cumulative_length.{ext}')
            plot_cumulative_length(
                assemblies, title, plot_path,
                genome_size=args.genome_size, nchroms=args.nchroms,
            )
            print(f'Plot written to {plot_path}')


if __name__ == '__main__':
    main()
