import os


def _compute_detailed_asm_stats(contig_set, contig_sizes):
    """Compute assembly statistics at multiple size thresholds."""
    sizes = []
    for contig in contig_set:
        if contig in contig_sizes:
            sizes.append(contig_sizes[contig][0])
    if not sizes:
        return None
    sizes.sort(reverse=True)
    total = sum(sizes)
    thresholds = [0, 1000, 5000, 10000, 25000, 50000]
    counts = {}
    lengths = {}
    for t in thresholds:
        filtered = [s for s in sizes if s >= t]
        counts[t] = len(filtered)
        lengths[t] = sum(filtered)
    cumsum = 0
    n50 = n75 = 0
    l50 = l75 = 0
    for i, s in enumerate(sizes):
        cumsum += s
        if n50 == 0 and cumsum > total / 2.0:
            n50 = s
            l50 = i + 1
        if n75 == 0 and cumsum > total * 0.75:
            n75 = s
            l75 = i + 1
            break
    return {
        'counts': counts, 'lengths': lengths,
        'total_contigs': len(sizes), 'largest': sizes[0],
        'total_length': total,
        'n50': n50, 'l50': l50, 'n75': n75, 'l75': l75,
    }


def _compute_gc_and_ns(fasta_path):
    """Compute GC% and Ns per 100 kbp from a written FASTA file."""
    gc = 0
    at = 0
    n_count = 0
    total = 0
    with open(fasta_path) as f:
        for line in f:
            if line[0:1] == '>':
                continue
            seq = line.strip().upper()
            gc += seq.count('G') + seq.count('C')
            at += seq.count('A') + seq.count('T')
            n_count += seq.count('N')
            total += len(seq)
    gc_pct = 100.0 * gc / (gc + at) if (gc + at) > 0 else 0.0
    ns_per_100k = 100000.0 * n_count / total if total > 0 else 0.0
    return gc_pct, ns_per_100k


def write_report(primary_fasta_path, contig_set, ortho_scores, contig_sizes):
    """Write assembly statistics and ortholog completeness report."""
    report_path = primary_fasta_path.replace('_primary.fasta', '_report.txt')
    asm_name = os.path.basename(primary_fasta_path).replace('.fasta', '')

    stats = _compute_detailed_asm_stats(contig_set, contig_sizes)
    if stats is None:
        return
    gc_pct, ns_per_100k = _compute_gc_and_ns(primary_fasta_path)

    total_ortho = ortho_scores['C'] + ortho_scores['F'] + ortho_scores['M']
    c_pct = 100.0 * ortho_scores['C'] / total_ortho if total_ortho > 0 else 0.0
    s_pct = 100.0 * ortho_scores['S'] / total_ortho if total_ortho > 0 else 0.0
    d_pct = 100.0 * ortho_scores['D'] / total_ortho if total_ortho > 0 else 0.0
    f_pct = 100.0 * ortho_scores['F'] / total_ortho if total_ortho > 0 else 0.0
    m_pct = 100.0 * ortho_scores['M'] / total_ortho if total_ortho > 0 else 0.0

    w = 28
    vw = 12
    with open(report_path, 'w') as f:
        f.write('All statistics are based on contigs of size >= 500 bp, unless otherwise noted '
                '(e.g., "# contigs (>= 0 bp)" and "Total length (>= 0 bp)" include all contigs).\n\n')
        f.write('{:<{w}}{}\n'.format('Assembly', asm_name, w=w))
        for t in [0, 1000, 5000, 10000, 25000, 50000]:
            f.write('{:<{w}}{:<{vw}}\n'.format(
                '# contigs (>= ' + str(t) + ' bp)', stats['counts'][t], w=w, vw=vw))
        for t in [0, 1000, 5000, 10000, 25000, 50000]:
            f.write('{:<{w}}{:<{vw}}\n'.format(
                'Total length (>= ' + str(t) + ' bp)', stats['lengths'][t], w=w, vw=vw))
        f.write('{:<{w}}{:<{vw}}\n'.format('# contigs', stats['total_contigs'], w=w, vw=vw))
        f.write('{:<{w}}{:<{vw}}\n'.format('Largest contig', stats['largest'], w=w, vw=vw))
        f.write('{:<{w}}{:<{vw}}\n'.format('Total length', stats['total_length'], w=w, vw=vw))
        f.write('{:<{w}}{:<{vw}.2f}\n'.format('GC (%)', gc_pct, w=w, vw=vw))
        f.write('{:<{w}}{:<{vw}}\n'.format('N50', stats['n50'], w=w, vw=vw))
        f.write('{:<{w}}{:<{vw}}\n'.format('N75', stats['n75'], w=w, vw=vw))
        f.write('{:<{w}}{:<{vw}}\n'.format('L50', stats['l50'], w=w, vw=vw))
        f.write('{:<{w}}{:<{vw}}\n'.format('L75', stats['l75'], w=w, vw=vw))
        f.write('{:<{w}}{:<{vw}.2f}\n'.format("# N's per 100 kbp", ns_per_100k, w=w, vw=vw))
        f.write('#\n')
        f.write('# Ortholog completeness (n=' + str(total_ortho) + ')\n')
        f.write('#\n')
        f.write('\tC:' + '{:.1f}'.format(c_pct) + '%'
                '[S:' + '{:.1f}'.format(s_pct) + '%,'
                'D:' + '{:.1f}'.format(d_pct) + '%],'
                'F:' + '{:.1f}'.format(f_pct) + '%,'
                'M:' + '{:.1f}'.format(m_pct) + '%,'
                'n:' + str(total_ortho) + '\n')
        f.write('\n')
        f.write('\t' + str(ortho_scores['C']) + '\tComplete orthologs (C)\n')
        f.write('\t' + str(ortho_scores['S']) + '\tComplete and single-copy orthologs (S)\n')
        f.write('\t' + str(ortho_scores['D']) + '\tComplete and duplicated orthologs (D)\n')
        f.write('\t' + str(ortho_scores['F']) + '\tFragmented orthologs (F)\n')
        f.write('\t' + str(ortho_scores['M']) + '\tMissing orthologs (M)\n')
        f.write('\t' + str(total_ortho) + '\tTotal ortholog groups searched\n')

    return report_path
