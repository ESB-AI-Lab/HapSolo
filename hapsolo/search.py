#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
search_orthologs.py — Ortholog gene search and classification using miniprot.

Aligns OrthoDB protein profiles against a genome assembly with miniprot
(Heng Li), then classifies each ortholog group as Complete, Fragmented,
or Missing based on alignment coverage and score thresholds.

Classification filtering (methods):
    The classifier applies post-alignment filters to approximate BUSCO v6.1
    ortholog completeness scores without requiring the full BUSCO pipeline
    (HMM profiling, Augustus gene prediction, etc.). Default filter
    parameters were tuned to minimize divergence from BUSCO v6.1 results:

        sr  = 2.5  — minimum score/cutoff ratio; rejects weak hits that
                      pass the OrthoDB score_cutoff but fall well below
                      the expected score for a true ortholog
        cov = 0.7  — minimum alignment coverage; rejects partial alignments
                      that cover less than 70% of the query protein
        gap = 0.9  — secondary contig score gap; when an ortholog is found
                      on multiple contigs, the 2nd+ contig must score within
                      90% of the best contig's score to retain its Complete
                      classification (reduces false duplicates)

    Validation on Culex tarsalis (diptera_odb12.2, n=3914) shows the
    filtered classifier tracks BUSCO v6.1 within ~1% across all categories
    (C, S, D, F, M). Without filtering (sr=1.0, cov=0.0, gap=0.0), the
    classifier inflates duplicates by ~11% and underreports singles by ~6%,
    which misleads the HapSolo optimizer into over-purging contigs.
    Validation on additional species is ongoing.

    Users can override defaults via --classify-params "sr=X,cov=Y,gap=Z"
    or pass classify_params dict to classify_buscos() directly.

Supports OrthoDB lineage datasets from ODB9, ODB10, ODB12, and ODB12.2.

Output: per-contig TSV files consumable by hapsolo.py's importBuscos().

Dependencies: miniprot (https://github.com/lh3/miniprot)
"""
import argparse
import glob
import gzip
import os
import subprocess
import sys
import tempfile


def find_protein_file(lineage_dir):
    """Locate the protein sequence FASTA in an OrthoDB lineage dataset.

    Tries ODB10+ format first (refseq_db.faa or refseq_db.faa.gz),
    then ODB9 format (ancestral_variants, ancestral).
    Returns (path, odb_version) where odb_version is 'odb9' or 'odb10'.
    """
    odb10_files = ['refseq_db.faa', 'refseq_db.faa.gz']
    odb9_files = ['ancestral_variants', 'ancestral']

    for name in odb10_files:
        path = os.path.join(lineage_dir, name)
        if os.path.exists(path):
            return path, 'odb10'

    for name in odb9_files:
        path = os.path.join(lineage_dir, name)
        if os.path.exists(path):
            return path, 'odb9'

    faa_files = glob.glob(os.path.join(lineage_dir, '*.faa'))
    if faa_files:
        return faa_files[0], 'odb10'
    faa_gz_files = glob.glob(os.path.join(lineage_dir, '*.faa.gz'))
    if faa_gz_files:
        return faa_gz_files[0], 'odb10'

    return None, None


def _open_protein_file(path):
    """Open a protein FASTA file, handling gzip transparently."""
    if path.endswith('.gz'):
        return gzip.open(path, 'rt')
    return open(path)


def load_scores_cutoff(lineage_dir):
    """Load the per-BUSCO score thresholds from the lineage dataset.

    Returns dict: {busco_id: min_score}
    """
    path = os.path.join(lineage_dir, 'scores_cutoff')
    if not os.path.exists(path):
        return {}

    cutoffs = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            fields = line.split('\t')
            if len(fields) >= 2:
                busco_id = fields[0]
                try:
                    score = float(fields[1])
                except ValueError:
                    continue
                cutoffs[busco_id] = score
    return cutoffs


def load_lengths_cutoff(lineage_dir, odb_version):
    """Load the per-BUSCO expected protein lengths from the lineage dataset.

    ODB9:      BUSCO_ID  0          sd           mean_length  (4 cols)
    ODB10/11:  BUSCO_ID  n_species  mean_length  sd           (4 cols)
    ODB12.2:   BUSCO_ID  mean_length  sd                      (3 cols)

    Returns dict: {busco_id: (mean_length, std_dev)}
    """
    path = os.path.join(lineage_dir, 'lengths_cutoff')
    if not os.path.exists(path):
        return {}

    cutoffs = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            fields = line.split('\t')
            busco_id = fields[0]
            try:
                if len(fields) == 3:
                    # ODB12.2: ID, mean_length, sd
                    cutoffs[busco_id] = (float(fields[1]), float(fields[2]))
                elif odb_version == 'odb9':
                    # ODB9: ID, 0, sd, mean_length
                    cutoffs[busco_id] = (float(fields[3]), float(fields[2]))
                elif len(fields) >= 4:
                    # ODB10/11: ID, n_species, mean_length, sd
                    cutoffs[busco_id] = (float(fields[2]), float(fields[3]))
            except (ValueError, IndexError):
                continue
    return cutoffs


def build_protein_to_busco_map(protein_file, scores_cutoff):
    """Map protein sequence names to canonical ortholog group IDs.

    Header formats by version:
      ODB9:     >EOG09360002       or >EOG09360002_0 (variant)
      ODB10/11: >100at7147:species or >100at7147_101020_0:002d1a
      ODB12:    >336167at7147_343691_0:002d1a

    For ODB10+, the ortholog group ID is the NNNatTAXID prefix before
    the species/variant suffix (e.g. 100at7147 from 100at7147_101020_0).

    Returns (mapping dict {protein_name: ortholog_group_id},
             set of canonical ortholog group IDs).
    """
    mapping = {}
    all_busco_ids = set()

    with _open_protein_file(protein_file) as f:
        for line in f:
            if not line.startswith('>'):
                continue
            header = line[1:].strip().split()[0]
            raw_id = header.split(':')[0]

            if raw_id in scores_cutoff:
                mapping[header] = raw_id
                all_busco_ids.add(raw_id)
                continue

            # Try progressively shorter underscore-delimited prefixes
            matched = False
            if '_' in raw_id:
                parts = raw_id.split('_')
                for i in range(1, len(parts)):
                    candidate = '_'.join(parts[:i])
                    if candidate in scores_cutoff:
                        mapping[header] = candidate
                        all_busco_ids.add(candidate)
                        matched = True
                        break

            if not matched:
                mapping[header] = raw_id
                all_busco_ids.add(raw_id)

    return mapping, all_busco_ids


def run_miniprot(genome_fasta, protein_fasta, output_paf, threads=1,
                 end_bonus=25, splice_model=2):
    """Run miniprot to align proteins against the genome.

    Handles gzipped protein FASTA files by decompressing to a temp file.
    Returns the path to the PAF output file.
    """
    fasta_to_use = protein_fasta
    tmp_decompressed = None

    if protein_fasta.endswith('.gz'):
        tmp_decompressed = tempfile.NamedTemporaryFile(
            suffix='.faa', delete=False, dir=os.path.dirname(output_paf))
        print('Decompressing ' + protein_fasta + ' ...')
        with gzip.open(protein_fasta, 'rb') as fin:
            while True:
                chunk = fin.read(1024 * 1024)
                if not chunk:
                    break
                tmp_decompressed.write(chunk)
        tmp_decompressed.close()
        fasta_to_use = tmp_decompressed.name

    cmd = [
        'miniprot',
        '-t', str(threads),
        '--paf',
        '-I',   # no secondary alignments (report best only)
        '-B', str(end_bonus),
        '-j', str(splice_model),
        genome_fasta,
        fasta_to_use,
    ]

    print('Running: ' + ' '.join(cmd))
    try:
        with open(output_paf, 'w') as fout:
            proc = subprocess.run(cmd, stdout=fout, stderr=subprocess.PIPE)
    finally:
        if tmp_decompressed is not None:
            os.unlink(tmp_decompressed.name)

    if proc.returncode != 0:
        print('miniprot failed:')
        print(proc.stderr.decode('utf-8', errors='replace'))
        sys.exit(1)

    return output_paf


def parse_miniprot_paf(paf_file, protein_to_busco):
    """Parse miniprot PAF output into per-BUSCO, per-contig hits.

    Uses protein_to_busco mapping to resolve variant names
    (e.g. EOG09360002_0) back to canonical BUSCO group IDs.

    Returns list of dicts: [{busco_id, contig, query_len, query_start, query_end,
                             target_start, target_end, score, aligned_len}, ...]
    """
    hits = []
    with open(paf_file) as f:
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) < 12:
                continue

            query_name = fields[0]
            busco_id = protein_to_busco.get(query_name, query_name.split(':')[0])

            query_len = int(fields[1])
            query_start = int(fields[2])
            query_end = int(fields[3])
            contig = fields[5]
            target_start = int(fields[7])
            target_end = int(fields[8])

            # Get alignment score from AS tag
            score = 0
            for tag in fields[12:]:
                if tag.startswith('AS:i:'):
                    score = int(tag[5:])
                    break

            aligned_len = query_end - query_start  # in protein coordinates

            hits.append({
                'busco_id': busco_id,
                'contig': contig,
                'query_len': query_len,
                'query_start': query_start,
                'query_end': query_end,
                'target_start': target_start,
                'target_end': target_end,
                'score': score,
                'aligned_len': aligned_len,
            })

    return hits


CLASSIFY_PARAM_KEYS = {'sr', 'cov', 'gap'}
CLASSIFY_PARAM_DEFAULTS = {'sr': 2.5, 'cov': 0.7, 'gap': 0.90}


def parse_classify_params(param_string):
    """Parse classify parameter string like 'sr=2.5,cov=0.7,gap=0.9'.

    Allowed keys:
        sr  — minimum score/cutoff ratio (default 2.5)
        cov — minimum alignment coverage floor (default 0.7)
        gap — 2nd+ contig must score >= gap * best contig's score (default 0.9)

    Unknown keys are ignored with a warning.
    """
    params = dict(CLASSIFY_PARAM_DEFAULTS)
    if not param_string:
        return params

    unknown = []
    for part in param_string.split(','):
        part = part.strip()
        if not part:
            continue
        if '=' not in part:
            unknown.append(part)
            continue
        key, val = part.split('=', 1)
        key = key.strip()
        if key not in CLASSIFY_PARAM_KEYS:
            unknown.append(key)
            continue
        try:
            params[key] = float(val.strip())
        except ValueError:
            unknown.append(key + '=' + val.strip())

    if unknown:
        print('Warning: unknown classify parameters ignored: '
              + ', '.join(unknown))

    return params


def classify_buscos(hits, all_busco_ids, scores_cutoff, lengths_cutoff,
                    classify_params=None):
    """Classify BUSCO hits as Complete, Fragmented, or Missing per contig.

    classify_params dict controls post-score-cutoff filtering:
        sr  — reject hits where score/cutoff < sr
        cov — reject hits where aligned_len/query_len < cov
        gap — after classification, drop secondary contigs whose best
              score < gap * best contig's score for that gene

    Returns dict: {contig: {busco_id: (status, start, end, score, length)}}
    """
    if classify_params is None:
        classify_params = dict(CLASSIFY_PARAM_DEFAULTS)

    min_score_ratio = classify_params.get('sr', CLASSIFY_PARAM_DEFAULTS['sr'])
    min_coverage = classify_params.get('cov', CLASSIFY_PARAM_DEFAULTS['cov'])
    score_gap = classify_params.get('gap', CLASSIFY_PARAM_DEFAULTS['gap'])

    # Group hits by BUSCO ID
    hits_by_busco = {}
    for hit in hits:
        bid = hit['busco_id']
        if bid not in hits_by_busco:
            hits_by_busco[bid] = []
        hits_by_busco[bid].append(hit)

    # Classify each BUSCO
    # result[contig][busco_id] = (status, start, end, score, length)
    contig_results = {}

    for busco_id in all_busco_ids:
        if busco_id not in hits_by_busco:
            continue

        busco_hits = hits_by_busco[busco_id]

        # Get classification thresholds
        min_score = scores_cutoff.get(busco_id, 0)
        if busco_id in lengths_cutoff:
            mean_len, std_dev = lengths_cutoff[busco_id]
            complete_threshold = mean_len - 2 * std_dev
        else:
            best_qlen = max(h['query_len'] for h in busco_hits)
            complete_threshold = best_qlen * 0.85

        # Filter by score cutoff, score ratio, and coverage
        significant_hits = []
        for h in busco_hits:
            if h['score'] < min_score:
                continue
            if min_score > 0 and h['score'] / min_score < min_score_ratio:
                continue
            coverage = h['aligned_len'] / h['query_len'] if h['query_len'] > 0 else 0
            if coverage < min_coverage:
                continue
            significant_hits.append(h)

        if not significant_hits:
            continue

        for hit in significant_hits:
            contig = hit['contig']
            if contig not in contig_results:
                contig_results[contig] = {}

            aligned_len = hit['aligned_len']

            if aligned_len >= complete_threshold:
                status = 'Complete'
            else:
                status = 'Fragmented'

            # Keep the best hit per BUSCO per contig
            if busco_id not in contig_results[contig]:
                contig_results[contig][busco_id] = (
                    status, hit['target_start'], hit['target_end'],
                    hit['score'], aligned_len)
            else:
                existing = contig_results[contig][busco_id]
                if hit['score'] > existing[3]:
                    contig_results[contig][busco_id] = (
                        status, hit['target_start'], hit['target_end'],
                        hit['score'], aligned_len)
                elif status == 'Complete' and existing[0] == 'Fragmented':
                    contig_results[contig][busco_id] = (
                        status, hit['target_start'], hit['target_end'],
                        hit['score'], aligned_len)

    # Apply score gap filter: for each gene, find the best score across
    # all contigs. Drop secondary contigs whose score < gap * best.
    if score_gap > 0:
        best_score_per_gene = {}
        for contig, buscos in contig_results.items():
            for busco_id, (status, s, e, score, length) in buscos.items():
                if status == 'Complete':
                    if busco_id not in best_score_per_gene or score > best_score_per_gene[busco_id]:
                        best_score_per_gene[busco_id] = score

        for contig in list(contig_results.keys()):
            buscos = contig_results[contig]
            to_remove = []
            for busco_id, (status, s, e, score, length) in buscos.items():
                if status != 'Complete':
                    continue
                if busco_id not in best_score_per_gene:
                    continue
                threshold = score_gap * best_score_per_gene[busco_id]
                if score < threshold:
                    to_remove.append(busco_id)
            for busco_id in to_remove:
                del buscos[busco_id]
            if not buscos:
                del contig_results[contig]

    return contig_results


def write_odb_output(contig_results, all_busco_ids, output_dir, lineage_name,
                     contig_fasta_dir='contigs'):
    """Write per-contig ortholog classification TSV files.

    Creates the directory structure expected by hapsolo.py's importBuscos():
      output_dir/odbaln_CONTIG/run_CONTIG/full_table_CONTIG.tsv

    Also writes a concatenated summary file:
      output_dir/full_table_results.tsv
    """
    all_contigs = sorted(contig_results.keys())

    # Per-contig files
    for contig in all_contigs:
        odb_dir = os.path.join(output_dir, 'odbaln_' + contig, 'run_' + contig)
        os.makedirs(odb_dir, exist_ok=True)

        tsv_path = os.path.join(odb_dir, 'full_table_' + contig + '.tsv')
        with open(tsv_path, 'w') as f:
            f.write('# search_orthologs 1.0 (miniprot)\n')
            f.write('# The lineage dataset is: ' + lineage_name + '\n')
            f.write('# To reproduce this run: python search_orthologs.py -i '
                    + contig_fasta_dir + '/' + contig + '.fasta -l '
                    + lineage_name + '\n')
            f.write('#\n')
            f.write('# Busco id\tStatus\tContig\tStart\tEnd\tScore\tLength\n')

            contig_buscos = contig_results.get(contig, {})
            for busco_id in sorted(all_busco_ids):
                if busco_id in contig_buscos:
                    status, start, end, score, length = contig_buscos[busco_id]
                    f.write(busco_id + '\t' + status + '\t' + contig + '\t'
                            + str(start) + '\t' + str(end) + '\t'
                            + str(score) + '\t' + str(length) + '\n')
                else:
                    f.write(busco_id + '\tMissing\n')

    # Concatenated summary file — all non-Missing hits plus one Missing
    # row per BUSCO that has no hits on any contig
    summary_path = os.path.join(output_dir, 'full_table_results.tsv')
    seen_buscos = set()
    with open(summary_path, 'w') as f:
        f.write('# search_orthologs 1.0 (miniprot)\n')
        f.write('# The lineage dataset is: ' + lineage_name + '\n')
        f.write('# Busco id\tStatus\tContig\tStart\tEnd\tScore\tLength\n')
        for contig in all_contigs:
            for busco_id in sorted(all_busco_ids):
                if busco_id in contig_results[contig]:
                    status, start, end, score, length = contig_results[contig][busco_id]
                    f.write(busco_id + '\t' + status + '\t' + contig + '\t'
                            + str(start) + '\t' + str(end) + '\t'
                            + str(score) + '\t' + str(length) + '\n')
                    seen_buscos.add(busco_id)
        for busco_id in sorted(all_busco_ids - seen_buscos):
            f.write(busco_id + '\tMissing\n')

    print('Results written for ' + str(len(all_contigs)) + ' contigs to '
          + output_dir)
    print('Concatenated results: ' + summary_path)


def detect_lineage_name(lineage_dir):
    """Extract the lineage name from the directory path."""
    name = os.path.basename(os.path.normpath(lineage_dir))
    # Check for dataset.cfg
    cfg = os.path.join(lineage_dir, 'dataset.cfg')
    if os.path.exists(cfg):
        with open(cfg) as f:
            for line in f:
                if line.startswith('name'):
                    parts = line.strip().split('=')
                    if len(parts) >= 2:
                        return parts[1].strip()
    return name


def main():
    parser = argparse.ArgumentParser(
        description='Lightweight ortholog classifier using miniprot. '
                    'Replaces BUSCO for HapSolo pipeline. '
                    'Supports ODB9, ODB10, ODB12, and ODB12.2 '
                    'lineage datasets (auto-detects format).')
    parser.add_argument('-i', '--input', required=True,
                        help='Input genome FASTA file')
    parser.add_argument('-l', '--lineage', required=True,
                        help='Path to OrthoDB lineage dataset directory '
                             '(e.g., diptera_odb10/)')
    parser.add_argument('-o', '--output', required=True,
                        help='Output directory for BUSCO results')
    parser.add_argument('-t', '--threads', type=int, default=1,
                        help='Number of threads for miniprot (default: 1)')
    parser.add_argument('--contig-dir', default='contigs',
                        help='Contig FASTA directory name for output headers '
                             '(default: contigs)')
    parser.add_argument('-B', '--end-bonus', type=int, default=25,
                        help='miniprot end-of-alignment bonus score; higher '
                             'values reduce protein-end clipping (default: 25)')
    parser.add_argument('-j', '--splice-model', type=int, default=2,
                        choices=[0, 1, 2],
                        help='miniprot splice model: 0=none, 1=general, '
                             '2=vertebrate/insect (default: 2)')
    parser.add_argument('--classify-params', type=str, default=None,
                        help='Classification filter parameters as '
                             'comma-separated key=value pairs. '
                             'Allowed: sr (min score/cutoff ratio, default 2.5), '
                             'cov (min alignment coverage, default 0.7), '
                             'gap (2nd contig score gap vs best, default 0.9). '
                             'Example: --classify-params "sr=2.5,cov=0.7,gap=0.9"')
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input):
        print('Error: Input file not found: ' + args.input)
        sys.exit(1)
    if not os.path.isdir(args.lineage):
        print('Error: Lineage directory not found: ' + args.lineage)
        sys.exit(1)

    # Find protein sequences and detect ODB version
    protein_file, odb_version = find_protein_file(args.lineage)
    if protein_file is None:
        print('Error: No protein sequence file found in lineage directory.')
        print('Expected one of: refseq_db.faa, ancestral_variants, ancestral')
        sys.exit(1)
    print('Protein sequences: ' + protein_file + ' (' + odb_version + ')')

    # Load classification thresholds
    scores_cutoff = load_scores_cutoff(args.lineage)
    lengths_cutoff = load_lengths_cutoff(args.lineage, odb_version)
    print('Score cutoffs loaded: ' + str(len(scores_cutoff)) + ' BUSCOs')
    print('Length cutoffs loaded: ' + str(len(lengths_cutoff)) + ' BUSCOs')

    # Build protein-name → BUSCO-group-ID mapping (collapses ODB9 variants)
    protein_to_busco, all_busco_ids = build_protein_to_busco_map(
        protein_file, scores_cutoff)

    # Restrict to groups with proper classification thresholds.
    # Some datasets (e.g. ODB12) have proteins from many more groups
    # than scores_cutoff covers — those lack proper thresholds and
    # would produce unreliable classifications.
    extra = all_busco_ids - set(scores_cutoff.keys())
    if extra:
        print('Warning: ' + str(len(extra)) + ' ortholog groups in protein '
              'file have no score cutoff and will be excluded')
        all_busco_ids = all_busco_ids & set(scores_cutoff.keys())
    if not lengths_cutoff:
        print('Warning: No lengths_cutoff file found. This dataset appears to '
              'be ODB12 (OrthoDB v12.1), which lacks curated length thresholds. '
              'Complete vs Fragmented classification will use a less precise '
              'heuristic. We strongly recommend using ODB12.2 instead.')
    print('Total BUSCO groups in lineage: ' + str(len(all_busco_ids)))

    # Detect lineage name
    lineage_name = detect_lineage_name(args.lineage)
    print('Lineage: ' + lineage_name)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Run miniprot alignment
    paf_file = os.path.join(args.output, 'miniprot_odbaln.paf')
    run_miniprot(args.input, protein_file, paf_file, args.threads,
                 end_bonus=args.end_bonus, splice_model=args.splice_model)

    # Parse results
    hits = parse_miniprot_paf(paf_file, protein_to_busco)
    print('Total alignment hits: ' + str(len(hits)))

    # Parse classification filter parameters
    classify_params = parse_classify_params(args.classify_params)
    print('Classify params: sr=' + str(classify_params['sr'])
          + ', cov=' + str(classify_params['cov'])
          + ', gap=' + str(classify_params['gap']))

    # Classify
    contig_results = classify_buscos(hits, all_busco_ids, scores_cutoff,
                                     lengths_cutoff, classify_params)

    # Write per-contig TSVs and concatenated summary
    write_odb_output(contig_results, all_busco_ids, args.output,
                     lineage_name, args.contig_dir)

    # Print summary
    total_complete = 0
    total_fragmented = 0
    total_missing = 0
    all_contigs_buscos = {}
    for contig, buscos in contig_results.items():
        for bid, (status, s, e, sc, l) in buscos.items():
            if bid not in all_contigs_buscos or status == 'Complete':
                all_contigs_buscos[bid] = status
    for bid in all_busco_ids:
        st = all_contigs_buscos.get(bid, 'Missing')
        if st == 'Complete':
            total_complete += 1
        elif st == 'Fragmented':
            total_fragmented += 1
        else:
            total_missing += 1

    print('\nSummary:')
    print('  Complete:    ' + str(total_complete))
    print('  Fragmented:  ' + str(total_fragmented))
    print('  Missing:     ' + str(total_missing))
    print('  Total:       ' + str(len(all_busco_ids)))


if __name__ == '__main__':
    main()
