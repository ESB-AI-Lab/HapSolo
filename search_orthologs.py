#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
search_orthologs.py — Ortholog gene search and classification using miniprot.

Aligns OrthoDB protein profiles against a genome assembly with miniprot
(Heng Li), then classifies each ortholog group as Complete, Fragmented,
or Missing based on alignment coverage and score thresholds.

Supports OrthoDB lineage datasets from ODB9, ODB10, and ODB11.

Output: per-contig TSV files consumable by hapsolo.py's importBuscos().

Dependencies: miniprot (https://github.com/lh3/miniprot)
"""
import argparse
import glob
import os
import subprocess
import sys


def find_protein_file(lineage_dir):
    """Locate the protein sequence FASTA in an OrthoDB lineage dataset.

    Tries ODB10/11 format first, then ODB9 format.
    Returns (path, odb_version) where odb_version is 'odb9' or 'odb10'.
    """
    odb9_files = ['ancestral_variants', 'ancestral']
    odb10_files = ['refseq_db.faa']

    for name in odb10_files:
        path = os.path.join(lineage_dir, name)
        if os.path.exists(path):
            return path, 'odb10'

    for name in odb9_files:
        path = os.path.join(lineage_dir, name)
        if os.path.exists(path):
            return path, 'odb9'

    # Try glob for any .faa file — assume ODB10 format
    faa_files = glob.glob(os.path.join(lineage_dir, '*.faa'))
    if faa_files:
        return faa_files[0], 'odb10'

    return None, None


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

    ODB9:     BUSCO_ID  0          sd   mean_length
    ODB10/11: BUSCO_ID  n_species  mean_length  sd

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
            if len(fields) < 4:
                continue
            busco_id = fields[0]
            try:
                if odb_version == 'odb9':
                    # ODB9: ID, 0, sd, mean_length
                    cutoffs[busco_id] = (float(fields[3]), float(fields[2]))
                else:
                    # ODB10/11: ID, n_species, mean_length, sd
                    cutoffs[busco_id] = (float(fields[2]), float(fields[3]))
            except ValueError:
                continue
    return cutoffs


def build_protein_to_busco_map(protein_file, scores_cutoff):
    """Map protein sequence names to canonical BUSCO group IDs.

    ODB9 ancestral_variants has multiple variants per group
    (e.g. EOG09360002_0 .. _9).  We collapse to the base group ID
    by checking against scores_cutoff keys.

    ODB10/11 headers use colon separators (e.g. 100at7147:name).

    Returns (mapping dict {protein_name: busco_group_id},
             set of canonical BUSCO group IDs).
    """
    mapping = {}
    all_busco_ids = set()

    with open(protein_file) as f:
        for line in f:
            if not line.startswith('>'):
                continue
            header = line[1:].strip().split()[0]
            raw_id = header.split(':')[0]

            if raw_id in scores_cutoff:
                mapping[header] = raw_id
                all_busco_ids.add(raw_id)
            elif '_' in raw_id:
                base_id = raw_id.rsplit('_', 1)[0]
                if base_id in scores_cutoff:
                    mapping[header] = base_id
                    all_busco_ids.add(base_id)
                else:
                    mapping[header] = raw_id
                    all_busco_ids.add(raw_id)
            else:
                mapping[header] = raw_id
                all_busco_ids.add(raw_id)

    return mapping, all_busco_ids


def run_miniprot(genome_fasta, protein_fasta, output_paf, threads=1):
    """Run miniprot to align proteins against the genome.

    Returns the path to the PAF output file.
    """
    cmd = [
        'miniprot',
        '-t', str(threads),
        '--paf',
        '-I',   # no secondary alignments (report best only)
        genome_fasta,
        protein_fasta,
    ]

    print('Running: ' + ' '.join(cmd))
    with open(output_paf, 'w') as fout:
        proc = subprocess.run(cmd, stdout=fout, stderr=subprocess.PIPE)

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


def classify_buscos(hits, all_busco_ids, scores_cutoff, lengths_cutoff):
    """Classify BUSCO hits as Complete, Fragmented, or Missing per contig.

    Returns dict: {contig: {busco_id: (status, start, end, score, length)}}
    """
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
            # No hits at all -> Missing (will be added per-contig later)
            continue

        busco_hits = hits_by_busco[busco_id]

        # Get classification thresholds
        min_score = scores_cutoff.get(busco_id, 0)
        if busco_id in lengths_cutoff:
            mean_len, std_dev = lengths_cutoff[busco_id]
            complete_threshold = mean_len - 2 * std_dev
        else:
            # No length info: use 95% of query length from best hit
            best_qlen = max(h['query_len'] for h in busco_hits)
            complete_threshold = best_qlen * 0.95

        # Filter by score cutoff
        significant_hits = [h for h in busco_hits if h['score'] >= min_score]

        if not significant_hits:
            # No significant hits -> Missing
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
                # Upgrade Fragmented to Complete if better hit found
                elif status == 'Complete' and existing[0] == 'Fragmented':
                    contig_results[contig][busco_id] = (
                        status, hit['target_start'], hit['target_end'],
                        hit['score'], aligned_len)

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
        description='Lightweight BUSCO classifier using miniprot. '
                    'Replaces BUSCO for HapSolo pipeline. '
                    'Supports ODB9, ODB10, and ODB11 lineage datasets.')
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
    print('Total BUSCO groups in lineage: ' + str(len(all_busco_ids)))

    # Detect lineage name
    lineage_name = detect_lineage_name(args.lineage)
    print('Lineage: ' + lineage_name)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Run miniprot alignment
    paf_file = os.path.join(args.output, 'miniprot_odbaln.paf')
    run_miniprot(args.input, protein_file, paf_file, args.threads)

    # Parse results
    hits = parse_miniprot_paf(paf_file, protein_to_busco)
    print('Total alignment hits: ' + str(len(hits)))

    # Classify
    contig_results = classify_buscos(hits, all_busco_ids, scores_cutoff,
                                     lengths_cutoff)

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
