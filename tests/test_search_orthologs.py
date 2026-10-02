#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for search_orthologs.py — ortholog classifier functions.

Tests cover: load_scores_cutoff, load_lengths_cutoff, build_protein_to_busco_map,
classify_buscos, detect_lineage_name, find_protein_file, parse_miniprot_paf,
write_odb_output.

Does NOT test run_miniprot (requires miniprot binary).
"""
import gzip
import os
import shutil
import tempfile
import unittest

from hapsolo.search import (
    load_scores_cutoff,
    load_lengths_cutoff,
    build_protein_to_busco_map,
    classify_buscos,
    detect_lineage_name,
    find_protein_file,
    parse_miniprot_paf,
    write_odb_output,
    _open_protein_file,
)


class TestLoadScoresCutoff(unittest.TestCase):
    """Tests for load_scores_cutoff()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _write_scores(self, content):
        path = os.path.join(self.tmpdir, 'scores_cutoff')
        with open(path, 'w') as f:
            f.write(content)

    def test_odb10_two_column_format(self):
        """Standard ODB10 format: busco_id<TAB>score."""
        self._write_scores(
            "# Ortholog DB\n"
            "100at7147\t450.0\n"
            "200at7147\t300.0\n"
            "300at7147\t200.0\n"
        )
        result = load_scores_cutoff(self.tmpdir)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result['100at7147'], 450.0)
        self.assertAlmostEqual(result['200at7147'], 300.0)
        self.assertAlmostEqual(result['300at7147'], 200.0)

    def test_odb9_format(self):
        """ODB9 format: busco_id<TAB>score (same 2-column)."""
        self._write_scores(
            "EOG09360001\t500.0\n"
            "EOG09360002\t350.5\n"
        )
        result = load_scores_cutoff(self.tmpdir)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result['EOG09360001'], 500.0)
        self.assertAlmostEqual(result['EOG09360002'], 350.5)

    def test_skip_comment_and_blank_lines(self):
        """Comments and blank lines are ignored."""
        self._write_scores(
            "# This is a comment\n"
            "\n"
            "# Another comment\n"
            "BID001\t100.0\n"
            "\n"
            "BID002\t200.0\n"
        )
        result = load_scores_cutoff(self.tmpdir)
        self.assertEqual(len(result), 2)

    def test_empty_file(self):
        """Empty file returns empty dict."""
        self._write_scores("")
        result = load_scores_cutoff(self.tmpdir)
        self.assertEqual(result, {})

    def test_missing_file(self):
        """Missing scores_cutoff file returns empty dict."""
        result = load_scores_cutoff(self.tmpdir)
        self.assertEqual(result, {})

    def test_malformed_score_skipped(self):
        """Lines with non-numeric scores are skipped."""
        self._write_scores(
            "BID001\t100.0\n"
            "BID002\tNOTANUMBER\n"
            "BID003\t300.0\n"
        )
        result = load_scores_cutoff(self.tmpdir)
        self.assertEqual(len(result), 2)
        self.assertNotIn('BID002', result)

    def test_extra_columns_ignored(self):
        """Extra columns beyond the first two are silently ignored."""
        self._write_scores(
            "BID001\t100.0\textra_col\tmore\n"
            "BID002\t200.0\tanother\n"
        )
        result = load_scores_cutoff(self.tmpdir)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result['BID001'], 100.0)

    def test_single_column_line_skipped(self):
        """Lines with only one column are skipped (len(fields) < 2)."""
        self._write_scores(
            "BID001\t100.0\n"
            "SINGLE_COLUMN\n"
            "BID002\t200.0\n"
        )
        result = load_scores_cutoff(self.tmpdir)
        self.assertEqual(len(result), 2)


class TestLoadLengthsCutoff(unittest.TestCase):
    """Tests for load_lengths_cutoff()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _write_lengths(self, content):
        path = os.path.join(self.tmpdir, 'lengths_cutoff')
        with open(path, 'w') as f:
            f.write(content)

    def test_odb10_four_column_format(self):
        """ODB10/11 format: ID, n_species, mean_length, sd."""
        self._write_lengths(
            "100at7147\t42\t500.0\t50.0\n"
            "200at7147\t38\t300.0\t30.0\n"
        )
        result = load_lengths_cutoff(self.tmpdir, 'odb10')
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result['100at7147'][0], 500.0)  # mean
        self.assertAlmostEqual(result['100at7147'][1], 50.0)   # sd
        self.assertAlmostEqual(result['200at7147'][0], 300.0)
        self.assertAlmostEqual(result['200at7147'][1], 30.0)

    def test_odb9_four_column_format(self):
        """ODB9 format: ID, 0, sd, mean_length."""
        self._write_lengths(
            "EOG09360001\t0\t50.0\t500.0\n"
            "EOG09360002\t0\t30.0\t300.0\n"
        )
        result = load_lengths_cutoff(self.tmpdir, 'odb9')
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result['EOG09360001'][0], 500.0)  # mean
        self.assertAlmostEqual(result['EOG09360001'][1], 50.0)   # sd

    def test_odb12_2_three_column_format(self):
        """ODB12.2 format: ID, mean_length, sd (3 columns)."""
        self._write_lengths(
            "336167at7147\t450.0\t40.0\n"
            "336168at7147\t250.0\t25.0\n"
        )
        result = load_lengths_cutoff(self.tmpdir, 'odb10')  # version doesn't matter for 3 cols
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result['336167at7147'][0], 450.0)
        self.assertAlmostEqual(result['336167at7147'][1], 40.0)

    def test_missing_file(self):
        """Missing lengths_cutoff returns empty dict (e.g., ODB12)."""
        result = load_lengths_cutoff(self.tmpdir, 'odb10')
        self.assertEqual(result, {})

    def test_empty_file(self):
        """Empty file returns empty dict."""
        self._write_lengths("")
        result = load_lengths_cutoff(self.tmpdir, 'odb10')
        self.assertEqual(result, {})

    def test_skip_comments_and_blanks(self):
        """Comments and blank lines are skipped."""
        self._write_lengths(
            "# Header comment\n"
            "\n"
            "BID001\t10\t500.0\t50.0\n"
        )
        result = load_lengths_cutoff(self.tmpdir, 'odb10')
        self.assertEqual(len(result), 1)

    def test_malformed_values_skipped(self):
        """Lines with non-numeric values are skipped via ValueError."""
        self._write_lengths(
            "BID001\t10\t500.0\t50.0\n"
            "BID002\t10\tBAD\t50.0\n"
            "BID003\t10\t300.0\t30.0\n"
        )
        result = load_lengths_cutoff(self.tmpdir, 'odb10')
        self.assertEqual(len(result), 2)
        self.assertNotIn('BID002', result)

    def test_two_column_line_skipped(self):
        """Lines with only 2 columns trigger IndexError and are skipped."""
        self._write_lengths(
            "BID001\t500.0\n"  # only 2 cols, but 3-col branch expects float(fields[1])
        )
        # Actually 2-col hits the len(fields)==3 branch? No: len==2 doesn't match 3.
        # Falls to odb_version check. If odb10, tries fields[3] -> IndexError -> skip.
        result = load_lengths_cutoff(self.tmpdir, 'odb10')
        self.assertEqual(len(result), 0)

    def test_odb9_vs_odb10_different_column_meaning(self):
        """Same 4-column line interpreted differently by ODB version."""
        self._write_lengths(
            "BID001\t0\t50.0\t500.0\n"
        )
        # ODB9: mean=fields[3]=500.0, sd=fields[2]=50.0
        r9 = load_lengths_cutoff(self.tmpdir, 'odb9')
        self.assertAlmostEqual(r9['BID001'][0], 500.0)
        self.assertAlmostEqual(r9['BID001'][1], 50.0)

        # ODB10: mean=fields[2]=50.0, sd=fields[3]=500.0
        r10 = load_lengths_cutoff(self.tmpdir, 'odb10')
        self.assertAlmostEqual(r10['BID001'][0], 50.0)
        self.assertAlmostEqual(r10['BID001'][1], 500.0)


class TestBuildProteinToBuscoMap(unittest.TestCase):
    """Tests for build_protein_to_busco_map()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _write_protein_fasta(self, content, filename='refseq_db.faa'):
        path = os.path.join(self.tmpdir, filename)
        with open(path, 'w') as f:
            f.write(content)
        return path

    def _write_protein_fasta_gz(self, content, filename='refseq_db.faa.gz'):
        path = os.path.join(self.tmpdir, filename)
        with gzip.open(path, 'wt') as f:
            f.write(content)
        return path

    def test_odb10_header_exact_match(self):
        """Protein ID before colon exactly matches scores_cutoff key."""
        fasta = self._write_protein_fasta(
            ">100at7147:ABCDEF\n"
            "MSEQVENCE\n"
            ">200at7147:GHIJKL\n"
            "MOTHERSEQ\n"
        )
        scores = {'100at7147': 450.0, '200at7147': 300.0}
        mapping, all_ids = build_protein_to_busco_map(fasta, scores)
        self.assertEqual(mapping['100at7147:ABCDEF'], '100at7147')
        self.assertEqual(mapping['200at7147:GHIJKL'], '200at7147')
        self.assertEqual(all_ids, {'100at7147', '200at7147'})

    def test_odb10_header_prefix_match(self):
        """Protein header has species/variant suffix; prefix matches scores_cutoff."""
        fasta = self._write_protein_fasta(
            ">100at7147_101020_0:002d1a\n"
            "MSEQVENCE\n"
        )
        scores = {'100at7147': 450.0}
        mapping, all_ids = build_protein_to_busco_map(fasta, scores)
        self.assertEqual(mapping['100at7147_101020_0:002d1a'], '100at7147')
        self.assertEqual(all_ids, {'100at7147'})

    def test_odb9_header_no_colon(self):
        """ODB9 headers have no colon; raw_id matches directly."""
        fasta = self._write_protein_fasta(
            ">EOG09360001\n"
            "MSEQVENCE\n"
            ">EOG09360002\n"
            "MOTHERSEQ\n"
        )
        scores = {'EOG09360001': 500.0, 'EOG09360002': 350.0}
        mapping, all_ids = build_protein_to_busco_map(fasta, scores)
        self.assertEqual(mapping['EOG09360001'], 'EOG09360001')
        self.assertEqual(mapping['EOG09360002'], 'EOG09360002')

    def test_odb9_variant_suffix(self):
        """ODB9 variant headers like EOG09360002_0 resolve to the base ID."""
        fasta = self._write_protein_fasta(
            ">EOG09360002_0\n"
            "MSEQVENCE\n"
            ">EOG09360002_1\n"
            "MOTHERSEQ\n"
        )
        scores = {'EOG09360002': 350.0}
        mapping, all_ids = build_protein_to_busco_map(fasta, scores)
        self.assertEqual(mapping['EOG09360002_0'], 'EOG09360002')
        self.assertEqual(mapping['EOG09360002_1'], 'EOG09360002')
        self.assertEqual(all_ids, {'EOG09360002'})

    def test_unmatched_protein_uses_raw_id(self):
        """Proteins not in scores_cutoff use their raw_id as the mapping."""
        fasta = self._write_protein_fasta(
            ">UNKNOWN_PROT:HASH\n"
            "MSEQVENCE\n"
        )
        scores = {'100at7147': 450.0}
        mapping, all_ids = build_protein_to_busco_map(fasta, scores)
        self.assertEqual(mapping['UNKNOWN_PROT:HASH'], 'UNKNOWN_PROT')
        self.assertIn('UNKNOWN_PROT', all_ids)

    def test_empty_fasta(self):
        """Empty protein FASTA returns empty mapping."""
        fasta = self._write_protein_fasta("")
        scores = {'100at7147': 450.0}
        mapping, all_ids = build_protein_to_busco_map(fasta, scores)
        self.assertEqual(mapping, {})
        self.assertEqual(all_ids, set())

    def test_gzipped_fasta(self):
        """Gzipped protein FASTA is read transparently."""
        fasta = self._write_protein_fasta_gz(
            ">100at7147:ABCDEF\n"
            "MSEQVENCE\n"
        )
        scores = {'100at7147': 450.0}
        mapping, all_ids = build_protein_to_busco_map(fasta, scores)
        self.assertEqual(mapping['100at7147:ABCDEF'], '100at7147')

    def test_multiple_proteins_same_busco(self):
        """Multiple protein variants map to the same BUSCO ID."""
        fasta = self._write_protein_fasta(
            ">100at7147_101020_0:hash1\n"
            "MSEQ1\n"
            ">100at7147_101020_1:hash2\n"
            "MSEQ2\n"
            ">100at7147_202030_0:hash3\n"
            "MSEQ3\n"
        )
        scores = {'100at7147': 450.0}
        mapping, all_ids = build_protein_to_busco_map(fasta, scores)
        self.assertEqual(len(mapping), 3)
        for v in mapping.values():
            self.assertEqual(v, '100at7147')
        self.assertEqual(all_ids, {'100at7147'})

    def test_sequence_lines_ignored(self):
        """Non-header lines (sequence data) are ignored."""
        fasta = self._write_protein_fasta(
            ">BID001:hash\n"
            "MSEQVENCE\n"
            "CONTINUED\n"
            ">BID002:hash\n"
            "ANOTHER\n"
        )
        scores = {'BID001': 100.0, 'BID002': 200.0}
        mapping, all_ids = build_protein_to_busco_map(fasta, scores)
        self.assertEqual(len(mapping), 2)

    def test_header_with_spaces(self):
        """Header with description after space — only first word used."""
        fasta = self._write_protein_fasta(
            ">100at7147:ABCDEF some description here\n"
            "MSEQVENCE\n"
        )
        scores = {'100at7147': 450.0}
        mapping, all_ids = build_protein_to_busco_map(fasta, scores)
        self.assertIn('100at7147:ABCDEF', mapping)


class TestClassifyBuscos(unittest.TestCase):
    """Tests for classify_buscos()."""

    def test_complete_classification(self):
        """Hit with aligned_len >= mean - 2*sd is Complete."""
        hits = [{
            'busco_id': 'BID001',
            'contig': 'contig1',
            'query_len': 500,
            'query_start': 0,
            'query_end': 480,
            'target_start': 1000,
            'target_end': 2440,
            'score': 500,
            'aligned_len': 480,
        }]
        all_ids = {'BID001'}
        scores = {'BID001': 100.0}
        lengths = {'BID001': (500.0, 50.0)}  # threshold = 500 - 2*50 = 400
        result = classify_buscos(hits, all_ids, scores, lengths)
        self.assertEqual(result['contig1']['BID001'][0], 'Complete')

    def test_fragmented_classification(self):
        """Hit with aligned_len < mean - 2*sd is Fragmented."""
        hits = [{
            'busco_id': 'BID001',
            'contig': 'contig1',
            'query_len': 500,
            'query_start': 0,
            'query_end': 200,
            'target_start': 1000,
            'target_end': 1600,
            'score': 500,
            'aligned_len': 200,
        }]
        all_ids = {'BID001'}
        scores = {'BID001': 100.0}
        lengths = {'BID001': (500.0, 50.0)}  # threshold = 400
        no_filter = {'sr': 1.0, 'cov': 0.0, 'gap': 0.0}
        result = classify_buscos(hits, all_ids, scores, lengths, no_filter)
        self.assertEqual(result['contig1']['BID001'][0], 'Fragmented')

    def test_missing_classification_no_hits(self):
        """BUSCO with no hits is Missing (not in contig_results)."""
        hits = []
        all_ids = {'BID001'}
        scores = {'BID001': 100.0}
        lengths = {'BID001': (500.0, 50.0)}
        result = classify_buscos(hits, all_ids, scores, lengths)
        # Missing BUSCOs are simply absent from contig_results
        self.assertEqual(result, {})

    def test_missing_due_to_low_score(self):
        """Hits below score cutoff are treated as Missing."""
        hits = [{
            'busco_id': 'BID001',
            'contig': 'contig1',
            'query_len': 500,
            'query_start': 0,
            'query_end': 480,
            'target_start': 1000,
            'target_end': 2440,
            'score': 50,    # Below cutoff of 100
            'aligned_len': 480,
        }]
        all_ids = {'BID001'}
        scores = {'BID001': 100.0}
        lengths = {'BID001': (500.0, 50.0)}
        result = classify_buscos(hits, all_ids, scores, lengths)
        self.assertEqual(result, {})

    def test_best_hit_per_contig_kept(self):
        """When multiple hits for same BUSCO on same contig, best score wins."""
        hits = [
            {
                'busco_id': 'BID001', 'contig': 'contig1',
                'query_len': 500, 'query_start': 0, 'query_end': 480,
                'target_start': 1000, 'target_end': 2440,
                'score': 200, 'aligned_len': 480,
            },
            {
                'busco_id': 'BID001', 'contig': 'contig1',
                'query_len': 500, 'query_start': 0, 'query_end': 490,
                'target_start': 3000, 'target_end': 4470,
                'score': 600, 'aligned_len': 490,
            },
        ]
        all_ids = {'BID001'}
        scores = {'BID001': 100.0}
        lengths = {'BID001': (500.0, 50.0)}
        result = classify_buscos(hits, all_ids, scores, lengths)
        self.assertEqual(result['contig1']['BID001'][3], 600)  # score of best hit

    def test_upgrade_fragmented_to_complete(self):
        """A Complete hit upgrades a prior Fragmented classification on the same contig."""
        hits = [
            {
                'busco_id': 'BID001', 'contig': 'contig1',
                'query_len': 500, 'query_start': 0, 'query_end': 200,
                'target_start': 1000, 'target_end': 1600,
                'score': 600, 'aligned_len': 200,  # Fragmented (200 < 400)
            },
            {
                'busco_id': 'BID001', 'contig': 'contig1',
                'query_len': 500, 'query_start': 0, 'query_end': 480,
                'target_start': 3000, 'target_end': 4440,
                'score': 500, 'aligned_len': 480,  # Complete (480 >= 400), lower score
            },
        ]
        all_ids = {'BID001'}
        scores = {'BID001': 100.0}
        lengths = {'BID001': (500.0, 50.0)}
        result = classify_buscos(hits, all_ids, scores, lengths)
        # The second hit has lower score but is Complete, should upgrade
        self.assertEqual(result['contig1']['BID001'][0], 'Complete')

    def test_multiple_contigs(self):
        """Same BUSCO found on different contigs."""
        hits = [
            {
                'busco_id': 'BID001', 'contig': 'contig1',
                'query_len': 500, 'query_start': 0, 'query_end': 480,
                'target_start': 1000, 'target_end': 2440,
                'score': 500, 'aligned_len': 480,
            },
            {
                'busco_id': 'BID001', 'contig': 'contig2',
                'query_len': 500, 'query_start': 0, 'query_end': 200,
                'target_start': 2000, 'target_end': 2600,
                'score': 300, 'aligned_len': 200,
            },
        ]
        all_ids = {'BID001'}
        scores = {'BID001': 100.0}
        lengths = {'BID001': (500.0, 50.0)}
        no_filter = {'sr': 1.0, 'cov': 0.0, 'gap': 0.0}
        result = classify_buscos(hits, all_ids, scores, lengths, no_filter)
        self.assertEqual(result['contig1']['BID001'][0], 'Complete')
        self.assertEqual(result['contig2']['BID001'][0], 'Fragmented')

    def test_no_lengths_cutoff_uses_query_heuristic(self):
        """Without lengths_cutoff, uses 95% of best query_len as threshold."""
        hits = [{
            'busco_id': 'BID001',
            'contig': 'contig1',
            'query_len': 100,
            'query_start': 0,
            'query_end': 96,
            'target_start': 1000,
            'target_end': 1288,
            'score': 200,
            'aligned_len': 96,
        }]
        all_ids = {'BID001'}
        scores = {'BID001': 100.0}
        lengths = {}  # No length data (like ODB12)
        no_filter = {'sr': 1.0, 'cov': 0.0, 'gap': 0.0}
        result = classify_buscos(hits, all_ids, scores, lengths, no_filter)
        # threshold = 100 * 0.95 = 95, aligned_len = 96 >= 95 -> Complete
        self.assertEqual(result['contig1']['BID001'][0], 'Complete')

    def test_no_lengths_cutoff_fragmented(self):
        """Without lengths_cutoff, short alignment is Fragmented."""
        hits = [{
            'busco_id': 'BID001',
            'contig': 'contig1',
            'query_len': 100,
            'query_start': 0,
            'query_end': 50,
            'target_start': 1000,
            'target_end': 1150,
            'score': 200,
            'aligned_len': 50,
        }]
        all_ids = {'BID001'}
        scores = {'BID001': 100.0}
        lengths = {}
        no_filter = {'sr': 1.0, 'cov': 0.0, 'gap': 0.0}
        result = classify_buscos(hits, all_ids, scores, lengths, no_filter)
        # threshold = 100 * 0.95 = 95, aligned_len = 50 < 95 -> Fragmented
        self.assertEqual(result['contig1']['BID001'][0], 'Fragmented')

    def test_busco_not_in_scores_cutoff(self):
        """BUSCO not in scores_cutoff uses min_score=0 (all hits pass)."""
        hits = [{
            'busco_id': 'NOVEL001',
            'contig': 'contig1',
            'query_len': 100,
            'query_start': 0,
            'query_end': 96,
            'target_start': 1000,
            'target_end': 1288,
            'score': 1,  # Very low but >= 0
            'aligned_len': 96,
        }]
        all_ids = {'NOVEL001'}
        scores = {}  # Empty scores cutoff
        lengths = {}
        result = classify_buscos(hits, all_ids, scores, lengths)
        self.assertIn('contig1', result)
        self.assertIn('NOVEL001', result['contig1'])

    def test_empty_all_busco_ids(self):
        """Empty all_busco_ids results in empty classification."""
        hits = [{
            'busco_id': 'BID001',
            'contig': 'contig1',
            'query_len': 500, 'query_start': 0, 'query_end': 480,
            'target_start': 1000, 'target_end': 2440,
            'score': 500, 'aligned_len': 480,
        }]
        all_ids = set()
        scores = {'BID001': 100.0}
        lengths = {'BID001': (500.0, 50.0)}
        result = classify_buscos(hits, all_ids, scores, lengths)
        self.assertEqual(result, {})


class TestDetectLineageName(unittest.TestCase):
    """Tests for detect_lineage_name()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_name_from_directory(self):
        """Returns directory basename when no dataset.cfg."""
        lineage_dir = os.path.join(self.tmpdir, 'diptera_odb10')
        os.makedirs(lineage_dir)
        result = detect_lineage_name(lineage_dir)
        self.assertEqual(result, 'diptera_odb10')

    def test_name_from_dataset_cfg(self):
        """Extracts name from dataset.cfg when present."""
        lineage_dir = os.path.join(self.tmpdir, 'diptera_odb10')
        os.makedirs(lineage_dir)
        cfg_path = os.path.join(lineage_dir, 'dataset.cfg')
        with open(cfg_path, 'w') as f:
            f.write("species=Drosophila melanogaster\n")
            f.write("name=diptera\n")
            f.write("creation_date=2020-01-01\n")
        result = detect_lineage_name(lineage_dir)
        self.assertEqual(result, 'diptera')

    def test_trailing_slash_normalized(self):
        """Trailing slash in path is normalized."""
        lineage_dir = os.path.join(self.tmpdir, 'embryophyta_odb9')
        os.makedirs(lineage_dir)
        result = detect_lineage_name(lineage_dir + '/')
        self.assertEqual(result, 'embryophyta_odb9')

    def test_cfg_without_name_key(self):
        """dataset.cfg without name= falls through to directory basename."""
        lineage_dir = os.path.join(self.tmpdir, 'lepidoptera_odb10')
        os.makedirs(lineage_dir)
        cfg_path = os.path.join(lineage_dir, 'dataset.cfg')
        with open(cfg_path, 'w') as f:
            f.write("species=Bombyx mori\n")
            f.write("creation_date=2020-01-01\n")
        result = detect_lineage_name(lineage_dir)
        self.assertEqual(result, 'lepidoptera_odb10')

    def test_cfg_with_spaces_around_equals(self):
        """Name extraction handles spaces around = sign."""
        lineage_dir = os.path.join(self.tmpdir, 'test_lineage')
        os.makedirs(lineage_dir)
        cfg_path = os.path.join(lineage_dir, 'dataset.cfg')
        with open(cfg_path, 'w') as f:
            f.write("name = hymenoptera\n")
        result = detect_lineage_name(lineage_dir)
        # The code splits on '=' so parts[1] = ' hymenoptera', .strip() removes space
        self.assertEqual(result, 'hymenoptera')


class TestFindProteinFile(unittest.TestCase):
    """Tests for find_protein_file()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_odb10_refseq_faa(self):
        """Finds refseq_db.faa (ODB10+ format)."""
        path = os.path.join(self.tmpdir, 'refseq_db.faa')
        with open(path, 'w') as f:
            f.write(">protein\nMSEQ\n")
        result_path, version = find_protein_file(self.tmpdir)
        self.assertEqual(result_path, path)
        self.assertEqual(version, 'odb10')

    def test_odb10_refseq_faa_gz(self):
        """Finds refseq_db.faa.gz (compressed ODB10+)."""
        path = os.path.join(self.tmpdir, 'refseq_db.faa.gz')
        with gzip.open(path, 'wt') as f:
            f.write(">protein\nMSEQ\n")
        result_path, version = find_protein_file(self.tmpdir)
        self.assertEqual(result_path, path)
        self.assertEqual(version, 'odb10')

    def test_odb9_ancestral_variants(self):
        """Finds ancestral_variants (ODB9 format)."""
        path = os.path.join(self.tmpdir, 'ancestral_variants')
        with open(path, 'w') as f:
            f.write(">EOG001\nMSEQ\n")
        result_path, version = find_protein_file(self.tmpdir)
        self.assertEqual(result_path, path)
        self.assertEqual(version, 'odb9')

    def test_odb9_ancestral(self):
        """Finds ancestral (ODB9 format, no variants)."""
        path = os.path.join(self.tmpdir, 'ancestral')
        with open(path, 'w') as f:
            f.write(">EOG001\nMSEQ\n")
        result_path, version = find_protein_file(self.tmpdir)
        self.assertEqual(result_path, path)
        self.assertEqual(version, 'odb9')

    def test_odb10_preferred_over_odb9(self):
        """refseq_db.faa is preferred when both ODB9 and ODB10 files exist."""
        for name in ['refseq_db.faa', 'ancestral_variants']:
            with open(os.path.join(self.tmpdir, name), 'w') as f:
                f.write(">prot\nMSEQ\n")
        result_path, version = find_protein_file(self.tmpdir)
        self.assertTrue(result_path.endswith('refseq_db.faa'))
        self.assertEqual(version, 'odb10')

    def test_fallback_to_any_faa(self):
        """Falls back to any .faa file if standard names not found."""
        path = os.path.join(self.tmpdir, 'custom_proteins.faa')
        with open(path, 'w') as f:
            f.write(">prot\nMSEQ\n")
        result_path, version = find_protein_file(self.tmpdir)
        self.assertEqual(result_path, path)
        self.assertEqual(version, 'odb10')

    def test_fallback_to_any_faa_gz(self):
        """Falls back to any .faa.gz file if nothing else found."""
        path = os.path.join(self.tmpdir, 'custom_proteins.faa.gz')
        with gzip.open(path, 'wt') as f:
            f.write(">prot\nMSEQ\n")
        result_path, version = find_protein_file(self.tmpdir)
        self.assertEqual(result_path, path)
        self.assertEqual(version, 'odb10')

    def test_empty_directory_returns_none(self):
        """Empty directory returns (None, None)."""
        result_path, version = find_protein_file(self.tmpdir)
        self.assertIsNone(result_path)
        self.assertIsNone(version)

    def test_unrelated_files_ignored(self):
        """Non-protein files are not picked up."""
        for name in ['scores_cutoff', 'lengths_cutoff', 'dataset.cfg', 'README.md']:
            with open(os.path.join(self.tmpdir, name), 'w') as f:
                f.write("not a protein file\n")
        result_path, version = find_protein_file(self.tmpdir)
        self.assertIsNone(result_path)
        self.assertIsNone(version)


class TestParseminiprotPaf(unittest.TestCase):
    """Tests for parse_miniprot_paf()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _write_paf(self, lines):
        path = os.path.join(self.tmpdir, 'test.paf')
        with open(path, 'w') as f:
            for line in lines:
                f.write(line + '\n')
        return path

    def _make_paf_line(self, query='100at7147:HASH', qlen=500, qstart=10,
                       qend=480, strand='+', target='contig1', tlen=50000,
                       tstart=1000, tend=2400, matches=470, block_len=480,
                       mapq=60, extra_tags=None):
        """Build a minimal 12+ column PAF line."""
        fields = [query, str(qlen), str(qstart), str(qend), strand,
                  target, str(tlen), str(tstart), str(tend),
                  str(matches), str(block_len), str(mapq)]
        if extra_tags:
            fields.extend(extra_tags)
        return '\t'.join(fields)

    def test_basic_parsing(self):
        """Parse a standard PAF line with AS tag."""
        paf = self._write_paf([
            self._make_paf_line(extra_tags=['AS:i:500', 'ms:i:480']),
        ])
        protein_to_busco = {'100at7147:HASH': '100at7147'}
        hits = parse_miniprot_paf(paf, protein_to_busco)
        self.assertEqual(len(hits), 1)
        h = hits[0]
        self.assertEqual(h['busco_id'], '100at7147')
        self.assertEqual(h['contig'], 'contig1')
        self.assertEqual(h['query_len'], 500)
        self.assertEqual(h['query_start'], 10)
        self.assertEqual(h['query_end'], 480)
        self.assertEqual(h['target_start'], 1000)
        self.assertEqual(h['target_end'], 2400)
        self.assertEqual(h['score'], 500)
        self.assertEqual(h['aligned_len'], 470)  # query_end - query_start

    def test_missing_as_tag_score_zero(self):
        """Score defaults to 0 when no AS tag present."""
        paf = self._write_paf([
            self._make_paf_line(extra_tags=['ms:i:480']),  # No AS tag
        ])
        protein_to_busco = {'100at7147:HASH': '100at7147'}
        hits = parse_miniprot_paf(paf, protein_to_busco)
        self.assertEqual(hits[0]['score'], 0)

    def test_unknown_protein_fallback(self):
        """Unknown protein name uses the part before colon."""
        paf = self._write_paf([
            self._make_paf_line(query='UNKNOWN:HASH', extra_tags=['AS:i:100']),
        ])
        protein_to_busco = {}  # Empty mapping
        hits = parse_miniprot_paf(paf, protein_to_busco)
        self.assertEqual(hits[0]['busco_id'], 'UNKNOWN')

    def test_short_lines_skipped(self):
        """Lines with < 12 fields are skipped."""
        paf = self._write_paf([
            "short\tline\twith\tfew\tfields",
            self._make_paf_line(extra_tags=['AS:i:100']),
        ])
        protein_to_busco = {'100at7147:HASH': '100at7147'}
        hits = parse_miniprot_paf(paf, protein_to_busco)
        self.assertEqual(len(hits), 1)

    def test_empty_paf(self):
        """Empty PAF file returns empty list."""
        paf = self._write_paf([])
        hits = parse_miniprot_paf(paf, {})
        self.assertEqual(hits, [])

    def test_multiple_hits(self):
        """Multiple hits are all returned."""
        paf = self._write_paf([
            self._make_paf_line(query='A:H1', target='contig1', extra_tags=['AS:i:100']),
            self._make_paf_line(query='B:H2', target='contig2', extra_tags=['AS:i:200']),
            self._make_paf_line(query='A:H1', target='contig3', extra_tags=['AS:i:300']),
        ])
        protein_to_busco = {'A:H1': 'BID_A', 'B:H2': 'BID_B'}
        hits = parse_miniprot_paf(paf, protein_to_busco)
        self.assertEqual(len(hits), 3)
        busco_ids = [h['busco_id'] for h in hits]
        self.assertEqual(busco_ids, ['BID_A', 'BID_B', 'BID_A'])


class TestWriteOdbOutput(unittest.TestCase):
    """Tests for write_odb_output()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.tmpdir, 'output')

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_per_contig_tsv_created(self):
        """Per-contig TSV files are created in the expected directory structure."""
        contig_results = {
            'contig1': {
                'BID001': ('Complete', 1000, 2440, 500, 480),
            },
        }
        all_busco_ids = {'BID001', 'BID002'}
        write_odb_output(contig_results, all_busco_ids, self.output_dir,
                         'diptera_odb10')
        tsv_path = os.path.join(
            self.output_dir, 'odbaln_contig1', 'run_contig1',
            'full_table_contig1.tsv')
        self.assertTrue(os.path.exists(tsv_path))

    def test_summary_file_created(self):
        """Concatenated summary file is created."""
        contig_results = {
            'contig1': {'BID001': ('Complete', 1000, 2440, 500, 480)},
        }
        all_busco_ids = {'BID001'}
        write_odb_output(contig_results, all_busco_ids, self.output_dir,
                         'diptera_odb10')
        summary_path = os.path.join(self.output_dir, 'full_table_results.tsv')
        self.assertTrue(os.path.exists(summary_path))

    def test_per_contig_file_content(self):
        """Per-contig TSV contains correct classification lines."""
        contig_results = {
            'contig1': {
                'BID001': ('Complete', 1000, 2440, 500, 480),
            },
        }
        all_busco_ids = {'BID001', 'BID002'}
        write_odb_output(contig_results, all_busco_ids, self.output_dir,
                         'diptera_odb10')
        tsv_path = os.path.join(
            self.output_dir, 'odbaln_contig1', 'run_contig1',
            'full_table_contig1.tsv')
        with open(tsv_path) as f:
            content = f.read()
        self.assertIn('BID001\tComplete\tcontig1\t1000\t2440\t500\t480', content)
        self.assertIn('BID002\tMissing', content)

    def test_summary_file_content(self):
        """Summary file has non-Missing hits and Missing entries for absent BUSCOs."""
        contig_results = {
            'contig1': {'BID001': ('Complete', 1000, 2440, 500, 480)},
        }
        all_busco_ids = {'BID001', 'BID002'}
        write_odb_output(contig_results, all_busco_ids, self.output_dir,
                         'test_lineage')
        summary_path = os.path.join(self.output_dir, 'full_table_results.tsv')
        with open(summary_path) as f:
            content = f.read()
        self.assertIn('BID001\tComplete\tcontig1', content)
        self.assertIn('BID002\tMissing', content)

    def test_multiple_contigs_output(self):
        """Output created for multiple contigs."""
        contig_results = {
            'contig1': {'BID001': ('Complete', 1000, 2440, 500, 480)},
            'contig2': {'BID001': ('Fragmented', 2000, 2600, 300, 200)},
        }
        all_busco_ids = {'BID001'}
        write_odb_output(contig_results, all_busco_ids, self.output_dir,
                         'test_lineage')
        for contig in ['contig1', 'contig2']:
            tsv = os.path.join(
                self.output_dir, 'odbaln_' + contig, 'run_' + contig,
                'full_table_' + contig + '.tsv')
            self.assertTrue(os.path.exists(tsv), f"Missing TSV for {contig}")

    def test_header_lines_present(self):
        """Per-contig TSV has header/comment lines."""
        contig_results = {
            'contig1': {'BID001': ('Complete', 1000, 2440, 500, 480)},
        }
        all_busco_ids = {'BID001'}
        write_odb_output(contig_results, all_busco_ids, self.output_dir,
                         'diptera_odb10')
        tsv_path = os.path.join(
            self.output_dir, 'odbaln_contig1', 'run_contig1',
            'full_table_contig1.tsv')
        with open(tsv_path) as f:
            lines = f.readlines()
        comment_lines = [l for l in lines if l.startswith('#')]
        self.assertGreaterEqual(len(comment_lines), 4)
        # Check lineage name appears in header
        header_text = ''.join(comment_lines)
        self.assertIn('diptera_odb10', header_text)


class TestOpenProteinFile(unittest.TestCase):
    """Tests for _open_protein_file()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_plain_text_file(self):
        """Opens plain text file normally."""
        path = os.path.join(self.tmpdir, 'test.faa')
        with open(path, 'w') as f:
            f.write(">protein1\nMSEQ\n")
        with _open_protein_file(path) as fh:
            content = fh.read()
        self.assertIn('>protein1', content)

    def test_gzipped_file(self):
        """Opens gzipped file in text mode."""
        path = os.path.join(self.tmpdir, 'test.faa.gz')
        with gzip.open(path, 'wt') as f:
            f.write(">protein1\nMSEQ\n")
        with _open_protein_file(path) as fh:
            content = fh.read()
        self.assertIn('>protein1', content)


class TestIntegrationClassifyPipeline(unittest.TestCase):
    """Integration tests that chain multiple functions together."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.lineage_dir = os.path.join(self.tmpdir, 'test_odb10')
        os.makedirs(self.lineage_dir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _setup_lineage(self):
        """Create a minimal ODB10-style lineage dataset."""
        # scores_cutoff
        with open(os.path.join(self.lineage_dir, 'scores_cutoff'), 'w') as f:
            f.write("100at7147\t450.0\n")
            f.write("200at7147\t300.0\n")
            f.write("300at7147\t200.0\n")

        # lengths_cutoff
        with open(os.path.join(self.lineage_dir, 'lengths_cutoff'), 'w') as f:
            f.write("100at7147\t42\t500.0\t50.0\n")
            f.write("200at7147\t38\t300.0\t30.0\n")
            f.write("300at7147\t25\t200.0\t20.0\n")

        # protein FASTA
        with open(os.path.join(self.lineage_dir, 'refseq_db.faa'), 'w') as f:
            f.write(">100at7147_101020_0:hash1\n")
            f.write("MSEQVENCEPROTEIN\n")
            f.write(">200at7147_202030_0:hash2\n")
            f.write("MOTHERSEQUENCE\n")
            f.write(">300at7147_303040_0:hash3\n")
            f.write("MTHIRDPROTEIN\n")

        # dataset.cfg
        with open(os.path.join(self.lineage_dir, 'dataset.cfg'), 'w') as f:
            f.write("name=test_lineage\n")
            f.write("species=Test species\n")

    def test_full_load_and_map_pipeline(self):
        """Load scores, lengths, build mapping, verify consistency."""
        self._setup_lineage()

        scores = load_scores_cutoff(self.lineage_dir)
        self.assertEqual(len(scores), 3)

        lengths = load_lengths_cutoff(self.lineage_dir, 'odb10')
        self.assertEqual(len(lengths), 3)

        protein_file, odb_version = find_protein_file(self.lineage_dir)
        self.assertIsNotNone(protein_file)
        self.assertEqual(odb_version, 'odb10')

        mapping, all_ids = build_protein_to_busco_map(protein_file, scores)
        self.assertEqual(len(mapping), 3)
        self.assertEqual(all_ids, {'100at7147', '200at7147', '300at7147'})

        lineage_name = detect_lineage_name(self.lineage_dir)
        self.assertEqual(lineage_name, 'test_lineage')

    def test_load_classify_and_write_pipeline(self):
        """Full pipeline: load data, classify hits, write output."""
        self._setup_lineage()

        scores = load_scores_cutoff(self.lineage_dir)
        lengths = load_lengths_cutoff(self.lineage_dir, 'odb10')
        protein_file, _ = find_protein_file(self.lineage_dir)
        mapping, all_ids = build_protein_to_busco_map(protein_file, scores)

        # Simulate miniprot hits
        hits = [
            {
                'busco_id': '100at7147', 'contig': 'contig1',
                'query_len': 500, 'query_start': 0, 'query_end': 480,
                'target_start': 1000, 'target_end': 2440,
                'score': 500, 'aligned_len': 480,
            },
            {
                'busco_id': '200at7147', 'contig': 'contig1',
                'query_len': 300, 'query_start': 0, 'query_end': 100,
                'target_start': 5000, 'target_end': 5300,
                'score': 400, 'aligned_len': 100,
            },
            # 300at7147 is Missing (no hits)
        ]

        no_filter = {'sr': 1.0, 'cov': 0.0, 'gap': 0.0}
        contig_results = classify_buscos(hits, all_ids, scores, lengths, no_filter)
        self.assertIn('contig1', contig_results)
        self.assertEqual(contig_results['contig1']['100at7147'][0], 'Complete')
        # 200at7147: threshold = 300 - 2*30 = 240, aligned = 100 < 240 -> Fragmented
        self.assertEqual(contig_results['contig1']['200at7147'][0], 'Fragmented')

        # Write output
        output_dir = os.path.join(self.tmpdir, 'output')
        write_odb_output(contig_results, all_ids, output_dir, 'test_lineage')

        # Verify output files exist and are correct
        tsv = os.path.join(output_dir, 'odbaln_contig1', 'run_contig1',
                           'full_table_contig1.tsv')
        self.assertTrue(os.path.exists(tsv))

        with open(tsv) as f:
            content = f.read()
        self.assertIn('100at7147\tComplete', content)
        self.assertIn('200at7147\tFragmented', content)
        self.assertIn('300at7147\tMissing', content)

        summary = os.path.join(output_dir, 'full_table_results.tsv')
        self.assertTrue(os.path.exists(summary))
        with open(summary) as f:
            content = f.read()
        self.assertIn('300at7147\tMissing', content)

    def test_parse_paf_then_classify(self):
        """Parse a synthetic PAF file, then classify the results."""
        self._setup_lineage()

        scores = load_scores_cutoff(self.lineage_dir)
        lengths = load_lengths_cutoff(self.lineage_dir, 'odb10')
        protein_file, _ = find_protein_file(self.lineage_dir)
        mapping, all_ids = build_protein_to_busco_map(protein_file, scores)

        # Write synthetic PAF
        paf_path = os.path.join(self.tmpdir, 'test.paf')
        with open(paf_path, 'w') as f:
            # Complete hit for 100at7147
            f.write('\t'.join([
                '100at7147_101020_0:hash1', '500', '0', '480', '+',
                'contig1', '50000', '1000', '2440', '470', '480', '60',
                'AS:i:500'
            ]) + '\n')
            # Fragmented hit for 200at7147
            f.write('\t'.join([
                '200at7147_202030_0:hash2', '300', '0', '100', '+',
                'contig1', '50000', '5000', '5300', '95', '100', '60',
                'AS:i:400'
            ]) + '\n')

        hits = parse_miniprot_paf(paf_path, mapping)
        self.assertEqual(len(hits), 2)

        no_filter = {'sr': 1.0, 'cov': 0.0, 'gap': 0.0}
        contig_results = classify_buscos(hits, all_ids, scores, lengths, no_filter)
        self.assertEqual(contig_results['contig1']['100at7147'][0], 'Complete')
        self.assertEqual(contig_results['contig1']['200at7147'][0], 'Fragmented')


if __name__ == '__main__':
    unittest.main()
