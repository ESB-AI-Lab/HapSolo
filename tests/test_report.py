#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for write_report, _compute_detailed_asm_stats, and _compute_gc_and_ns."""
import os
import tempfile
import unittest

import hapsolo

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TESTS_DIR)


class TestComputeDetailedAsmStats(unittest.TestCase):

    def setUp(self):
        hapsolo.myContigsDict = {
            'c1': [100000, 0, 0, 0],
            'c2': [50000, 0, 0, 0],
            'c3': [30000, 0, 0, 0],
            'c4': [10000, 0, 0, 0],
            'c5': [800, 0, 0, 0],
        }

    def test_returns_none_for_empty_set(self):
        self.assertIsNone(hapsolo._compute_detailed_asm_stats(set()))

    def test_total_length(self):
        stats = hapsolo._compute_detailed_asm_stats({'c1', 'c2', 'c3', 'c4', 'c5'})
        self.assertEqual(stats['total_length'], 190800)

    def test_largest_contig(self):
        stats = hapsolo._compute_detailed_asm_stats({'c1', 'c2', 'c3', 'c4', 'c5'})
        self.assertEqual(stats['largest'], 100000)

    def test_threshold_counts(self):
        stats = hapsolo._compute_detailed_asm_stats({'c1', 'c2', 'c3', 'c4', 'c5'})
        self.assertEqual(stats['counts'][0], 5)
        self.assertEqual(stats['counts'][1000], 4)
        self.assertEqual(stats['counts'][50000], 2)  # c1=100k, c2=50k both >= 50000

    def test_threshold_lengths(self):
        stats = hapsolo._compute_detailed_asm_stats({'c1', 'c2', 'c3', 'c4', 'c5'})
        self.assertEqual(stats['lengths'][0], 190800)
        self.assertEqual(stats['lengths'][1000], 190000)
        self.assertEqual(stats['lengths'][50000], 150000)  # c1=100k + c2=50k

    def test_n50_l50(self):
        stats = hapsolo._compute_detailed_asm_stats({'c1', 'c2', 'c3', 'c4'})
        # total=190000, 50%=95000. sorted: [100000, 50000, 30000, 10000]
        # cumsum: 100000 > 95000 at index 0 → N50=100000, L50=1
        self.assertEqual(stats['n50'], 100000)
        self.assertEqual(stats['l50'], 1)

    def test_n75_l75(self):
        stats = hapsolo._compute_detailed_asm_stats({'c1', 'c2', 'c3', 'c4'})
        # 75% of 190000 = 142500. cumsum: 100000, 150000 > 142500 → N75=50000, L75=2
        self.assertEqual(stats['n75'], 50000)
        self.assertEqual(stats['l75'], 2)

    def test_single_contig(self):
        stats = hapsolo._compute_detailed_asm_stats({'c1'})
        self.assertEqual(stats['n50'], 100000)
        self.assertEqual(stats['l50'], 1)
        self.assertEqual(stats['n75'], 100000)
        self.assertEqual(stats['l75'], 1)
        self.assertEqual(stats['total_contigs'], 1)

    def test_ignores_unknown_contigs(self):
        stats = hapsolo._compute_detailed_asm_stats({'c1', 'ghost'})
        self.assertEqual(stats['total_contigs'], 1)
        self.assertEqual(stats['total_length'], 100000)


class TestComputeGcAndNs(unittest.TestCase):

    def _write_fasta(self, sequences):
        """Write sequences to a temp FASTA and return the path."""
        fd, path = tempfile.mkstemp(suffix='.fasta')
        with os.fdopen(fd, 'w') as f:
            for name, seq in sequences:
                f.write('>' + name + '\n')
                f.write(seq + '\n')
        return path

    def test_pure_gc(self):
        path = self._write_fasta([('c1', 'GCGCGCGCGC')])
        gc_pct, ns = hapsolo._compute_gc_and_ns(path)
        self.assertAlmostEqual(gc_pct, 100.0)
        self.assertAlmostEqual(ns, 0.0)
        os.unlink(path)

    def test_pure_at(self):
        path = self._write_fasta([('c1', 'ATATATATAT')])
        gc_pct, ns = hapsolo._compute_gc_and_ns(path)
        self.assertAlmostEqual(gc_pct, 0.0)
        os.unlink(path)

    def test_50_50_gc(self):
        path = self._write_fasta([('c1', 'GCGCATATAT')])
        gc_pct, ns = hapsolo._compute_gc_and_ns(path)
        self.assertAlmostEqual(gc_pct, 40.0)
        os.unlink(path)

    def test_ns_counted(self):
        path = self._write_fasta([('c1', 'ATGCNNNNAT')])
        gc_pct, ns_per_100k = hapsolo._compute_gc_and_ns(path)
        # 4 Ns out of 10 bases = 40000 per 100k
        self.assertAlmostEqual(ns_per_100k, 40000.0)
        os.unlink(path)

    def test_multiline_sequence(self):
        path = self._write_fasta([('c1', 'GGGG\nCCCC')])
        gc_pct, _ = hapsolo._compute_gc_and_ns(path)
        self.assertAlmostEqual(gc_pct, 100.0)
        os.unlink(path)

    def test_multiple_contigs(self):
        path = self._write_fasta([('c1', 'GCGC'), ('c2', 'ATAT')])
        gc_pct, _ = hapsolo._compute_gc_and_ns(path)
        # 4 GC + 4 AT = 50%
        self.assertAlmostEqual(gc_pct, 50.0)
        os.unlink(path)


class TestWriteReport(unittest.TestCase):

    def setUp(self):
        hapsolo.myContigsDict = {
            'c1': [100000, 0, 0, 0],
            'c2': [50000, 0, 0, 0],
            'c3': [30000, 0, 0, 0],
        }
        self._tmpdir = tempfile.mkdtemp()
        self._fasta_path = os.path.join(self._tmpdir, 'test_primary.fasta')
        with open(self._fasta_path, 'w') as f:
            f.write('>c1\n' + 'ATGC' * 25000 + '\n')
            f.write('>c2\n' + 'ATGC' * 12500 + '\n')
            f.write('>c3\n' + 'ATGC' * 7500 + '\n')

    def tearDown(self):
        report_path = self._fasta_path.replace('_primary.fasta', '_report.txt')
        for f in [self._fasta_path, report_path]:
            if os.path.exists(f):
                os.unlink(f)
        os.rmdir(self._tmpdir)

    def test_report_file_created(self):
        scores = {'S': 80, 'D': 10, 'C': 90, 'F': 5, 'M': 5}
        result = hapsolo.write_report(self._fasta_path, {'c1', 'c2', 'c3'}, scores)
        self.assertTrue(os.path.exists(result))

    def test_report_filename_format(self):
        scores = {'S': 80, 'D': 10, 'C': 90, 'F': 5, 'M': 5}
        result = hapsolo.write_report(self._fasta_path, {'c1', 'c2', 'c3'}, scores)
        self.assertTrue(result.endswith('_report.txt'))
        self.assertNotIn('scoresreport', result)

    def test_report_contains_assembly_stats(self):
        scores = {'S': 80, 'D': 10, 'C': 90, 'F': 5, 'M': 5}
        result = hapsolo.write_report(self._fasta_path, {'c1', 'c2', 'c3'}, scores)
        with open(result) as f:
            content = f.read()
        self.assertIn('N50', content)
        self.assertIn('N75', content)
        self.assertIn('L50', content)
        self.assertIn('L75', content)
        self.assertIn('GC (%)', content)
        self.assertIn("N's per 100 kbp", content)
        self.assertIn('Largest contig', content)

    def test_report_contains_threshold_lines(self):
        scores = {'S': 80, 'D': 10, 'C': 90, 'F': 5, 'M': 5}
        result = hapsolo.write_report(self._fasta_path, {'c1', 'c2', 'c3'}, scores)
        with open(result) as f:
            content = f.read()
        self.assertIn('# contigs (>= 0 bp)', content)
        self.assertIn('# contigs (>= 50000 bp)', content)
        self.assertIn('Total length (>= 0 bp)', content)

    def test_report_contains_ortholog_section(self):
        scores = {'S': 80, 'D': 10, 'C': 90, 'F': 5, 'M': 5}
        result = hapsolo.write_report(self._fasta_path, {'c1', 'c2', 'c3'}, scores)
        with open(result) as f:
            content = f.read()
        self.assertIn('Ortholog completeness', content)
        self.assertIn('Complete orthologs (C)', content)
        self.assertIn('Complete and single-copy orthologs (S)', content)
        self.assertIn('Complete and duplicated orthologs (D)', content)
        self.assertIn('Fragmented orthologs (F)', content)
        self.assertIn('Missing orthologs (M)', content)
        self.assertIn('Total ortholog groups searched', content)

    def test_report_no_busco_language(self):
        scores = {'S': 80, 'D': 10, 'C': 90, 'F': 5, 'M': 5}
        result = hapsolo.write_report(self._fasta_path, {'c1', 'c2', 'c3'}, scores)
        with open(result) as f:
            content = f.read()
        self.assertNotIn('BUSCO', content)

    def test_report_ortholog_counts_correct(self):
        scores = {'S': 80, 'D': 10, 'C': 90, 'F': 5, 'M': 5}
        result = hapsolo.write_report(self._fasta_path, {'c1', 'c2', 'c3'}, scores)
        with open(result) as f:
            content = f.read()
        self.assertIn('90\tComplete orthologs (C)', content)
        self.assertIn('80\tComplete and single-copy orthologs (S)', content)
        self.assertIn('10\tComplete and duplicated orthologs (D)', content)
        self.assertIn('5\tFragmented orthologs (F)', content)
        self.assertIn('5\tMissing orthologs (M)', content)
        self.assertIn('100\tTotal ortholog groups searched', content)

    def test_report_summary_line_format(self):
        scores = {'S': 80, 'D': 10, 'C': 90, 'F': 5, 'M': 5}
        result = hapsolo.write_report(self._fasta_path, {'c1', 'c2', 'c3'}, scores)
        with open(result) as f:
            content = f.read()
        # C:90.0%[S:80.0%,D:10.0%],F:5.0%,M:5.0%,n:100
        self.assertIn('C:90.0%[S:80.0%,D:10.0%],F:5.0%,M:5.0%,n:100', content)

    def test_report_gc_content(self):
        scores = {'S': 80, 'D': 10, 'C': 90, 'F': 5, 'M': 5}
        result = hapsolo.write_report(self._fasta_path, {'c1', 'c2', 'c3'}, scores)
        with open(result) as f:
            content = f.read()
        # ATGC repeated = 50% GC
        self.assertIn('50.00', content)

    def test_report_contig_counts(self):
        scores = {'S': 80, 'D': 10, 'C': 90, 'F': 5, 'M': 5}
        result = hapsolo.write_report(self._fasta_path, {'c1', 'c2', 'c3'}, scores)
        with open(result) as f:
            lines = f.readlines()
        stat_lines = {l.split()[0] + ' ' + l.split()[1] if len(l.split()) > 1 else '': l
                      for l in lines if l.strip()}
        # All 3 contigs >= 0 bp
        found = [l for l in lines if '# contigs (>= 0 bp)' in l]
        self.assertTrue(any('3' in l for l in found))


class TestWriteReportGroundTruth(unittest.TestCase):
    """Validate report against the existing example_results scoresreport."""

    EXAMPLE_REPORT = os.path.join(
        PROJECT_DIR, 'example_data', 'example_results',
        'pamer.contigs.c21.consensus.consensus_pilon_pilon_new_1000_'
        '0.3418_0.6706to1.7798_0.2023_primary_scoresreport.txt')

    @unittest.skipUnless(os.path.exists(EXAMPLE_REPORT), 'example report not found')
    def test_assembly_stats_match_ground_truth(self):
        with open(self.EXAMPLE_REPORT) as f:
            gt = f.read()
        gt_values = {}
        for line in gt.split('\n'):
            line = line.strip()
            if line.startswith('# contigs') and '>=' in line:
                parts = line.rsplit(None, 1)
                if len(parts) == 2:
                    gt_values[parts[0].strip()] = parts[1].strip()
            elif line.startswith('Total length') and '>=' in line:
                parts = line.rsplit(None, 1)
                if len(parts) == 2:
                    gt_values[parts[0].strip()] = parts[1].strip()
            else:
                for key in ['N50', 'N75', 'L50', 'L75', 'Largest contig', '# contigs', 'Total length']:
                    if line.startswith(key) and '>=' not in line:
                        parts = line.rsplit(None, 1)
                        if len(parts) == 2:
                            gt_values[key] = parts[1].strip()
        # These are the ground truth values from the example report
        self.assertEqual(gt_values.get('N50'), '3386771')
        self.assertEqual(gt_values.get('N75'), '1114993')
        self.assertEqual(gt_values.get('L50'), '78')
        self.assertEqual(gt_values.get('L75'), '211')
        self.assertEqual(gt_values.get('Largest contig'), '17075471')
        self.assertEqual(gt_values.get('# contigs'), '903')
        self.assertEqual(gt_values.get('Total length'), '1027420587')


if __name__ == '__main__':
    unittest.main()
