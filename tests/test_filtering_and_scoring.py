#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for alignment filter thresholds, BUSCO score calculation,
ReduceASM filtering, and CalculateInverseProportion.

Uses realistic value ranges from actual Anopheles funestus data:
- QPct: 0.2 - 1.0
- PID: 0.2 - 1.0
- QRAlignLenPct: 0.2 - ~5.0
- qSize: 0 - 6749
"""
import os
import sys
import shutil
import unittest
from math import exp, log

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TESTS_DIR)
FIXTURES_DIR = os.path.join(TESTS_DIR, 'fixtures')

sys.argv = [
    'hapsolo.py',
    '-i', os.path.join(FIXTURES_DIR, 'test_assembly.fasta'),
    '--paf', os.path.join(FIXTURES_DIR, 'test_alignment.paf'),
    '-b', os.path.join(FIXTURES_DIR, 'busco')
]
sys.path.insert(0, PROJECT_DIR)
import hapsolo


class TestFilterThresholds(unittest.TestCase):
    """Test that alignment filter thresholds work correctly during PAF loading."""

    def setUp(self):
        self.workdir = os.path.join(TESTS_DIR, 'tmp_filter_test')
        if os.path.exists(self.workdir):
            shutil.rmtree(self.workdir)
        os.makedirs(self.workdir)

    def tearDown(self):
        if os.path.exists(self.workdir):
            shutil.rmtree(self.workdir)

    def _write_paf(self, lines):
        """Write PAF lines to a file and return the path."""
        path = os.path.join(self.workdir, 'test.paf')
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        return path

    def test_mincontigsize_filter(self):
        """Alignments with qLen < myMinContigSize should be excluded."""
        lines = [
            # qLen = 5000, should pass when minContig=1000
            "contigA\t5000\t0\t4000\t+\tcontigB\t6000\t100\t4100\t3900\t4000\t0\ttp:A:S\tcm:i:2600\ts1:i:3900\tdv:f:0.0001\trl:i:500",
            # qLen = 500, should fail when minContig=1000
            "contigC\t500\t0\t400\t+\tcontigD\t6000\t100\t500\t390\t400\t0\ttp:A:S\tcm:i:260\ts1:i:390\tdv:f:0.0001\trl:i:50",
        ]
        paf = self._write_paf(lines)
        hapsolo.myMinContigSize = 1000
        hapsolo.myMinPID = 0.0
        hapsolo.myMinQPctMin = 0.0
        hapsolo.myMinQRPctMin = 0.0
        result = hapsolo.CreateMM2AlignmentDataStructure(paf)
        # Only the first alignment should pass (qLen=5000 >= 1000)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['qName'], 'contigA')

    def test_minpid_filter(self):
        """Alignments with PID < myMinPID should be excluded."""
        lines = [
            # PID = matches/qAlignLen = 3900/4000 = 0.975
            "contigA\t5000\t0\t4000\t+\tcontigB\t6000\t100\t4100\t3900\t4000\t0\ttp:A:S\tcm:i:2600\ts1:i:3900\tdv:f:0.0001\trl:i:500",
            # PID = 50/4000 = 0.0125 (very low)
            "contigC\t5000\t0\t4000\t+\tcontigD\t6000\t100\t4100\t50\t4000\t0\ttp:A:S\tcm:i:30\ts1:i:50\tdv:f:0.05\trl:i:500",
        ]
        paf = self._write_paf(lines)
        hapsolo.myMinContigSize = 0
        hapsolo.myMinPID = 0.5
        hapsolo.myMinQPctMin = 0.0
        hapsolo.myMinQRPctMin = 0.0
        result = hapsolo.CreateMM2AlignmentDataStructure(paf)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['qName'], 'contigA')

    def test_minqpct_filter(self):
        """Alignments with QPct < myMinQPctMin should be excluded."""
        lines = [
            # QPct = qAlignLen/qLen = 4000/5000 = 0.8
            "contigA\t5000\t0\t4000\t+\tcontigB\t6000\t100\t4100\t3900\t4000\t0\ttp:A:S\tcm:i:2600\ts1:i:3900\tdv:f:0.0001\trl:i:500",
            # QPct = 100/5000 = 0.02 (very low)
            "contigC\t5000\t0\t100\t+\tcontigD\t6000\t100\t200\t95\t100\t0\ttp:A:S\tcm:i:60\ts1:i:95\tdv:f:0.001\trl:i:500",
        ]
        paf = self._write_paf(lines)
        hapsolo.myMinContigSize = 0
        hapsolo.myMinPID = 0.0
        hapsolo.myMinQPctMin = 0.5
        hapsolo.myMinQRPctMin = 0.0
        result = hapsolo.CreateMM2AlignmentDataStructure(paf)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['qName'], 'contigA')

    def test_all_filters_combined(self):
        """All filters should be applied simultaneously."""
        lines = [
            # High quality: PID=0.975, QPct=0.8, qLen=5000 -> passes all
            "contigA\t5000\t0\t4000\t+\tcontigB\t6000\t100\t4100\t3900\t4000\t0\ttp:A:S\tcm:i:2600\ts1:i:3900\tdv:f:0.0001\trl:i:500",
            # Low PID only
            "contigC\t5000\t0\t4000\t+\tcontigD\t6000\t100\t4100\t50\t4000\t0\ttp:A:S\tcm:i:30\ts1:i:50\tdv:f:0.05\trl:i:500",
            # Low QPct only
            "contigE\t5000\t0\t100\t+\tcontigF\t6000\t100\t200\t95\t100\t0\ttp:A:S\tcm:i:60\ts1:i:95\tdv:f:0.001\trl:i:500",
            # Small contig only
            "contigG\t500\t0\t400\t+\tcontigH\t6000\t100\t500\t390\t400\t0\ttp:A:S\tcm:i:260\ts1:i:390\tdv:f:0.0001\trl:i:50",
        ]
        paf = self._write_paf(lines)
        hapsolo.myMinContigSize = 1000
        hapsolo.myMinPID = 0.5
        hapsolo.myMinQPctMin = 0.5
        hapsolo.myMinQRPctMin = 0.0
        result = hapsolo.CreateMM2AlignmentDataStructure(paf)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['qName'], 'contigA')

    def test_zero_thresholds_pass_all(self):
        """With all thresholds at 0, all non-self alignments should pass."""
        lines = [
            "contigA\t5000\t0\t4000\t+\tcontigB\t6000\t100\t4100\t3900\t4000\t0\ttp:A:S\tcm:i:2600\ts1:i:3900\tdv:f:0.0001\trl:i:500",
            "contigC\t500\t0\t100\t+\tcontigD\t6000\t100\t200\t50\t100\t0\ttp:A:S\tcm:i:30\ts1:i:50\tdv:f:0.05\trl:i:50",
            # Self-alignment should still be excluded
            "contigA\t5000\t0\t4999\t+\tcontigA\t5000\t0\t4999\t4990\t4999\t0\ttp:A:S\tcm:i:3300\ts1:i:4990\tdv:f:0.0000\trl:i:500",
        ]
        paf = self._write_paf(lines)
        hapsolo.myMinContigSize = 0
        hapsolo.myMinPID = 0.0
        hapsolo.myMinQPctMin = 0.0
        hapsolo.myMinQRPctMin = 0.0
        result = hapsolo.CreateMM2AlignmentDataStructure(paf)
        # 2 non-self alignments
        self.assertEqual(len(result), 2)

    def test_notebook_thresholds(self):
        """Replicate the notebook's QPct>=0.7 and PID>=0.7 filtering."""
        lines = [
            # PID=0.975, QPct=0.8 -> passes 0.7/0.7
            "contigA\t5000\t0\t4000\t+\tcontigB\t6000\t100\t4100\t3900\t4000\t0\ttp:A:S\tcm:i:2600\ts1:i:3900\tdv:f:0.0001\trl:i:500",
            # PID=0.5, QPct=0.8 -> fails PID
            "contigC\t5000\t0\t4000\t+\tcontigD\t6000\t100\t4100\t2000\t4000\t0\ttp:A:S\tcm:i:1320\ts1:i:2000\tdv:f:0.01\trl:i:500",
            # PID=0.975, QPct=0.1 -> fails QPct
            "contigE\t5000\t0\t500\t+\tcontigF\t6000\t100\t600\t487\t500\t0\ttp:A:S\tcm:i:320\ts1:i:487\tdv:f:0.0001\trl:i:500",
            # PID=0.8, QPct=0.8 -> passes both
            "contigG\t5000\t0\t4000\t+\tcontigH\t6000\t100\t4100\t3200\t4000\t0\ttp:A:S\tcm:i:2100\ts1:i:3200\tdv:f:0.0005\trl:i:500",
        ]
        paf = self._write_paf(lines)
        hapsolo.myMinContigSize = 0
        hapsolo.myMinPID = 0.7
        hapsolo.myMinQPctMin = 0.7
        hapsolo.myMinQRPctMin = 0.0
        result = hapsolo.CreateMM2AlignmentDataStructure(paf)
        self.assertEqual(len(result), 2)
        names = set(result['qName'])
        self.assertIn('contigA', names)
        self.assertIn('contigG', names)


class TestCalculateInverseProportion(unittest.TestCase):
    """Test the CalculateInverseProportion function used in QRAlignLenPct filtering."""

    def test_value_at_1(self):
        """Inverse of 1.0 should be 1.0."""
        result = hapsolo.CalculateInverseProportion(1.0)
        self.assertAlmostEqual(result, 1.0)

    def test_value_at_half(self):
        """Inverse of 0.5: exp(-log2(0.5)) = exp(1) = e."""
        result = hapsolo.CalculateInverseProportion(0.5)
        self.assertAlmostEqual(result, exp(1.0))

    def test_value_at_quarter(self):
        """Inverse of 0.25: exp(-log2(0.25)) = exp(2) = e^2."""
        result = hapsolo.CalculateInverseProportion(0.25)
        self.assertAlmostEqual(result, exp(2.0))

    def test_low_value_passthrough(self):
        """Values < 0.02 should be returned as-is."""
        result = hapsolo.CalculateInverseProportion(0.01)
        self.assertAlmostEqual(result, 0.01)

    def test_monotonically_decreasing(self):
        """Smaller input values should produce larger inverse values."""
        inv_08 = hapsolo.CalculateInverseProportion(0.8)
        inv_05 = hapsolo.CalculateInverseProportion(0.5)
        inv_03 = hapsolo.CalculateInverseProportion(0.3)
        self.assertLess(inv_08, inv_05)
        self.assertLess(inv_05, inv_03)


class TestReduceASM(unittest.TestCase):
    """Test the ReduceASM function that filters contigs based on alignment thresholds.

    Note: conftest.py auto-restores all globals (mypddf, allcontigsset, etc.).
    """

    def test_reduce_asm_filters_bad_contigs(self):
        """ReduceASM should return contigs NOT in filtered alignment set."""
        import pandas as pd
        # Set up the global alignment DataFrame
        hapsolo.mypddf = pd.DataFrame({
            'qName': ['contigA', 'contigA', 'contigB'],
            'tName': ['contigC', 'contigD', 'contigE'],
            'qSize': [5000, 5000, 3000],
            'QPct': [0.8, 0.9, 0.3],
            'PID': [0.85, 0.9, 0.85],
            'QRAlignLenPct': [0.95, 1.0, 0.95],
        })
        hapsolo.allcontigsset = {'contigA', 'contigB', 'contigC', 'contigD', 'contigE'}

        # With thresholds that pass contigA but not contigB:
        # contigA has QPct=0.8,0.9 and PID=0.85,0.9 -> passes 0.7 thresholds
        # contigB has QPct=0.3 -> fails QPct threshold
        result = hapsolo.ReduceASM(0.7, 0.7, 0.2)
        # contigA is "bad" (has alignments passing thresholds), so should NOT be in good set
        self.assertNotIn('contigA', result)
        # contigB's alignment doesn't pass QPct threshold, so contigB stays "good"
        self.assertIn('contigB', result)

    def test_reduce_asm_high_thresholds_keeps_all(self):
        """Very high thresholds should keep all contigs (none pass filters)."""
        import pandas as pd
        hapsolo.mypddf = pd.DataFrame({
            'qName': ['contigA'],
            'tName': ['contigB'],
            'qSize': [5000],
            'QPct': [0.8],
            'PID': [0.85],
            'QRAlignLenPct': [0.95],
        })
        hapsolo.allcontigsset = {'contigA', 'contigB'}
        result = hapsolo.ReduceASM(0.99, 0.99, 0.2)
        # Nothing passes the strict PID/QPct, so all contigs are "good"
        self.assertEqual(result, {'contigA', 'contigB'})


class TestCalculateBuscos(unittest.TestCase):
    """Test the BUSCO scoring function with known BUSCO distributions.

    Note: conftest.py auto-restores all globals between tests.
    """

    def test_busco_scoring_with_fixture(self):
        """Test BUSCO score calculation with busco_scoring fixture data."""
        busco_dir = os.path.join(FIXTURES_DIR, 'busco_scoring')
        b2c, c2b = hapsolo.importBuscos(busco_dir)

        # With all 3 contigs: count Complete BUSCOs across all contigs
        # contigA Complete: BUSCO001, BUSCO002, BUSCO003
        # contigB Complete: BUSCO001, BUSCO003, BUSCO005, BUSCO006
        # contigC Complete: BUSCO002, BUSCO008, BUSCO009
        # BUSCO001: appears in contigA + contigB = 2 (duplicate)
        # BUSCO002: appears in contigA + contigC = 2 (duplicate)
        # BUSCO003: appears in contigA + contigB = 2 (duplicate)
        # BUSCO005: appears in contigB only = 1 (single)
        # BUSCO006: appears in contigB only = 1 (single)
        # BUSCO008: appears in contigC only = 1 (single)
        # BUSCO009: appears in contigC only = 1 (single)
        all_contigs = ['contigA', 'contigB', 'contigC']
        result = hapsolo.calculateBuscos(all_contigs, b2c, c2b)
        self.assertEqual(result['D'], 3)   # BUSCO001, BUSCO002, BUSCO003
        self.assertEqual(result['S'], 4)   # BUSCO005, BUSCO006, BUSCO008, BUSCO009
        self.assertEqual(result['C'], 7)   # S + D

    def test_busco_scoring_single_contig(self):
        """With only one contig, no BUSCOs should be duplicated."""
        busco_dir = os.path.join(FIXTURES_DIR, 'busco_scoring')
        b2c, c2b = hapsolo.importBuscos(busco_dir)

        # contigA alone: BUSCO001, BUSCO002, BUSCO003 all Complete, BUSCO004 Fragmented
        result = hapsolo.calculateBuscos(['contigA'], b2c, c2b)
        self.assertEqual(result['D'], 0)   # No duplicates with single contig
        self.assertEqual(result['S'], 3)   # BUSCO001, BUSCO002, BUSCO003
        self.assertEqual(result['F'], 1)   # BUSCO004

    def test_busco_scoring_subset_reduces_duplicates(self):
        """Removing a contig should reduce duplicate BUSCOs."""
        busco_dir = os.path.join(FIXTURES_DIR, 'busco_scoring')
        b2c, c2b = hapsolo.importBuscos(busco_dir)

        # contigA + contigC (no contigB):
        # BUSCO001: contigA only = single
        # BUSCO002: contigA + contigC = duplicate
        # BUSCO003: contigA only = single
        # BUSCO008: contigC only = single
        # BUSCO009: contigC only = single
        result = hapsolo.calculateBuscos(['contigA', 'contigC'], b2c, c2b)
        self.assertEqual(result['D'], 1)   # BUSCO002
        self.assertEqual(result['S'], 4)   # BUSCO001, BUSCO003, BUSCO008, BUSCO009
        self.assertEqual(result['C'], 5)   # S + D

    def test_busco_fragmented_not_counted_when_complete_exists(self):
        """Fragmented BUSCOs should not be counted if a complete copy exists."""
        busco_dir = os.path.join(FIXTURES_DIR, 'busco_scoring')
        b2c, c2b = hapsolo.importBuscos(busco_dir)

        # contigA has BUSCO004 as Fragmented
        # contigB has BUSCO007 as Fragmented
        # With contigA alone: BUSCO004 is fragmented (no complete copy) -> F=1
        result_a = hapsolo.calculateBuscos(['contigA'], b2c, c2b)
        self.assertEqual(result_a['F'], 1)

        # With all contigs: BUSCO004 is fragmented on contigA, not complete anywhere -> F=1
        # BUSCO007 is fragmented on contigB, not complete anywhere -> F=1
        result_all = hapsolo.calculateBuscos(['contigA', 'contigB', 'contigC'], b2c, c2b)
        self.assertEqual(result_all['F'], 2)  # BUSCO004 and BUSCO007

    def test_busco_missing_count(self):
        """Missing BUSCOs = total - single - duplicate - fragmented."""
        busco_dir = os.path.join(FIXTURES_DIR, 'busco_scoring')
        b2c, c2b = hapsolo.importBuscos(busco_dir)

        result = hapsolo.calculateBuscos(['contigA', 'contigB', 'contigC'], b2c, c2b)
        total_buscos = len(b2c)
        expected_missing = total_buscos - result['D'] - result['S'] - result['F']
        self.assertEqual(result['M'], expected_missing)

    def test_busco_empty_contig_list(self):
        """Empty contig list should result in all BUSCOs missing."""
        busco_dir = os.path.join(FIXTURES_DIR, 'busco_scoring')
        b2c, c2b = hapsolo.importBuscos(busco_dir)

        result = hapsolo.calculateBuscos([], b2c, c2b)
        self.assertEqual(result['S'], 0)
        self.assertEqual(result['D'], 0)
        self.assertEqual(result['F'], 0)
        self.assertEqual(result['M'], len(b2c))


class TestCalculatePctAlignEdgeCases(unittest.TestCase):
    """Additional edge cases for CalculatePctAlign beyond the basic tests."""

    def test_large_values(self):
        """Should handle large alignment values like real data (1.7M bp contigs)."""
        result = hapsolo.CalculatePctAlign(1754046, 1777443)
        self.assertAlmostEqual(result, 1754046 / 1777443, places=5)

    def test_very_small_alignment(self):
        """Should handle very small alignments correctly."""
        result = hapsolo.CalculatePctAlign(3, 5000)
        self.assertAlmostEqual(result, 3 / 5000, places=5)

    def test_equal_values(self):
        """When alignment equals total, result should be 1.0."""
        result = hapsolo.CalculatePctAlign(24191, 24191)
        self.assertAlmostEqual(result, 1.0)


class TestMyLinearFxn(unittest.TestCase):
    """Test the cost function used by hill climbing."""

    def test_default_weights(self):
        """With default weights (thetaS=1,D=1,M=1,F=0): (D+M)/(S)."""
        # Default weights: thetaS=1.0, thetaD=1.0, thetaM=1.0, thetaF=0.0
        # myLinearFxn(M, S, D, F, C) = (F*thetaF + D*thetaD + M*thetaM) / (S*thetaS)
        result = hapsolo.myLinearFxn(5, 10, 3, 2, 15)  # M=5, S=10, D=3, F=2
        # With defaults: (0*2 + 1*3 + 1*5) / (1*10) = 8/10 = 0.8
        self.assertAlmostEqual(result, 0.8)

    def test_perfect_assembly(self):
        """Assembly with all single BUSCOs should have low cost."""
        # M=0, S=100, D=0, F=0
        result = hapsolo.myLinearFxn(0, 100, 0, 0, 100)
        self.assertAlmostEqual(result, 0.0)

    def test_all_duplicated(self):
        """Assembly with all duplicated should have high cost."""
        # M=0, S=1, D=99, F=0
        result = hapsolo.myLinearFxn(0, 1, 99, 0, 100)
        self.assertAlmostEqual(result, 99.0)

    def test_lower_is_better(self):
        """Reducing duplicates should reduce cost."""
        cost_high = hapsolo.myLinearFxn(5, 10, 20, 2, 37)
        cost_low = hapsolo.myLinearFxn(5, 10, 5, 2, 22)
        self.assertLess(cost_low, cost_high)


class TestCalculateAsmStats(unittest.TestCase):
    """Test assembly statistics calculation (N50, L50, etc.)."""

    def test_basic_stats(self):
        """Verify N50/L50 calculation with known contig sizes."""
        # Set up myContigsDict: name -> [size, headerPos, startPos, endPos]
        hapsolo.myContigsDict = {
            'c1': [1000, 0, 0, 0],
            'c2': [2000, 0, 0, 0],
            'c3': [3000, 0, 0, 0],
            'c4': [4000, 0, 0, 0],
        }
        # Total = 10000, 50% = 5000
        # Sorted desc: 4000, 3000, 2000, 1000
        # Cumulative: 4000 (<5000), 4000+3000=7000 (>5000) -> N50=3000, L50=2
        asmsize, n50, l50, largest = hapsolo.calculateasmstats({'c1', 'c2', 'c3', 'c4'})
        self.assertEqual(asmsize, 10000)
        self.assertEqual(n50, 3000)
        self.assertEqual(l50, 2)  # 2 contigs needed to reach 50%
        self.assertEqual(largest, 4000)

    def test_single_contig(self):
        """Single contig assembly should have N50 = contig size."""
        hapsolo.myContigsDict = {
            'c1': [50000, 0, 0, 0],
        }
        asmsize, n50, l50, largest = hapsolo.calculateasmstats({'c1'})
        self.assertEqual(asmsize, 50000)
        self.assertEqual(n50, 50000)
        self.assertEqual(l50, 1)  # 1 contig needed
        self.assertEqual(largest, 50000)

    def test_subset_of_contigs(self):
        """Stats should only consider the given contig set."""
        hapsolo.myContigsDict = {
            'c1': [1000, 0, 0, 0],
            'c2': [2000, 0, 0, 0],
            'c3': [3000, 0, 0, 0],
        }
        # Only c2 and c3: total=5000, sorted desc: 3000, 2000
        # Cumulative: 3000 (>2500) -> N50=3000, L50=1
        asmsize, n50, l50, largest = hapsolo.calculateasmstats({'c2', 'c3'})
        self.assertEqual(asmsize, 5000)
        self.assertEqual(l50, 1)  # first contig alone exceeds 50%
        self.assertEqual(n50, 3000)
        self.assertEqual(largest, 3000)

    def test_contig_not_in_dict_ignored(self):
        """Contigs not in myContigsDict should be silently skipped."""
        hapsolo.myContigsDict = {
            'c1': [5000, 0, 0, 0],
        }
        asmsize, n50, l50, largest = hapsolo.calculateasmstats({'c1', 'missing_contig'})
        self.assertEqual(asmsize, 5000)


class TestWriteNewAssembly(unittest.TestCase):
    """Test FASTA output writing."""

    def setUp(self):
        self.workdir = os.path.join(TESTS_DIR, 'tmp_write_test')
        if os.path.exists(self.workdir):
            shutil.rmtree(self.workdir)
        os.makedirs(self.workdir)

    def tearDown(self):
        # Clean up asms/ dir created by WriteNewAssembly
        asms_dir = os.path.join(self.workdir, 'asms')
        if os.path.exists(asms_dir):
            shutil.rmtree(asms_dir)
        if os.path.exists(self.workdir):
            shutil.rmtree(self.workdir)

    def test_writes_good_contigs_only(self):
        """Only contigs in the good set should appear in output."""
        fasta = os.path.join(FIXTURES_DIR, 'test_assembly.fasta')
        hapsolo.myContigsDict = hapsolo.CalculateContigSizes(fasta)

        # Write only tig00000001 and tig00000003
        good_set = {'tig00000001', 'tig00000003'}
        # WriteNewAssembly creates asms/ relative to CWD
        orig_dir = os.getcwd()
        os.chdir(self.workdir)
        try:
            hapsolo.WriteNewAssembly(fasta, 'test_output.fasta', good_set)
        finally:
            os.chdir(orig_dir)

        outfile = os.path.join(self.workdir, 'asms', 'test_output.fasta')
        self.assertTrue(os.path.exists(outfile))

        # Count output contigs
        headers = []
        with open(outfile) as f:
            for line in f:
                if line.startswith('>'):
                    headers.append(line.strip()[1:].split()[0])
        self.assertEqual(len(headers), 2)
        self.assertIn('tig00000001', headers)
        self.assertIn('tig00000003', headers)
        self.assertNotIn('tig00000002', headers)

    def _write_unsorted_fasta(self):
        """60 contigs in a non-alphabetical order, so set or name order would differ from it."""
        import random
        names = ['ctg%03d_%s' % (i, 'xyz'[i % 3]) for i in range(60)]
        random.Random(7).shuffle(names)
        path = os.path.join(self.workdir, 'unsorted.fasta')
        with open(path, 'w') as fh:
            for i, n in enumerate(names):
                fh.write('>%s\n%s\n%s\n' % (n, 'ACGT' * (5 + i), 'TTGA' * 3))
        return path, names

    def test_output_follows_input_assembly_order(self):
        """Kept contigs are written in the input FASTA's order, not set or name order."""
        fasta, names = self._write_unsorted_fasta()
        hapsolo.myContigsDict = hapsolo.CalculateContigSizes(fasta)
        keep = set(names[::2])
        orig_dir = os.getcwd()
        os.chdir(self.workdir)
        try:
            hapsolo.WriteNewAssembly(fasta, 'ordered.fasta', keep)
        finally:
            os.chdir(orig_dir)
        with open(os.path.join(self.workdir, 'asms', 'ordered.fasta')) as fh:
            headers = [l[1:].strip() for l in fh if l.startswith('>')]
        self.assertEqual(headers, [n for n in names if n in keep])
        self.assertNotEqual(headers, sorted(headers))

    def test_output_identical_across_processes(self):
        """Two processes with different str hash seeds must write byte-identical FASTAs."""
        import subprocess
        fasta, names = self._write_unsorted_fasta()
        project_dir = os.path.dirname(TESTS_DIR)
        script = ('import sys; sys.path.insert(0, %r)\n'
                  'from hapsolo.assembly import CalculateContigSizes, WriteNewAssembly\n'
                  'sizes, _ = CalculateContigSizes(%r)\n'
                  'keep = set(%r)\n'
                  'WriteNewAssembly(%r, sys.argv[1], keep, sizes, outdir=%r)\n'
                  % (project_dir, fasta, names[::3], fasta, self.workdir))
        outputs = []
        for seed in ('1', '2'):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            res = subprocess.run([sys.executable, '-c', script, 'seed%s.fasta' % seed],
                                 env=env, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stderr)
            with open(os.path.join(self.workdir, 'seed%s.fasta' % seed), 'rb') as fh:
                outputs.append(fh.read())
        self.assertEqual(outputs[0], outputs[1])


class TestUniquepriorityqueue(unittest.TestCase):
    """Test the priority queue used by hill climbing."""

    def test_adds_and_sorts(self):
        """Should add a value and keep sorted by score (index 0)."""
        pqlist = [[0.5, {'c1', 'c2'}, {'c3'}, {}, [0.7, 0.7, 0.7]]]
        new_val = [0.3, {'c1', 'c2', 'c3'}, set(), {}, [0.8, 0.8, 0.8]]
        result = hapsolo.uniquepriorityqueue(pqlist, new_val)
        # Lower score first
        self.assertAlmostEqual(result[0][0], 0.3)

    def test_removes_duplicates(self):
        """Duplicate contig sets should keep the one with better score."""
        same_set = frozenset({'c1', 'c2'})
        pqlist = [[0.5, {'c1', 'c2'}, {'c3'}, {}, [0.7, 0.7, 0.7]]]
        new_val = [0.3, {'c1', 'c2'}, {'c3'}, {}, [0.8, 0.8, 0.8]]
        result = hapsolo.uniquepriorityqueue(pqlist, new_val)
        # Only one entry should remain (duplicates removed)
        self.assertEqual(len(result), 1)
        # The one with lower score should survive
        self.assertAlmostEqual(result[0][0], 0.3)


if __name__ == '__main__':
    unittest.main()
