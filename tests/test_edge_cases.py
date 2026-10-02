#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Edge case tests for hapsolo.py functions.

Tests boundary conditions, empty inputs, single-element inputs,
and zero-value scenarios that could trigger crashes or wrong results.
"""
import os
import sys
import shutil
import unittest
import pandas as pd

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


# ── CalculateContigSizes edge cases ─────────────────────────────────────────

class TestCalculateContigSizesSingleContig(unittest.TestCase):

    def setUp(self):
        self.workdir = os.path.join(TESTS_DIR, 'tmp_single_contig')
        if os.path.exists(self.workdir):
            shutil.rmtree(self.workdir)
        os.makedirs(self.workdir)

    def tearDown(self):
        if os.path.exists(self.workdir):
            shutil.rmtree(self.workdir)

    def test_single_contig_fasta(self):
        """FASTA with exactly 1 contig should work."""
        fasta = os.path.join(self.workdir, 'single.fasta')
        with open(fasta, 'w') as f:
            f.write('>onlycontig\n')
            f.write('ATCGATCGATCG\n')
        result = hapsolo.CalculateContigSizes(fasta)
        self.assertEqual(len(result), 1)
        self.assertIn('onlycontig', result)
        self.assertEqual(result['onlycontig'][0], 12)

    def test_single_contig_multiline(self):
        """Single contig with multiple sequence lines."""
        fasta = os.path.join(self.workdir, 'single_multi.fasta')
        with open(fasta, 'w') as f:
            f.write('>mycontig\n')
            f.write('ATCG' * 20 + '\n')  # 80 chars
            f.write('GCTA' * 20 + '\n')  # 80 chars
            f.write('AAAA' * 10 + '\n')  # 40 chars
        result = hapsolo.CalculateContigSizes(fasta)
        self.assertEqual(result['mycontig'][0], 200)

    def test_single_line_sequence(self):
        """Contig with only 1 sequence line (no wrap)."""
        fasta = os.path.join(self.workdir, 'single_line.fasta')
        with open(fasta, 'w') as f:
            f.write('>short\n')
            f.write('ATCG\n')
        result = hapsolo.CalculateContigSizes(fasta)
        self.assertEqual(result['short'][0], 4)


# ── WriteNewAssembly edge cases ─────────────────────────────────────────────

class TestWriteNewAssemblySingleContig(unittest.TestCase):

    def setUp(self):
        self.workdir = os.path.join(TESTS_DIR, 'tmp_write_single')
        if os.path.exists(self.workdir):
            shutil.rmtree(self.workdir)
        os.makedirs(self.workdir)

    def tearDown(self):
        if os.path.exists(self.workdir):
            shutil.rmtree(self.workdir)

    def test_write_single_contig(self):
        """WriteNewAssembly with only 1 contig in good set."""
        fasta = os.path.join(FIXTURES_DIR, 'test_assembly.fasta')
        hapsolo.myContigsDict = hapsolo.CalculateContigSizes(fasta)

        orig_dir = os.getcwd()
        os.chdir(self.workdir)
        try:
            hapsolo.WriteNewAssembly(fasta, 'single.fasta', {'tig00000002'})
        finally:
            os.chdir(orig_dir)

        outfile = os.path.join(self.workdir, 'asms', 'single.fasta')
        self.assertTrue(os.path.exists(outfile))

        headers = []
        seq = ''
        with open(outfile) as f:
            for line in f:
                if line.startswith('>'):
                    headers.append(line.strip()[1:])
                else:
                    seq += line.strip()
        self.assertEqual(len(headers), 1)
        self.assertEqual(headers[0], 'tig00000002')
        self.assertEqual(len(seq), 160)

    def test_write_preserves_sequence_content(self):
        """Extracted sequence must match the original."""
        fasta = os.path.join(FIXTURES_DIR, 'test_assembly.fasta')
        hapsolo.myContigsDict = hapsolo.CalculateContigSizes(fasta)

        # Read original sequence for tig00000001
        orig_seq = ''
        reading = False
        with open(fasta) as f:
            for line in f:
                if line.startswith('>tig00000001'):
                    reading = True
                    continue
                elif line.startswith('>'):
                    reading = False
                elif reading:
                    orig_seq += line.strip()

        orig_dir = os.getcwd()
        os.chdir(self.workdir)
        try:
            hapsolo.WriteNewAssembly(fasta, 'out.fasta', {'tig00000001'})
        finally:
            os.chdir(orig_dir)

        # Read extracted sequence
        extracted_seq = ''
        with open(os.path.join(self.workdir, 'asms', 'out.fasta')) as f:
            for line in f:
                if not line.startswith('>'):
                    extracted_seq += line.strip()

        self.assertEqual(extracted_seq, orig_seq)


# ── Zero-length alignment edge cases ────────────────────────────────────────

class TestZeroLengthAlignment(unittest.TestCase):

    def setUp(self):
        self.workdir = os.path.join(TESTS_DIR, 'tmp_zero_align')
        if os.path.exists(self.workdir):
            shutil.rmtree(self.workdir)
        os.makedirs(self.workdir)
        hapsolo.myMinContigSize = 0
        hapsolo.myMinPID = 0.0
        hapsolo.myMinQPctMin = 0.0
        hapsolo.myMinQRPctMin = 0.0

    def tearDown(self):
        if os.path.exists(self.workdir):
            shutil.rmtree(self.workdir)

    def test_zero_length_alignment_no_crash(self):
        """PAF with qStart == qEnd (zero-length alignment) should not crash."""
        paf = os.path.join(self.workdir, 'zero.paf')
        with open(paf, 'w') as f:
            # qStart=100, qEnd=100 -> fqAlignLen=0
            f.write("contigA\t5000\t100\t100\t+\tcontigB\t6000\t200\t200\t0\t0\t0\ttp:A:S\tcm:i:0\ts1:i:0\tdv:f:0\trl:i:0\n")
            # Normal alignment for comparison
            f.write("contigC\t5000\t0\t4000\t+\tcontigD\t6000\t100\t4100\t3900\t4000\t0\ttp:A:S\tcm:i:2600\ts1:i:3900\tdv:f:0.0001\trl:i:500\n")
        result = hapsolo.CreateMM2AlignmentDataStructure(paf)
        # Zero-length alignment has QPct=0, PID=0, QRAlignLenPct=0
        # With thresholds at 0.0, it should still pass (0.0 >= 0.0)
        # But qName != tName check filters it only if they differ
        self.assertTrue(len(result) >= 1)

    def test_zero_length_both_sides(self):
        """Both query and target have zero-length alignment."""
        paf = os.path.join(self.workdir, 'zeroboth.paf')
        with open(paf, 'w') as f:
            f.write("contigA\t5000\t0\t0\t+\tcontigB\t6000\t0\t0\t0\t0\t0\ttp:A:S\tcm:i:0\ts1:i:0\tdv:f:0\trl:i:0\n")
        result = hapsolo.CreateMM2AlignmentDataStructure(paf)
        # Should not crash; fqAlignLen=0, frAlignLen=0
        # CalculatePctAlign(0, 0) returns 0.0
        self.assertIsNotNone(result)


# ── PAF minimum fields ──────────────────────────────────────────────────────

class TestPAFMinimumFields(unittest.TestCase):

    def setUp(self):
        self.workdir = os.path.join(TESTS_DIR, 'tmp_paf_min')
        if os.path.exists(self.workdir):
            shutil.rmtree(self.workdir)
        os.makedirs(self.workdir)
        hapsolo.myMinContigSize = 0
        hapsolo.myMinPID = 0.0
        hapsolo.myMinQPctMin = 0.0
        hapsolo.myMinQRPctMin = 0.0

    def tearDown(self):
        if os.path.exists(self.workdir):
            shutil.rmtree(self.workdir)

    def test_paf_with_12_fields(self):
        """PAF with exactly 12 fields (no optional tags) should work."""
        paf = os.path.join(self.workdir, 'min.paf')
        with open(paf, 'w') as f:
            # 12 fields: qName qLen qStart qEnd strand tName tLen tStart tEnd matches blockLen mapQ
            f.write("contigA\t5000\t0\t4000\t+\tcontigB\t6000\t100\t4100\t3900\t4000\t60\n")
        result = hapsolo.CreateMM2AlignmentDataStructure(paf)
        self.assertEqual(len(result), 1)

    def test_paf_with_11_fields_accepted(self):
        """PAF with 11 fields should be accepted (code checks < 11)."""
        paf = os.path.join(self.workdir, 'eleven.paf')
        with open(paf, 'w') as f:
            f.write("contigA\t5000\t0\t4000\t+\tcontigB\t6000\t100\t4100\t3900\t4000\n")
        result = hapsolo.CreateMM2AlignmentDataStructure(paf)
        self.assertEqual(len(result), 1)


# ── ReduceASM edge cases ────────────────────────────────────────────────────

class TestReduceASMEdgeCases(unittest.TestCase):

    def test_empty_dataframe(self):
        """ReduceASM with no alignment data should keep all contigs."""
        hapsolo.mypddf = pd.DataFrame(
            columns=['qName', 'tName', 'qSize', 'QPct', 'PID', 'QRAlignLenPct'])
        hapsolo.allcontigsset = {'contigA', 'contigB'}
        result = hapsolo.ReduceASM(0.5, 0.5, 0.2)
        self.assertEqual(result, {'contigA', 'contigB'})

    def test_all_contigs_filtered(self):
        """When all contigs have high-quality alignments, good set should be small."""
        hapsolo.mypddf = pd.DataFrame({
            'qName': ['contigA', 'contigB'],
            'tName': ['contigB', 'contigA'],
            'qSize': [5000, 5000],
            'QPct': [0.95, 0.95],
            'PID': [0.95, 0.95],
            'QRAlignLenPct': [0.95, 0.95],
        })
        hapsolo.allcontigsset = {'contigA', 'contigB'}
        result = hapsolo.ReduceASM(0.5, 0.5, 0.2)
        # Both contigs appear as qName with passing thresholds -> both removed
        self.assertEqual(len(result), 0)

    def test_qralignlenpct_upper_bound(self):
        """QRAlignLenPct above the inverse proportion should be filtered out."""
        # CalculateInverseProportion(0.3) = exp(-log2(0.3)) ≈ 5.68
        # QRAlignLenPct=6.0 exceeds this upper bound -> not "bad"
        hapsolo.mypddf = pd.DataFrame({
            'qName': ['contigA'],
            'tName': ['contigB'],
            'qSize': [5000],
            'QPct': [0.8],
            'PID': [0.9],
            'QRAlignLenPct': [6.0],
        })
        hapsolo.allcontigsset = {'contigA', 'contigB'}
        result = hapsolo.ReduceASM(0.5, 0.5, 0.3)
        # QRAlignLenPct=6.0 > inv(0.3)≈5.68, so this alignment is
        # filtered OUT of the "bad" set -> contigA stays "good"
        self.assertIn('contigA', result)


# ── calculateBuscos edge cases ──────────────────────────────────────────────

class TestCalculateBuscosEdgeCases(unittest.TestCase):

    def test_all_missing_buscos(self):
        """When all BUSCOs are Missing on every contig, S=D=F=0, M=total."""
        busco_dir = os.path.join(FIXTURES_DIR, 'busco_scoring')
        b2c, c2b = hapsolo.importBuscos(busco_dir)
        hapsolo.missingrefcontigset = set()
        hapsolo.smallcontigset = set()

        # Pass a contig that has NO Complete or Fragmented BUSCOs
        # Use empty list -> no contigs considered
        result = hapsolo.calculateBuscos([], b2c, c2b)
        self.assertEqual(result['S'], 0)
        self.assertEqual(result['D'], 0)
        self.assertEqual(result['F'], 0)
        self.assertEqual(result['M'], len(b2c))

    def test_single_contig_no_duplicates(self):
        """A single contig can never produce duplicate BUSCOs."""
        busco_dir = os.path.join(FIXTURES_DIR, 'busco_scoring')
        b2c, c2b = hapsolo.importBuscos(busco_dir)
        hapsolo.missingrefcontigset = set()
        hapsolo.smallcontigset = set()

        for contig in c2b:
            result = hapsolo.calculateBuscos([contig], b2c, c2b)
            self.assertEqual(result['D'], 0,
                             contig + ' alone should have 0 duplicates but got '
                             + str(result['D']))

    def test_busco_counts_add_up(self):
        """S + D + F + M must equal total BUSCOs for any contig set."""
        busco_dir = os.path.join(FIXTURES_DIR, 'busco_scoring')
        b2c, c2b = hapsolo.importBuscos(busco_dir)
        hapsolo.missingrefcontigset = set()
        hapsolo.smallcontigset = set()
        total = len(b2c)

        for contigs in [['contigA'], ['contigB'], ['contigA', 'contigC'],
                        ['contigA', 'contigB', 'contigC'], []]:
            result = hapsolo.calculateBuscos(contigs, b2c, c2b)
            got_total = result['S'] + result['D'] + result['F'] + result['M']
            self.assertEqual(got_total, total,
                             'Counts do not add up for ' + str(contigs)
                             + ': S=' + str(result['S']) + ' D=' + str(result['D'])
                             + ' F=' + str(result['F']) + ' M=' + str(result['M'])
                             + ' total=' + str(got_total) + ' expected=' + str(total))


# ── hillclimbing edge cases ─────────────────────────────────────────────────

class TestHillClimbingEdgeCases(unittest.TestCase):

    def _setup_minimal_globals(self):
        """Set up minimal globals for hillclimbing to run."""
        hapsolo.mypddf = pd.DataFrame({
            'qName': ['cA', 'cA'],
            'tName': ['cB', 'cC'],
            'qSize': [5000, 5000],
            'QPct': [0.8, 0.3],
            'PID': [0.9, 0.4],
            'QRAlignLenPct': [0.95, 0.5],
        })
        hapsolo.myContigsDict = {
            'cA': [5000, 0, 0, 0], 'cB': [6000, 0, 0, 0], 'cC': [4000, 0, 0, 0]
        }
        hapsolo.allcontigsset = {'cA', 'cB', 'cC'}
        hapsolo.qrycontigset = {'cA'}
        hapsolo.missingrefcontigset = {'cB', 'cC'}
        hapsolo.smallcontigset = set()
        # Set up BUSCO data
        hapsolo.busco2contigdict = {
            'B1': {'C': ['cA'], 'S': [], 'D': [], 'F': [], 'M': []},
            'B2': {'C': ['cB'], 'S': [], 'D': [], 'F': [], 'M': []},
            'B3': {'C': [], 'S': [], 'D': [], 'F': ['cC'], 'M': []},
        }
        hapsolo.contigs2buscodict = {
            'cA': {'C': ['B1'], 'S': [], 'D': [], 'F': [], 'M': []},
            'cB': {'C': ['B2'], 'S': [], 'D': [], 'F': [], 'M': []},
            'cC': {'C': [], 'S': [], 'D': [], 'F': ['B3'], 'M': []},
        }

    def test_single_iteration(self):
        """hillclimbing with numofiterations=1 should complete without error."""
        from random import seed
        seed(99)
        self._setup_minimal_globals()

        job_args = [0, 1, 0.0001, 0.5, 0.5, 0.5]
        result = hapsolo.hillclimbing(job_args)

        self.assertEqual(len(result), 3)
        self.assertGreater(len(result[0]), 0)
        self.assertEqual(len(result[1]), 1)  # costfxn has 1 entry
        self.assertEqual(len(result[2]), 1)  # costfxndelta has 1 entry

    def test_mode1_returns_immediately(self):
        """Mode 1 should return after initial evaluation (no random walk)."""
        from random import seed
        seed(99)
        self._setup_minimal_globals()
        hapsolo.mode = 1

        job_args = [0, 100, 0.0001, 0.7, 0.7, 0.7]
        result = hapsolo.hillclimbing(job_args)

        # Mode 1 returns at line 406 after first ReduceASM+calculateBuscos
        self.assertEqual(len(result), 3)
        bestnscoreslist = result[0]
        self.assertGreater(len(bestnscoreslist), 0)

    def test_zero_single_buscos_high_score(self):
        """When S=0, hillclimbing should assign a high penalty score (5000.0)."""
        from random import seed
        seed(99)
        # All BUSCOs are missing -> S=0 after filtering
        hapsolo.mypddf = pd.DataFrame({
            'qName': ['cA'],
            'tName': ['cB'],
            'qSize': [5000],
            'QPct': [0.8],
            'PID': [0.9],
            'QRAlignLenPct': [0.95],
        })
        hapsolo.myContigsDict = {'cA': [5000, 0, 0, 0], 'cB': [6000, 0, 0, 0]}
        hapsolo.allcontigsset = {'cA', 'cB'}
        hapsolo.qrycontigset = {'cA'}
        hapsolo.missingrefcontigset = {'cB'}
        hapsolo.smallcontigset = set()
        # No BUSCOs at all
        hapsolo.busco2contigdict = {'B1': {'C': [], 'S': [], 'D': [], 'F': [], 'M': []}}
        hapsolo.contigs2buscodict = {
            'cA': {'C': [], 'S': [], 'D': [], 'F': [], 'M': []},
            'cB': {'C': [], 'S': [], 'D': [], 'F': [], 'M': []},
        }
        hapsolo.mode = 1

        job_args = [0, 1, 0.0001, 0.7, 0.7, 0.7]
        result = hapsolo.hillclimbing(job_args)
        # Score should be the penalty value (50000000.0) since S=0
        self.assertEqual(result[0][0][0], 50000000.0)


# ── calculateasmstats edge cases ────────────────────────────────────────────

class TestCalculateAsmStatsEdgeCases(unittest.TestCase):

    def test_empty_set_returns_zeros(self):
        """Empty contig set should return all zeros."""
        hapsolo.myContigsDict = {'c1': [1000, 0, 0, 0]}
        asmsize, n50, l50, largest = hapsolo.calculateasmstats(set())
        self.assertEqual(asmsize, 0)
        self.assertEqual(n50, 0)
        self.assertEqual(l50, 0)
        self.assertEqual(largest, 0)

    def test_all_same_size(self):
        """Contigs all the same size: N50 = that size."""
        hapsolo.myContigsDict = {
            'c1': [1000, 0, 0, 0],
            'c2': [1000, 0, 0, 0],
            'c3': [1000, 0, 0, 0],
            'c4': [1000, 0, 0, 0],
        }
        # total=4000, 50%=2000. Sorted: [1000,1000,1000,1000]
        # Cumulative: 1000 (not >2000), 2000 (not >2000), 3000 (>2000)
        # N50=1000, L50=3 (need 3 contigs to EXCEED 50%)
        asmsize, n50, l50, largest = hapsolo.calculateasmstats({'c1', 'c2', 'c3', 'c4'})
        self.assertEqual(asmsize, 4000)
        self.assertEqual(n50, 1000)
        self.assertEqual(l50, 3)
        self.assertEqual(largest, 1000)

    def test_contigs_not_in_dict_skipped(self):
        """Contigs not in myContigsDict should be silently ignored."""
        hapsolo.myContigsDict = {'c1': [5000, 0, 0, 0]}
        asmsize, n50, l50, largest = hapsolo.calculateasmstats({'c1', 'ghost'})
        self.assertEqual(asmsize, 5000)


# ── myLinearFxn edge cases ───────────────────────────────────────────────────

class TestMyLinearFxnEdgeCases(unittest.TestCase):

    def test_sbusco_zero_guarded_by_caller(self):
        """hillclimbing guards sbusco=0 with a penalty score, not myLinearFxn."""
        # myLinearFxn itself does NOT guard against sbusco=0;
        # callers (hillclimbing lines 370-373, 397-399, 470-472) check first.
        # Verify the caller's guard produces the expected penalty.
        from random import seed
        seed(99)

        hapsolo.mypddf = pd.DataFrame({
            'qName': ['cA'], 'tName': ['cB'],
            'qSize': [5000], 'QPct': [0.8], 'PID': [0.9], 'QRAlignLenPct': [0.95],
        })
        hapsolo.myContigsDict = {'cA': [5000, 0, 0, 0], 'cB': [6000, 0, 0, 0]}
        hapsolo.allcontigsset = {'cA', 'cB'}
        hapsolo.qrycontigset = {'cA'}
        hapsolo.missingrefcontigset = {'cB'}
        hapsolo.smallcontigset = set()
        # No complete BUSCOs -> S will be 0
        hapsolo.busco2contigdict = {'B1': {'C': [], 'S': [], 'D': [], 'F': [], 'M': []}}
        hapsolo.contigs2buscodict = {
            'cA': {'C': [], 'S': [], 'D': [], 'F': [], 'M': []},
            'cB': {'C': [], 'S': [], 'D': [], 'F': [], 'M': []},
        }
        hapsolo.mode = 1

        job_args = [0, 1, 0.0001, 0.7, 0.7, 0.7]
        # Should NOT crash — hillclimbing returns penalty score when S=0
        result = hapsolo.hillclimbing(job_args)
        self.assertEqual(result[0][0][0], 50000000.0)

    def test_thetaS_zero_rejected(self):
        """thetaS=0 should be rejected at parse time (would cause division by zero)."""
        # After our fix, thetaS=0 is reset to 1.0 with a warning.
        # Since thetaS is set at module load from argparse, we can't re-parse.
        # Instead, verify that myLinearFxn crashes if thetaS is somehow 0.
        saved = hapsolo.thetaS
        try:
            hapsolo.thetaS = 0.0
            with self.assertRaises(ZeroDivisionError):
                hapsolo.myLinearFxn(5, 10, 3, 2, 15)
        finally:
            hapsolo.thetaS = saved

    def test_custom_theta_weights(self):
        """Cost function should respect custom theta weights."""
        saved_s = hapsolo.thetaS
        saved_d = hapsolo.thetaD
        saved_m = hapsolo.thetaM
        saved_f = hapsolo.thetaF
        try:
            hapsolo.thetaS = 2.0
            hapsolo.thetaD = 0.5
            hapsolo.thetaM = 0.5
            hapsolo.thetaF = 0.5
            # (0.5*F + 0.5*D + 0.5*M) / (2.0*S)
            # (0.5*2 + 0.5*3 + 0.5*5) / (2.0*10) = (1+1.5+2.5)/20 = 5/20 = 0.25
            result = hapsolo.myLinearFxn(5, 10, 3, 2, 15)
            self.assertAlmostEqual(result, 0.25)
        finally:
            hapsolo.thetaS = saved_s
            hapsolo.thetaD = saved_d
            hapsolo.thetaM = saved_m
            hapsolo.thetaF = saved_f


# ── uniquepriorityqueue with bestnscores edge cases ─────────────────────────

class TestBestNScoresEdgeCases(unittest.TestCase):

    def test_fewer_results_than_bestnscores(self):
        """uniquepriorityqueue should return fewer items than bestnscores if list is short."""
        saved = hapsolo.bestnscores
        try:
            hapsolo.bestnscores = 5
            pqlist = [[0.5, {'c1'}, {'c2'}, {}, [0.7, 0.7, 0.7]]]
            new_val = [0.3, {'c1', 'c2'}, set(), {}, [0.8, 0.8, 0.8]]
            result = hapsolo.uniquepriorityqueue(pqlist, new_val)
            # Only 2 entries exist, bestnscores=5 -> should return 2, not crash
            self.assertEqual(len(result), 2)
        finally:
            hapsolo.bestnscores = saved

    def test_bestnscores_one_with_duplicates(self):
        """With bestnscores=1, duplicate removal should still return exactly 1."""
        saved = hapsolo.bestnscores
        try:
            hapsolo.bestnscores = 1
            pqlist = [[0.5, {'c1', 'c2'}, {'c3'}, {}, [0.7, 0.7, 0.7]]]
            # Add duplicate contig set with different score
            new_val = [0.3, {'c1', 'c2'}, {'c3'}, {}, [0.8, 0.8, 0.8]]
            result = hapsolo.uniquepriorityqueue(pqlist, new_val)
            self.assertEqual(len(result), 1)
            # Should keep the lower score
            self.assertAlmostEqual(result[0][0], 0.3)
        finally:
            hapsolo.bestnscores = saved


if __name__ == '__main__':
    unittest.main()
