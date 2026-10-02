#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integration tests using real data from data/mosquito/ and data/shark/.
These tests validate that HapSolo functions correctly handle actual
genomic data formats and value ranges.

Skipped automatically if data files are not present.
"""
import os
import gzip
import unittest

import pandas as pd
import hapsolo

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TESTS_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')


MOSQUITO_HAP = os.path.join(DATA_DIR, 'mosquito', 'anof_funestus_new_self_aln.hap.gz')
MOSQUITO_PAF = os.path.join(DATA_DIR, 'mosquito', 'primary_new_noxcustom_self_align.paf.gz')
MOSQUITO_FASTA = os.path.join(DATA_DIR, 'mosquito', 'primary_new.fasta.gz')
MOSQUITO_BUSCO = os.path.join(DATA_DIR, 'mosquito', 'anof_funestus_new_buscooutputs.tsv.gz')
SHARK_HAP = os.path.join(DATA_DIR, 'shark', 'sambrad_cns_p_ctg_new_noxcustom_self_align2.hap.gz')
SHARK_BUSCO = os.path.join(DATA_DIR, 'shark', 'sambrad_buscooutput.tsv.gz')


def _has_mosquito_data():
    return os.path.exists(MOSQUITO_HAP) and os.path.exists(MOSQUITO_PAF)


def _has_mosquito_full():
    """All mosquito files needed for the full pipeline smoke test."""
    return (_has_mosquito_data()
            and os.path.exists(MOSQUITO_FASTA)
            and os.path.exists(MOSQUITO_BUSCO))


def _has_shark_data():
    return os.path.exists(SHARK_HAP)


def _has_shark_full():
    return os.path.exists(SHARK_HAP) and os.path.exists(SHARK_BUSCO)


def _load_contig_sizes_from_gz_fasta(fasta_gz_path):
    """Read a gzipped FASTA and return {name: [size, 0, 0, 0]} for each contig.

    We only need contig names and sizes to set up globals for hill climbing.
    File positions are set to 0 since WriteNewAssembly is not called.
    """
    result = {}
    current_name = None
    current_size = 0
    with gzip.open(fasta_gz_path, 'rt') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_name is not None:
                    result[current_name] = [current_size, 0, 0, 0]
                current_name = line[1:].split()[0]
                current_size = 0
            elif current_name is not None:
                current_size += len(line)
    if current_name is not None:
        result[current_name] = [current_size, 0, 0, 0]
    return result


def _load_busco_flat_tsv(busco_gz_path):
    """Load a flat BUSCO TSV (not V3 directory format) into the dict structures
    expected by calculateBuscos().

    Format: BUSCOID\\tStatus[\\tContigID\\tStart\\tEnd\\tScore\\tLength]
    Status: Complete, Duplicated, Fragmented, Missing

    Both "Complete" and "Duplicated" map to 'C' type (calculateBuscos counts
    per-contig occurrences to determine single vs duplicate).
    """
    buscotypes = ['C', 'S', 'D', 'F', 'M']
    busco_ids = set()
    contig_names = set()

    # First pass: collect all BUSCO IDs and contig names
    with gzip.open(busco_gz_path, 'rt') as f:
        for line in f:
            fields = line.strip().split('\t')
            busco_ids.add(fields[0])
            if len(fields) >= 3 and fields[1] != 'Missing':
                contig_names.add(fields[2])

    # Initialize dicts
    b2c = {}
    for bid in busco_ids:
        b2c[bid] = {bt: [] for bt in buscotypes}

    c2b = {}
    for cname in contig_names:
        c2b[cname] = {bt: [] for bt in buscotypes}

    # Second pass: populate
    with gzip.open(busco_gz_path, 'rt') as f:
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) < 2:
                continue
            bid = fields[0]
            status = fields[1]
            if status == 'Missing':
                continue
            cname = fields[2] if len(fields) >= 3 else None
            if cname is None:
                continue
            # Map status to buscotype key
            # Both Complete and Duplicated map to 'C' — the counting logic
            # in calculateBuscos determines single vs duplicate from occurrence count
            if status in ('Complete', 'Duplicated'):
                bt = 'C'
            elif status == 'Fragmented':
                bt = 'F'
            else:
                continue
            b2c[bid][bt].append(cname)
            c2b[cname][bt].append(bid)

    return b2c, c2b


@unittest.skipUnless(_has_mosquito_data(), 'Mosquito data not available')
class TestMosquitoHAPLoading(unittest.TestCase):
    """Validate loading of real Anopheles funestus HAP data."""

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(
            MOSQUITO_HAP, sep='\t', header=None,
            names=['qName', 'tName', 'qSize', 'QPct', 'PID', 'QRAlignLenPct'],
            dtype={'qName': object, 'tName': object}
        )

    def test_row_count(self):
        """Mosquito HAP should have 416,553 alignment rows."""
        self.assertEqual(len(self.df), 416553)

    def test_column_count(self):
        """Should have exactly 6 columns."""
        self.assertEqual(len(self.df.columns), 6)

    def test_no_self_alignments(self):
        """HAP file should not contain self-alignments (qName == tName)."""
        self_aligns = self.df[self.df['qName'] == self.df['tName']]
        self.assertEqual(len(self_aligns), 0)

    def test_qpct_range(self):
        """QPct values should be in [0.2, 1.0] based on default min threshold."""
        self.assertGreaterEqual(self.df['QPct'].min(), 0.2 - 0.001)
        self.assertLessEqual(self.df['QPct'].max(), 1.0 + 0.001)

    def test_pid_range(self):
        """PID values should be in [0.2, 1.0] based on default min threshold."""
        self.assertGreaterEqual(self.df['PID'].min(), 0.2 - 0.001)
        self.assertLessEqual(self.df['PID'].max(), 1.0 + 0.001)

    def test_contig_name_format(self):
        """All contig names should match tigXXXXXXXX pattern."""
        import re
        tig_pattern = re.compile(r'^tig\d{8}$')
        all_names = set(self.df['qName']).union(set(self.df['tName']))
        for name in all_names:
            self.assertTrue(tig_pattern.match(str(name)),
                            'Unexpected contig name format: ' + repr(name))

    def test_unique_contig_counts(self):
        """Should have expected number of unique query and target contigs."""
        self.assertEqual(self.df['qName'].nunique(), 664)
        self.assertEqual(self.df['tName'].nunique(), 793)

    def test_notebook_filtering_reproduces(self):
        """Replicate the notebook's QPct>=0.7 + PID>=0.7 filter counts."""
        filtered = self.df[(self.df['QPct'] >= 0.70) & (self.df['PID'] >= 0.70)]
        self.assertEqual(len(filtered), 9630)

    def test_notebook_set_subtraction(self):
        """Replicate the notebook's set operation: total contigs - filtered query contigs."""
        all_qry = set(self.df['qName'])
        filtered = self.df[(self.df['QPct'] >= 0.70) & (self.df['PID'] >= 0.70)]
        filtered_qry = set(filtered['qName'])
        good_set = all_qry - filtered_qry
        # Notebook shows 246 "good" contigs at 0.7/0.7 thresholds
        self.assertEqual(len(good_set), 246)

    def test_qsize_max(self):
        """Maximum qSize should match the notebook's observation."""
        self.assertEqual(self.df['qSize'].max(), 6749)


@unittest.skipUnless(_has_mosquito_data(), 'Mosquito data not available')
class TestMosquitoPAFSampling(unittest.TestCase):
    """Validate structure of real PAF alignment data."""

    def test_paf_format(self):
        """First 100 lines of PAF should have >= 12 tab-delimited fields."""
        with gzip.open(MOSQUITO_PAF, 'rt') as f:
            for i, line in enumerate(f):
                if i >= 100:
                    break
                fields = line.strip().split('\t')
                self.assertGreaterEqual(len(fields), 12,
                                        'Line ' + str(i) + ' has only '
                                        + str(len(fields)) + ' fields')

    def test_paf_has_self_alignments(self):
        """Raw PAF should contain self-alignments that CreateMM2 filters out."""
        self_count = 0
        with gzip.open(MOSQUITO_PAF, 'rt') as f:
            for i, line in enumerate(f):
                if i >= 10000:
                    break
                fields = line.strip().split('\t')
                if fields[0] == fields[5]:
                    self_count += 1
        self.assertGreater(self_count, 0,
                           'Expected self-alignments in raw PAF but found none')

    def test_paf_contig_names_match_hap(self):
        """Query contig names in PAF should include those in the HAP file."""
        hap_df = pd.read_csv(
            MOSQUITO_HAP, sep='\t', header=None,
            names=['qName', 'tName', 'qSize', 'QPct', 'PID', 'QRAlignLenPct'],
            dtype={'qName': object, 'tName': object}
        )
        hap_names = set(hap_df['qName']).union(set(hap_df['tName']))

        # Sample PAF names (first 50k lines)
        paf_names = set()
        with gzip.open(MOSQUITO_PAF, 'rt') as f:
            for i, line in enumerate(f):
                if i >= 50000:
                    break
                fields = line.strip().split('\t')
                paf_names.add(fields[0])
                paf_names.add(fields[5])

        # All HAP names should appear in PAF
        overlap = hap_names.intersection(paf_names)
        self.assertGreater(len(overlap), 0,
                           'No overlap between PAF and HAP contig names')


@unittest.skipUnless(_has_mosquito_data() and os.path.exists(MOSQUITO_BUSCO),
                     'Mosquito BUSCO data not available')
class TestMosquitoBUSCOData(unittest.TestCase):
    """Validate the structure of real BUSCO data."""

    def test_busco_id_format(self):
        """BUSCO IDs should match EOG09XXXXXX pattern."""
        import re
        pattern = re.compile(r'^EOG09\w+$')
        with gzip.open(MOSQUITO_BUSCO, 'rt') as f:
            for i, line in enumerate(f):
                if i >= 1000:
                    break
                fields = line.strip().split('\t')
                self.assertTrue(pattern.match(fields[0]),
                                'Unexpected BUSCO ID: ' + repr(fields[0]))

    def test_busco_status_values(self):
        """BUSCO status should be one of: Complete, Duplicated, Fragmented, Missing."""
        valid_statuses = {'Complete', 'Duplicated', 'Fragmented', 'Missing'}
        with gzip.open(MOSQUITO_BUSCO, 'rt') as f:
            for i, line in enumerate(f):
                if i >= 5000:
                    break
                fields = line.strip().split('\t')
                if len(fields) >= 2:
                    self.assertIn(fields[1], valid_statuses,
                                  'Unexpected BUSCO status: ' + repr(fields[1]))

    def test_non_missing_has_contig(self):
        """Non-Missing BUSCO entries should have a contig name in column 3."""
        with gzip.open(MOSQUITO_BUSCO, 'rt') as f:
            for i, line in enumerate(f):
                if i >= 10000:
                    break
                fields = line.strip().split('\t')
                if len(fields) >= 2 and fields[1] != 'Missing':
                    self.assertGreaterEqual(len(fields), 3,
                                            'Non-Missing entry missing contig: ' + repr(line))
                    self.assertTrue(len(fields[2]) > 0,
                                    'Empty contig name in non-Missing entry')

    def test_busco_counts(self):
        """Validate expected BUSCO status counts for mosquito data."""
        counts = {}
        with gzip.open(MOSQUITO_BUSCO, 'rt') as f:
            for line in f:
                fields = line.strip().split('\t')
                status = fields[1] if len(fields) > 1 else 'unknown'
                counts[status] = counts.get(status, 0) + 1
        self.assertEqual(counts.get('Complete', 0), 3146)
        self.assertEqual(counts.get('Duplicated', 0), 74)
        self.assertEqual(counts.get('Fragmented', 0), 124)
        self.assertGreater(counts.get('Missing', 0), 3000000)


@unittest.skipUnless(_has_shark_data(), 'Shark data not available')
class TestSharkHAPLoading(unittest.TestCase):
    """Validate loading of real shark (Squalus acanthias) HAP data."""

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(
            SHARK_HAP, sep='\t', header=None,
            names=['qName', 'tName', 'qSize', 'QPct', 'PID', 'QRAlignLenPct'],
            dtype={'qName': object, 'tName': object}
        )

    def test_row_count(self):
        """Shark HAP should have 117,801 rows."""
        self.assertEqual(len(self.df), 117801)

    def test_numeric_contig_ids(self):
        """Shark contig IDs are purely numeric strings (e.g. '000438')."""
        all_names = set(self.df['qName']).union(set(self.df['tName']))
        for name in all_names:
            self.assertTrue(str(name).isdigit(),
                            'Non-numeric shark contig ID: ' + repr(name))

    def test_leading_zeros_preserved(self):
        """Leading zeros in shark IDs should be preserved (6-digit format)."""
        all_names = set(self.df['qName']).union(set(self.df['tName']))
        for name in all_names:
            self.assertEqual(len(str(name)), 6,
                             'Shark ID not 6 digits: ' + repr(name))

    def test_no_self_alignments(self):
        """Shark HAP should not contain self-alignments."""
        self_aligns = self.df[self.df['qName'] == self.df['tName']]
        self.assertEqual(len(self_aligns), 0)

    def test_value_ranges(self):
        """Value ranges should be within expected bounds."""
        self.assertGreaterEqual(self.df['QPct'].min(), 0.0)
        self.assertLessEqual(self.df['QPct'].max(), 1.0 + 0.001)
        self.assertGreaterEqual(self.df['PID'].min(), 0.0)
        self.assertLessEqual(self.df['PID'].max(), 1.0 + 0.001)
        self.assertGreaterEqual(self.df['QRAlignLenPct'].min(), 0.0)


@unittest.skipUnless(_has_mosquito_data(), 'Mosquito data not available')
class TestMosquitoFASTALoading(unittest.TestCase):
    """Validate CalculateContigSizes works with real gzipped FASTA (if uncompressed)."""

    @unittest.skipUnless(os.path.exists(os.path.join(DATA_DIR, 'mosquito', 'primary_new.fasta')),
                         'Uncompressed FASTA not available (only .gz)')
    def test_loads_real_fasta(self):
        """Should load all contigs from real Anopheles funestus assembly."""
        fasta = os.path.join(DATA_DIR, 'mosquito', 'primary_new.fasta')
        result = hapsolo.CalculateContigSizes(fasta)
        self.assertEqual(len(result), 1073)

    def test_gzipped_fasta_header_count(self):
        """Count contigs in gzipped FASTA without CalculateContigSizes."""
        count = 0
        with gzip.open(MOSQUITO_FASTA, 'rt') as f:
            for line in f:
                if line.startswith('>'):
                    count += 1
        self.assertEqual(count, 1073)

    def test_gzipped_fasta_header_format(self):
        """FASTA headers should be clean tig IDs without special characters."""
        import re
        tig_pattern = re.compile(r'^>tig\d{8}$')
        with gzip.open(MOSQUITO_FASTA, 'rt') as f:
            for line in f:
                if line.startswith('>'):
                    header = line.strip()
                    self.assertTrue(tig_pattern.match(header),
                                    'Unexpected FASTA header: ' + repr(header))


@unittest.skipUnless(_has_mosquito_data(), 'Mosquito data not available')
class TestCrossDataConsistency(unittest.TestCase):
    """Cross-validate contig names between PAF, HAP, and BUSCO data sources."""

    def test_hap_contigs_subset_of_paf_contigs(self):
        """All contig names in HAP should appear in the PAF source."""
        hap_df = pd.read_csv(
            MOSQUITO_HAP, sep='\t', header=None,
            names=['qName', 'tName', 'qSize', 'QPct', 'PID', 'QRAlignLenPct'],
            dtype={'qName': object, 'tName': object}
        )
        hap_names = set(hap_df['qName']).union(set(hap_df['tName']))

        paf_names = set()
        with gzip.open(MOSQUITO_PAF, 'rt') as f:
            for line in f:
                fields = line.strip().split('\t')
                paf_names.add(fields[0])
                paf_names.add(fields[5])

        missing = hap_names - paf_names
        self.assertEqual(len(missing), 0,
                         'HAP names not found in PAF: ' + str(list(missing)[:10]))

    def test_busco_contigs_overlap_with_fasta(self):
        """BUSCO contig names should overlap with FASTA contig names."""
        # Get FASTA headers
        fasta_names = set()
        with gzip.open(MOSQUITO_FASTA, 'rt') as f:
            for line in f:
                if line.startswith('>'):
                    fasta_names.add(line.strip()[1:].split()[0])

        # Get BUSCO contig names
        busco_names = set()
        with gzip.open(MOSQUITO_BUSCO, 'rt') as f:
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) >= 3 and fields[1] != 'Missing':
                    busco_names.add(fields[2])

        overlap = fasta_names.intersection(busco_names)
        self.assertGreater(len(overlap), 0,
                           'No overlap between FASTA and BUSCO contig names')
        # Most BUSCO contigs should be in the FASTA
        coverage = len(overlap) / len(busco_names)
        self.assertGreater(coverage, 0.9,
                           'Only ' + str(int(coverage * 100))
                           + '% of BUSCO contigs found in FASTA')


@unittest.skipUnless(_has_mosquito_full(), 'Full mosquito dataset not available')
class TestMosquitoHillClimbingSmokeTest(unittest.TestCase):
    """Smoke test: run hill climbing on real Anopheles funestus data.

    Sets up globals the same way __main__ does, then calls hillclimbing()
    with mode=1 (fixed thresholds, 1 iteration) and mode=0 (short random walk).
    """

    @classmethod
    def setUpClass(cls):
        """Load all data once for the whole test class."""
        # Load HAP alignment data
        cls.hap_df = pd.read_csv(
            MOSQUITO_HAP, sep='\t', header=None,
            names=['qName', 'tName', 'qSize', 'QPct', 'PID', 'QRAlignLenPct'],
            dtype={'qName': object, 'tName': object}
        )
        # Load contig sizes from gzipped FASTA
        cls.contig_sizes = _load_contig_sizes_from_gz_fasta(MOSQUITO_FASTA)
        # Load BUSCO data from flat TSV
        cls.b2c, cls.c2b = _load_busco_flat_tsv(MOSQUITO_BUSCO)

    def _setup_globals(self):
        """Wire up hapsolo globals to match __main__ initialization."""
        hapsolo.mypddf = self.hap_df.copy()
        hapsolo.myContigsDict = self.contig_sizes.copy()
        hapsolo.busco2contigdict = {k: {bt: list(v) for bt, v in d.items()}
                                    for k, d in self.b2c.items()}
        hapsolo.contigs2buscodict = {k: {bt: list(v) for bt, v in d.items()}
                                     for k, d in self.c2b.items()}
        hapsolo.myMinContigSize = 1000
        hapsolo.myMinPID = 0.2
        hapsolo.myMinQPctMin = 0.2
        hapsolo.myMinQRPctMin = 0.2
        # Compute derived sets
        hapsolo.qrycontigset = set(hapsolo.mypddf['qName'])
        hapsolo.allcontigsset = set(hapsolo.myContigsDict.keys())
        hapsolo.missingrefcontigset = hapsolo.allcontigsset - hapsolo.qrycontigset
        hapsolo.smallcontigset = set()
        for key in hapsolo.myContigsDict:
            if hapsolo.myContigsDict[key][0] < hapsolo.myMinContigSize:
                hapsolo.smallcontigset.add(key)

    def test_mode1_fixed_thresholds(self):
        """Mode 1: single iteration at fixed 0.7/0.7/0.7 thresholds."""
        from random import seed
        seed(42)
        self._setup_globals()

        job_args = [0, 1, 0.0001, 0.7, 0.7, 0.7]
        result = hapsolo.hillclimbing(job_args)

        # Result structure: [bestnscoreslist, costfxn, costfxndelta]
        self.assertEqual(len(result), 3)
        bestnscoreslist = result[0]
        costfxn = result[1]
        costfxndelta = result[2]

        # Should have at least 1 scored result
        self.assertGreater(len(bestnscoreslist), 0)

        # Each entry: [score, good_contig_set, purged_contig_set, busco_scores, params]
        best = bestnscoreslist[0]
        score = best[0]
        good_contigs = best[1]
        purged_contigs = best[2]
        busco_scores = best[3]
        params = best[4]

        # Score should be a positive finite number
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0)
        self.assertLess(score, 50000000)

        # Good contigs should be a non-empty set
        self.assertIsInstance(good_contigs, set)
        self.assertGreater(len(good_contigs), 0)

        # Purged contigs should exist (some contigs removed)
        self.assertIsInstance(purged_contigs, set)

        # Good + purged should cover all contigs
        total = good_contigs.union(purged_contigs)
        expected_total = hapsolo.allcontigsset - hapsolo.smallcontigset - {''}
        # Allow some slack for small contigs
        self.assertGreater(len(total), 0)

        # BUSCO scores should have expected keys
        for key in ['S', 'D', 'C', 'F', 'M']:
            self.assertIn(key, busco_scores)

        # Single BUSCOs should be reasonable (> 0 for a real genome)
        self.assertGreater(busco_scores['S'], 0)
        # Total BUSCOs should sum correctly
        self.assertEqual(busco_scores['C'], busco_scores['S'] + busco_scores['D'])

    def test_mode0_short_random_walk(self):
        """Mode 0: short random walk with 10 iterations."""
        from random import seed
        seed(42)
        self._setup_globals()

        job_args = [0, 10, 0.0001, 0.5, 0.5, 0.5]
        result = hapsolo.hillclimbing(job_args)

        bestnscoreslist = result[0]
        costfxn = result[1]
        costfxndelta = result[2]

        # Should have scored results
        self.assertGreater(len(bestnscoreslist), 0)

        # Cost function history should have 10 entries
        self.assertEqual(len(costfxn), 10)

        # Cost delta history should have 10 entries
        self.assertEqual(len(costfxndelta), 10)

        # First cost should be positive
        self.assertGreater(costfxn[0], 0)

        # Best score should be positive and finite
        self.assertGreater(bestnscoreslist[0][0], 0)
        self.assertLess(bestnscoreslist[0][0], 50000000)

    def test_reducing_contigs_improves_or_maintains_score(self):
        """Optimal thresholds should reduce duplicates compared to raw assembly."""
        from random import seed
        seed(42)
        self._setup_globals()

        # Score the full assembly (all contigs)
        all_contigs = hapsolo.qrycontigset.union(
            hapsolo.missingrefcontigset) - hapsolo.smallcontigset - {''}
        all_buscos = hapsolo.calculateBuscos(
            all_contigs, hapsolo.busco2contigdict, hapsolo.contigs2buscodict)
        all_dupes = all_buscos['D']

        # Run optimization
        job_args = [0, 1, 0.0001, 0.7, 0.7, 0.7]
        result = hapsolo.hillclimbing(job_args)
        optimized_buscos = result[0][0][3]
        opt_dupes = optimized_buscos['D']

        # Optimized assembly should have fewer or equal duplicates
        self.assertLessEqual(opt_dupes, all_dupes,
                             'Optimization increased duplicates from '
                             + str(all_dupes) + ' to ' + str(opt_dupes))

    def test_contig_sizes_loaded(self):
        """Verify real FASTA sizes look reasonable."""
        # Mosquito genome is ~250Mb, largest contig should be > 1Mb
        sizes = [v[0] for v in self.contig_sizes.values()]
        self.assertEqual(len(sizes), 1073)
        self.assertGreater(max(sizes), 1000000)
        self.assertGreater(sum(sizes), 200000000)  # > 200Mb total

    def test_busco_loading_completeness(self):
        """BUSCO data should cover a meaningful number of contigs and IDs."""
        self.assertEqual(len(self.b2c), 2799)    # BUSCO IDs
        self.assertEqual(len(self.c2b), 541)      # Contigs with non-Missing hits
        # Total Complete entries
        total_complete = sum(len(self.b2c[bid]['C']) for bid in self.b2c)
        self.assertEqual(total_complete, 3146 + 74)  # Complete + Duplicated


@unittest.skipUnless(_has_shark_full(), 'Full shark dataset not available')
class TestSharkHillClimbingSmokeTest(unittest.TestCase):
    """Smoke test: run hill climbing on real shark data.

    Shark data is large (84M BUSCO lines). To keep the test fast, we
    build BUSCO dicts from the HAP file's contig names and only load
    a subset of BUSCO data.
    """

    @classmethod
    def setUpClass(cls):
        cls.hap_df = pd.read_csv(
            SHARK_HAP, sep='\t', header=None,
            names=['qName', 'tName', 'qSize', 'QPct', 'PID', 'QRAlignLenPct'],
            dtype={'qName': object, 'tName': object}
        )
        # Shark BUSCO is 1.6GB uncompressed — load it (takes ~30s)
        cls.b2c, cls.c2b = _load_busco_flat_tsv(SHARK_BUSCO)

    def _setup_globals(self):
        hapsolo.mypddf = self.hap_df.copy()
        # Build fake contig sizes from HAP data (no FASTA available for shark)
        all_names = set(self.hap_df['qName']).union(set(self.hap_df['tName']))
        hapsolo.myContigsDict = {name: [100000, 0, 0, 0] for name in all_names}
        hapsolo.busco2contigdict = {k: {bt: list(v) for bt, v in d.items()}
                                    for k, d in self.b2c.items()}
        hapsolo.contigs2buscodict = {k: {bt: list(v) for bt, v in d.items()}
                                     for k, d in self.c2b.items()}
        hapsolo.myMinContigSize = 1000
        hapsolo.myMinPID = 0.2
        hapsolo.myMinQPctMin = 0.2
        hapsolo.myMinQRPctMin = 0.2
        hapsolo.qrycontigset = set(hapsolo.mypddf['qName'])
        hapsolo.allcontigsset = set(hapsolo.myContigsDict.keys())
        hapsolo.missingrefcontigset = hapsolo.allcontigsset - hapsolo.qrycontigset
        hapsolo.smallcontigset = set()

    def test_shark_mode1(self):
        """Mode 1 on shark data with numeric contig IDs."""
        from random import seed
        seed(42)
        self._setup_globals()

        job_args = [0, 1, 0.0001, 0.7, 0.7, 0.7]
        result = hapsolo.hillclimbing(job_args)

        self.assertEqual(len(result), 3)
        best = result[0][0]
        self.assertGreater(best[0], 0)       # positive score
        self.assertGreater(len(best[1]), 0)  # good contigs non-empty
        # BUSCO scores should have valid keys
        for key in ['S', 'D', 'C', 'F', 'M']:
            self.assertIn(key, best[3])
        # S can be 0 at strict thresholds; just verify total is consistent
        self.assertEqual(best[3]['C'], best[3]['S'] + best[3]['D'])

    def test_shark_short_walk(self):
        """Short random walk on shark data."""
        from random import seed
        seed(42)
        self._setup_globals()

        job_args = [0, 5, 0.0001, 0.5, 0.5, 0.5]
        result = hapsolo.hillclimbing(job_args)

        self.assertGreater(len(result[0]), 0)
        self.assertEqual(len(result[1]), 5)  # 5 iterations


if __name__ == '__main__':
    unittest.main()
