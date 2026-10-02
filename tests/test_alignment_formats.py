#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for PAF alignment loading, gzipped file handling, HAP file output
format, self-alignment exclusion, and shark-style numeric contig IDs.

Uses realistic data patterns sampled from actual mosquito and shark genomes.
"""
import os
import sys
import gzip
import shutil
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TESTS_DIR)
FIXTURES_DIR = os.path.join(TESTS_DIR, 'fixtures')

# Mock sys.argv before importing hapsolo
sys.argv = [
    'hapsolo.py',
    '-i', os.path.join(FIXTURES_DIR, 'test_assembly.fasta'),
    '--paf', os.path.join(FIXTURES_DIR, 'test_alignment.paf'),
    '-b', os.path.join(FIXTURES_DIR, 'busco')
]
sys.path.insert(0, PROJECT_DIR)
import hapsolo


class _AlignmentTestBase(unittest.TestCase):
    """Shared setUp/tearDown that lowers filter thresholds and manages workdir.

    Note: conftest.py auto-restores all globals after each test.
    We only need to set thresholds and manage the workdir here.
    """

    def setUp(self):
        self.workdir = os.path.join(TESTS_DIR, 'tmp_' + self.__class__.__name__)
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


class TestPAFLoading(_AlignmentTestBase):
    """Test CreateMM2AlignmentDataStructure with PAF files."""

    def test_loads_paf(self):
        """Should load and parse an uncompressed PAF alignment file."""
        src = os.path.join(FIXTURES_DIR, 'test_alignment.paf')
        paf_copy = os.path.join(self.workdir, 'test.paf')
        shutil.copy2(src, paf_copy)
        result = hapsolo.CreateMM2AlignmentDataStructure(paf_copy)
        self.assertTrue(len(result) > 0)
        self.assertIn('qName', result.columns)
        self.assertIn('tName', result.columns)
        self.assertIn('QPct', result.columns)
        self.assertIn('PID', result.columns)
        self.assertIn('QRAlignLenPct', result.columns)

    def test_paf_column_count(self):
        """HAP DataFrame should have exactly 6 columns."""
        src = os.path.join(FIXTURES_DIR, 'test_alignment.paf')
        paf_copy = os.path.join(self.workdir, 'test.paf')
        shutil.copy2(src, paf_copy)
        result = hapsolo.CreateMM2AlignmentDataStructure(paf_copy)
        self.assertEqual(len(result.columns), 6)

    def test_paf_self_alignments_excluded(self):
        """Self-alignments (qName == tName) should be filtered out."""
        src = os.path.join(FIXTURES_DIR, 'test_alignment.paf')
        paf_copy = os.path.join(self.workdir, 'test.paf')
        shutil.copy2(src, paf_copy)
        result = hapsolo.CreateMM2AlignmentDataStructure(paf_copy)
        self_aligns = result[result['qName'] == result['tName']]
        self.assertEqual(len(self_aligns), 0,
                         'Self-alignments should be excluded but found: '
                         + str(self_aligns[['qName', 'tName']].to_dict()))

    def test_paf_numeric_columns_are_numeric(self):
        """qSize, QPct, PID, QRAlignLenPct should be numeric types."""
        src = os.path.join(FIXTURES_DIR, 'test_alignment.paf')
        paf_copy = os.path.join(self.workdir, 'test.paf')
        shutil.copy2(src, paf_copy)
        result = hapsolo.CreateMM2AlignmentDataStructure(paf_copy)
        import numpy as np
        for col in ['qSize', 'QPct', 'PID', 'QRAlignLenPct']:
            self.assertTrue(np.issubdtype(result[col].dtype, np.number),
                            col + ' should be numeric but is ' + str(result[col].dtype))

    def test_paf_qname_tname_are_strings(self):
        """qName and tName should be object (string) type."""
        src = os.path.join(FIXTURES_DIR, 'test_alignment.paf')
        paf_copy = os.path.join(self.workdir, 'test.paf')
        shutil.copy2(src, paf_copy)
        result = hapsolo.CreateMM2AlignmentDataStructure(paf_copy)
        self.assertEqual(result['qName'].dtype, object)
        self.assertEqual(result['tName'].dtype, object)


class TestPAFGzipped(_AlignmentTestBase):
    """Test CreateMM2AlignmentDataStructure with gzipped PAF files."""

    def test_loads_paf_gz(self):
        """Should load and parse a gzipped PAF file."""
        src = os.path.join(FIXTURES_DIR, 'test_alignment.paf.gz')
        paf_copy = os.path.join(self.workdir, 'test.paf.gz')
        shutil.copy2(src, paf_copy)
        result = hapsolo.CreateMM2AlignmentDataStructure(paf_copy)
        self.assertTrue(len(result) > 0)
        self.assertIn('qName', result.columns)

    def test_paf_gz_matches_uncompressed(self):
        """Gzipped PAF should produce the same result as uncompressed."""
        paf_src = os.path.join(FIXTURES_DIR, 'test_alignment.paf')
        paf_gz_src = os.path.join(FIXTURES_DIR, 'test_alignment.paf.gz')

        paf_copy = os.path.join(self.workdir, 'test.paf')
        paf_gz_copy = os.path.join(self.workdir, 'test_gz.paf.gz')
        shutil.copy2(paf_src, paf_copy)
        shutil.copy2(paf_gz_src, paf_gz_copy)

        result_plain = hapsolo.CreateMM2AlignmentDataStructure(paf_copy)
        result_gz = hapsolo.CreateMM2AlignmentDataStructure(paf_gz_copy)

        self.assertEqual(len(result_plain), len(result_gz))
        # Column values should match
        for col in ['qSize', 'QPct', 'PID', 'QRAlignLenPct']:
            self.assertTrue(
                (result_plain[col].values == result_gz[col].values).all(),
                col + ' values differ between plain and gzipped PAF')

    def test_paf_gz_self_alignments_excluded(self):
        """Gzipped PAF should also exclude self-alignments."""
        src = os.path.join(FIXTURES_DIR, 'test_alignment.paf.gz')
        paf_copy = os.path.join(self.workdir, 'test.paf.gz')
        shutil.copy2(src, paf_copy)
        result = hapsolo.CreateMM2AlignmentDataStructure(paf_copy)
        self_aligns = result[result['qName'] == result['tName']]
        self.assertEqual(len(self_aligns), 0)


class TestPSLGzipped(_AlignmentTestBase):
    """Test CreateBlatAlignmentDataStructure with gzipped PSL files."""

    def test_loads_psl_gz(self):
        """Should load and parse a gzipped PSL file."""
        src = os.path.join(FIXTURES_DIR, 'test_alignment.psl.gz')
        psl_copy = os.path.join(self.workdir, 'test.psl.gz')
        shutil.copy2(src, psl_copy)
        result = hapsolo.CreateBlatAlignmentDataStructure(psl_copy)
        self.assertTrue(len(result) > 0)
        self.assertIn('qName', result.columns)
        self.assertIn('PID', result.columns)

    def test_psl_gz_matches_uncompressed(self):
        """Gzipped PSL should produce same result as uncompressed."""
        psl_src = os.path.join(FIXTURES_DIR, 'test_alignment.psl')
        psl_gz_src = os.path.join(FIXTURES_DIR, 'test_alignment.psl.gz')

        psl_copy = os.path.join(self.workdir, 'test.psl')
        psl_gz_copy = os.path.join(self.workdir, 'test_gz.psl.gz')
        shutil.copy2(psl_src, psl_copy)
        shutil.copy2(psl_gz_src, psl_gz_copy)

        result_plain = hapsolo.CreateBlatAlignmentDataStructure(psl_copy)
        result_gz = hapsolo.CreateBlatAlignmentDataStructure(psl_gz_copy)

        self.assertEqual(len(result_plain), len(result_gz))
        for col in ['qSize', 'QPct', 'PID', 'QRAlignLenPct']:
            self.assertTrue(
                (result_plain[col].values == result_gz[col].values).all(),
                col + ' values differ between plain and gzipped PSL')


class TestHAPFileFormat(_AlignmentTestBase):
    """Validate the .hap intermediate file matches expected format."""

    def test_hap_file_tsv_6_columns(self):
        """The .hap file should be a TSV with 6 columns per line."""
        src = os.path.join(FIXTURES_DIR, 'test_alignment.paf')
        paf_copy = os.path.join(self.workdir, 'test.paf')
        shutil.copy2(src, paf_copy)
        hapsolo.CreateMM2AlignmentDataStructure(paf_copy)

        hap_file = os.path.join(self.workdir, 'test.hap')
        self.assertTrue(os.path.exists(hap_file), '.hap file not created')

        with open(hap_file) as f:
            for i, line in enumerate(f):
                fields = line.strip().split('\t')
                self.assertEqual(len(fields), 6,
                                 'Line ' + str(i) + ' has ' + str(len(fields))
                                 + ' fields, expected 6: ' + repr(line))

    def test_hap_file_quoted_names(self):
        """qName and tName in .hap file should be double-quoted."""
        src = os.path.join(FIXTURES_DIR, 'test_alignment.paf')
        paf_copy = os.path.join(self.workdir, 'test.paf')
        shutil.copy2(src, paf_copy)
        hapsolo.CreateMM2AlignmentDataStructure(paf_copy)

        hap_file = os.path.join(self.workdir, 'test.hap')
        with open(hap_file) as f:
            for i, line in enumerate(f):
                fields = line.strip().split('\t')
                qname, tname = fields[0], fields[1]
                self.assertTrue(qname.startswith('"') and qname.endswith('"'),
                                'qName not quoted on line ' + str(i) + ': ' + repr(qname))
                self.assertTrue(tname.startswith('"') and tname.endswith('"'),
                                'tName not quoted on line ' + str(i) + ': ' + repr(tname))

    def test_hap_file_numeric_fields(self):
        """qSize, QPct, PID, QRAlignLenPct in .hap should be parseable as numbers."""
        src = os.path.join(FIXTURES_DIR, 'test_alignment.paf')
        paf_copy = os.path.join(self.workdir, 'test.paf')
        shutil.copy2(src, paf_copy)
        hapsolo.CreateMM2AlignmentDataStructure(paf_copy)

        hap_file = os.path.join(self.workdir, 'test.hap')
        with open(hap_file) as f:
            for i, line in enumerate(f):
                fields = line.strip().split('\t')
                try:
                    int(fields[2])      # qSize
                    float(fields[3])    # QPct
                    float(fields[4])    # PID
                    float(fields[5])    # QRAlignLenPct
                except ValueError as e:
                    self.fail('Non-numeric field on line ' + str(i) + ': ' + str(e))

    def test_hap_file_loadable_by_pandas(self):
        """The .hap file should be loadable by pandas in the same way as in the notebook."""
        import pandas as pd
        src = os.path.join(FIXTURES_DIR, 'test_alignment.paf')
        paf_copy = os.path.join(self.workdir, 'test.paf')
        shutil.copy2(src, paf_copy)
        hapsolo.CreateMM2AlignmentDataStructure(paf_copy)

        hap_file = os.path.join(self.workdir, 'test.hap')
        # Load exactly as the notebook does
        df = pd.read_csv(hap_file, sep='\t', header=None,
                         names=['qName', 'tName', 'qSize', 'QPct', 'PID', 'QRAlignLenPct'],
                         dtype={'qName': object, 'tName': object})
        self.assertEqual(len(df.columns), 6)
        self.assertTrue(len(df) > 0)
        # Names should not contain quotes after pandas parsing
        for name in df['qName']:
            self.assertFalse(str(name).startswith('"'),
                             'pandas should strip quotes but got: ' + repr(name))

    def test_hap_psl_file_format(self):
        """PSL-generated .hap should have same format as PAF-generated .hap."""
        src = os.path.join(FIXTURES_DIR, 'test_alignment.psl')
        psl_copy = os.path.join(self.workdir, 'test.psl')
        shutil.copy2(src, psl_copy)
        hapsolo.CreateBlatAlignmentDataStructure(psl_copy)

        hap_file = os.path.join(self.workdir, 'test.hap')
        self.assertTrue(os.path.exists(hap_file))

        with open(hap_file) as f:
            for i, line in enumerate(f):
                fields = line.strip().split('\t')
                self.assertEqual(len(fields), 6)
                self.assertTrue(fields[0].startswith('"'))
                self.assertTrue(fields[1].startswith('"'))


class TestSharkStyleIDs(_AlignmentTestBase):
    """Test with shark-style purely numeric contig identifiers."""

    def test_numeric_ids_loaded(self):
        """Numeric contig IDs (e.g. '000438') should load correctly."""
        src = os.path.join(FIXTURES_DIR, 'test_alignment_shark.paf')
        paf_copy = os.path.join(self.workdir, 'shark.paf')
        shutil.copy2(src, paf_copy)
        result = hapsolo.CreateMM2AlignmentDataStructure(paf_copy)
        self.assertTrue(len(result) > 0)
        # Names should be strings, not parsed as integers
        for name in result['qName']:
            self.assertIsInstance(name, str)

    def test_numeric_ids_preserved(self):
        """Leading zeros in numeric IDs should be preserved."""
        src = os.path.join(FIXTURES_DIR, 'test_alignment_shark.paf')
        paf_copy = os.path.join(self.workdir, 'shark.paf')
        shutil.copy2(src, paf_copy)
        result = hapsolo.CreateMM2AlignmentDataStructure(paf_copy)
        all_names = set(result['qName']).union(set(result['tName']))
        # Should have names like '000438', not '438'
        for name in all_names:
            if name.isdigit():
                self.assertEqual(len(name), 6,
                                 'Leading zeros lost for ID: ' + repr(name))

    def test_shark_fasta_loading(self):
        """CalculateContigSizes should handle numeric contig IDs."""
        fasta = os.path.join(FIXTURES_DIR, 'test_assembly_shark.fasta')
        result = hapsolo.CalculateContigSizes(fasta)
        self.assertIn('000438', result)
        self.assertIn('000793', result)
        self.assertIn('006203', result)

    def test_shark_name_conversion(self):
        """Name conversion should work with numeric IDs."""
        fasta = os.path.join(FIXTURES_DIR, 'test_assembly_shark.fasta')
        canonical = set(hapsolo.CalculateContigSizes(fasta).keys())

        # Simulate external names with different format
        external = {'000438', '000793_arrow'}
        conv, unmatched = hapsolo.build_conversion_dict(canonical, external)

        # 000438 is exact match, no conversion needed
        self.assertNotIn('000438', conv)
        # 000793_arrow prefix-matches '000793' in canonical
        # (000793 is a prefix of 000793_arrow), so it should be converted
        self.assertEqual(conv.get('000793_arrow'), '000793')


class TestPAFValueCalculations(_AlignmentTestBase):
    """Verify PAF alignment value calculations match expected ranges."""

    def test_pid_range(self):
        """PID values should be between 0 and 1."""
        src = os.path.join(FIXTURES_DIR, 'test_alignment.paf')
        paf_copy = os.path.join(self.workdir, 'test.paf')
        shutil.copy2(src, paf_copy)
        result = hapsolo.CreateMM2AlignmentDataStructure(paf_copy)
        self.assertTrue((result['PID'] >= 0).all(), 'PID has values < 0')
        self.assertTrue((result['PID'] <= 1).all(), 'PID has values > 1')

    def test_qpct_range(self):
        """QPct values should be between 0 and 1."""
        src = os.path.join(FIXTURES_DIR, 'test_alignment.paf')
        paf_copy = os.path.join(self.workdir, 'test.paf')
        shutil.copy2(src, paf_copy)
        result = hapsolo.CreateMM2AlignmentDataStructure(paf_copy)
        self.assertTrue((result['QPct'] >= 0).all(), 'QPct has values < 0')
        self.assertTrue((result['QPct'] <= 1).all(), 'QPct has values > 1')

    def test_qralignlenpct_positive(self):
        """QRAlignLenPct should be positive."""
        src = os.path.join(FIXTURES_DIR, 'test_alignment.paf')
        paf_copy = os.path.join(self.workdir, 'test.paf')
        shutil.copy2(src, paf_copy)
        result = hapsolo.CreateMM2AlignmentDataStructure(paf_copy)
        self.assertTrue((result['QRAlignLenPct'] >= 0).all(),
                         'QRAlignLenPct has negative values')


if __name__ == '__main__':
    unittest.main()
