#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for hapsolo.py functions by mocking sys.argv before import.
Tests CalculateContigSizes, importBuscos, alignment loading, and the
name conversion pipeline.

Note: conftest.py auto-isolates all hapsolo globals between tests.
"""
import os
import sys
import shutil
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TESTS_DIR)
FIXTURES_DIR = os.path.join(TESTS_DIR, 'fixtures')

# Mock sys.argv so hapsolo's module-level argparse doesn't crash on import.
# Use the clean test fixtures as defaults.
sys.argv = [
    'hapsolo.py',
    '-i', os.path.join(FIXTURES_DIR, 'test_assembly.fasta'),
    '--psl', os.path.join(FIXTURES_DIR, 'test_alignment.psl'),
    '-b', os.path.join(FIXTURES_DIR, 'busco')
]

# Add project root to path so we can import hapsolo
sys.path.insert(0, PROJECT_DIR)
import hapsolo


class TestCalculateContigSizes(unittest.TestCase):

    def test_reads_all_contigs(self):
        """Should find all 3 contigs in test assembly."""
        fasta = os.path.join(FIXTURES_DIR, 'test_assembly.fasta')
        result = hapsolo.CalculateContigSizes(fasta)
        self.assertEqual(len(result), 3)
        self.assertIn('tig00000001', result)
        self.assertIn('tig00000002', result)
        self.assertIn('tig00000003', result)

    def test_contig_sizes_correct(self):
        """Each contig has 160 bp (2 lines of 80 chars each)."""
        fasta = os.path.join(FIXTURES_DIR, 'test_assembly.fasta')
        result = hapsolo.CalculateContigSizes(fasta)
        for name in result:
            self.assertEqual(result[name][0], 160,
                             'Expected 160 bp for ' + name + ', got ' + str(result[name][0]))

    def test_special_char_header_accepted(self):
        """Headers with pipes should be accepted without warnings."""
        test_fasta = os.path.join(TESTS_DIR, 'tmp_sc_test.fasta')
        try:
            with open(test_fasta, 'w') as f:
                f.write('>tig001|arrow\n')
                f.write('ATCGATCG\n')
            result = hapsolo.CalculateContigSizes(test_fasta)
            self.assertIn('tig001|arrow', result)
        finally:
            if os.path.exists(test_fasta):
                os.remove(test_fasta)


class TestSanitizeName(unittest.TestCase):
    """Tests the imported sanitize_name function from hapsolo.py."""

    def test_pipe(self):
        self.assertEqual(hapsolo.sanitize_name('tig001|arrow'), 'tig001_arrow')

    def test_clean(self):
        self.assertEqual(hapsolo.sanitize_name('tig001'), 'tig001')

    def test_dot_preserved(self):
        self.assertEqual(hapsolo.sanitize_name('tig.1'), 'tig.1')


class TestBuildConversionDict(unittest.TestCase):
    """Tests the imported build_conversion_dict function from hapsolo.py."""

    def test_sanitized_match(self):
        canonical = {'tig001_arrow'}
        external = {'tig001|arrow'}
        conv, unmatched = hapsolo.build_conversion_dict(canonical, external)
        self.assertEqual(conv['tig001|arrow'], 'tig001_arrow')
        self.assertEqual(len(unmatched), 0)

    def test_prefix_match(self):
        canonical = {'tig00000001_arrow_pilon'}
        external = {'tig00000001'}
        conv, unmatched = hapsolo.build_conversion_dict(canonical, external)
        self.assertEqual(conv['tig00000001'], 'tig00000001_arrow_pilon')

    def test_no_match(self):
        canonical = {'tig001'}
        external = {'scaffold999'}
        conv, unmatched = hapsolo.build_conversion_dict(canonical, external)
        self.assertIn('scaffold999', unmatched)


class TestImportBuscos(unittest.TestCase):
    """conftest.py resets busco2contigdict/contigs2buscodict automatically."""

    def test_loads_busco_data(self):
        """Should load BUSCO data from test fixtures."""
        busco_dir = os.path.join(FIXTURES_DIR, 'busco')
        b2c, c2b = hapsolo.importBuscos(busco_dir)

        self.assertTrue(len(b2c) > 0)
        self.assertTrue(len(c2b) > 0)
        self.assertIn('tig00000001', c2b)
        self.assertIn('tig00000002', c2b)
        self.assertIn('tig00000003', c2b)

    def test_busco_counts(self):
        """Verify BUSCO classification for known test data."""
        busco_dir = os.path.join(FIXTURES_DIR, 'busco')
        b2c, c2b = hapsolo.importBuscos(busco_dir)

        tig1 = c2b['tig00000001']
        self.assertEqual(len(tig1['C']), 2)  # BUSCO001, BUSCO002
        self.assertEqual(len(tig1['F']), 1)  # BUSCO003


class TestImportBuscosContigAttribution(unittest.TestCase):
    """Verify that each BUSCO file's entries are attributed to the correct contig.

    Regression tests for issue #20: if a BUSCO file has a malformed header
    (missing the 3rd comment line that contains the input FASTA path),
    the old code would silently use the contigname from the PREVIOUS file,
    attributing all BUSCOs to the wrong contig.
    """

    def test_each_contig_gets_own_buscos(self):
        """BUSCOs from each file must be attributed to only that file's contig."""
        busco_dir = os.path.join(FIXTURES_DIR, 'busco')
        b2c, c2b = hapsolo.importBuscos(busco_dir)

        # tig00000001 has BUSCO001 Complete, BUSCO002 Complete, BUSCO003 Fragmented
        self.assertIn('BUSCO001', c2b['tig00000001']['C'])
        self.assertIn('BUSCO002', c2b['tig00000001']['C'])
        self.assertIn('BUSCO003', c2b['tig00000001']['F'])

        # BUSCO001 is Complete on BOTH tig00000001 and tig00000002 (legitimate duplicate)
        # Verify each contig appears exactly once in the BUSCO's contig list
        busco001_contigs = b2c['BUSCO001']['C']
        self.assertEqual(busco001_contigs.count('tig00000001'), 1)
        self.assertEqual(busco001_contigs.count('tig00000002'), 1)

        # BUSCO003 is Fragmented on tig00000001 only — should NOT leak to tig00000002
        self.assertIn('BUSCO003', c2b['tig00000001']['F'])
        self.assertNotIn('BUSCO003', c2b['tig00000002']['F'])

        # BUSCO004 is Missing on tig00000001 — should have no Complete entries for tig00000001
        busco004_complete = b2c['BUSCO004']['C']
        self.assertNotIn('tig00000001', busco004_complete)

    def test_no_cross_contamination_between_files(self):
        """Use the busco_scoring fixture: verify no contig gets another's BUSCOs."""
        busco_dir = os.path.join(FIXTURES_DIR, 'busco_scoring')
        b2c, c2b = hapsolo.importBuscos(busco_dir)

        # From the fixture:
        # contigA Complete: BUSCO001, BUSCO002, BUSCO003
        # contigB Complete: BUSCO001, BUSCO003, BUSCO005, BUSCO006
        # contigC Complete: BUSCO002, BUSCO008, BUSCO009

        # BUSCO005 is Complete ONLY on contigB
        self.assertEqual(b2c['BUSCO005']['C'], ['contigB'])
        # BUSCO008 is Complete ONLY on contigC
        self.assertEqual(b2c['BUSCO008']['C'], ['contigC'])

        # contigA must NOT have BUSCO005 or BUSCO008 as Complete
        self.assertNotIn('BUSCO005', c2b['contigA']['C'])
        self.assertNotIn('BUSCO008', c2b['contigA']['C'])

        # contigC must NOT have BUSCO001 as Complete (only A and B have it)
        self.assertNotIn('BUSCO001', c2b['contigC']['C'])

    def test_malformed_busco_file_skipped(self):
        """A BUSCO file with missing header line 3 should be skipped, not leak."""
        busco_dir = os.path.join(FIXTURES_DIR, 'busco_leakage')
        b2c, c2b = hapsolo.importBuscos(busco_dir)

        # contigA (well-formed) should have BUSCO001 Complete
        self.assertIn('contigA', c2b)
        self.assertIn('BUSCO001', c2b['contigA']['C'])

        # contigC (well-formed) should have BUSCO003 Complete
        self.assertIn('contigC', c2b)
        self.assertIn('BUSCO003', c2b['contigC']['C'])

        # The malformed file's BUSCO002 should NOT be attributed to contigA or contigC
        # (the old bug would have leaked contigA's name into the malformed file)
        contigA_complete = set(c2b['contigA']['C'])
        contigC_complete = set(c2b['contigC']['C'])
        self.assertNotIn('BUSCO002', contigA_complete,
                         'BUSCO002 from malformed file leaked to contigA')
        self.assertNotIn('BUSCO002', contigC_complete,
                         'BUSCO002 from malformed file leaked to contigC')

    def test_malformed_file_buscos_not_attributed(self):
        """BUSCOs from a malformed file should not appear in any contig's dict."""
        busco_dir = os.path.join(FIXTURES_DIR, 'busco_leakage')
        b2c, c2b = hapsolo.importBuscos(busco_dir)

        # Collect ALL contigs that have BUSCO002 Complete
        busco002_contigs = b2c['BUSCO002']['C']

        # The malformed file had BUSCO002 Complete for "contigB" but contigB
        # was never registered in the first pass (no valid header), so
        # BUSCO002 should have zero Complete entries
        self.assertEqual(len(busco002_contigs), 0,
                         'BUSCO002 from malformed file was attributed to: '
                         + str(busco002_contigs))

    def test_well_formed_files_unaffected_by_malformed_neighbor(self):
        """Well-formed BUSCO files should produce correct results even when
        a malformed file is in the same directory."""
        busco_dir = os.path.join(FIXTURES_DIR, 'busco_leakage')
        b2c, c2b = hapsolo.importBuscos(busco_dir)

        # contigA: BUSCO001=Complete, BUSCO002=Missing, BUSCO003=Missing
        self.assertEqual(c2b['contigA']['C'], ['BUSCO001'])
        self.assertEqual(c2b['contigA']['F'], [])

        # contigC: BUSCO001=Missing, BUSCO002=Missing, BUSCO003=Complete
        self.assertEqual(c2b['contigC']['C'], ['BUSCO003'])
        self.assertEqual(c2b['contigC']['F'], [])

        # Total contig count should be 2 (only A and C, not B)
        self.assertEqual(len(c2b), 2)

    def test_busco_scoring_with_malformed_file_present(self):
        """calculateBuscos should produce correct scores even with a skipped file."""
        busco_dir = os.path.join(FIXTURES_DIR, 'busco_leakage')
        b2c, c2b = hapsolo.importBuscos(busco_dir)

        hapsolo.missingrefcontigset = set()
        hapsolo.smallcontigset = set()

        # Score with both valid contigs
        result = hapsolo.calculateBuscos(['contigA', 'contigC'], b2c, c2b)

        # BUSCO001: Complete on contigA only -> Single
        # BUSCO003: Complete on contigC only -> Single
        # BUSCO002: not Complete anywhere (malformed file skipped) -> Missing
        self.assertEqual(result['S'], 2)
        self.assertEqual(result['D'], 0)
        self.assertEqual(result['M'], 1)  # BUSCO002
        self.assertEqual(result['C'], 2)  # S + D


class TestBlatAlignmentLoading(unittest.TestCase):

    def setUp(self):
        self.workdir = os.path.join(TESTS_DIR, 'tmp_blat_test')
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

    def test_loads_psl(self):
        """Should load and parse PSL alignment file."""
        psl = os.path.join(FIXTURES_DIR, 'test_alignment.psl')
        psl_copy = os.path.join(self.workdir, 'test.psl')
        shutil.copy2(psl, psl_copy)
        result = hapsolo.CreateBlatAlignmentDataStructure(psl_copy)
        self.assertTrue(len(result) > 0)
        self.assertIn('qName', result.columns)
        self.assertIn('PID', result.columns)


class TestNameConversionPipeline(unittest.TestCase):
    """
    End-to-end test: FASTA has clean names, alignment has names with pipes.
    Conversion dict should remap the alignment names.
    """

    def setUp(self):
        self.workdir = os.path.join(TESTS_DIR, 'tmp_conv_test')
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

    def test_alignment_name_conversion(self):
        """Alignment with pipe-names should be remapped to match FASTA."""
        fasta = os.path.join(FIXTURES_DIR, 'test_assembly.fasta')
        canonical = set(hapsolo.CalculateContigSizes(fasta).keys())

        psl_src = os.path.join(FIXTURES_DIR, 'test_alignment_mismatch.psl')
        psl_copy = os.path.join(self.workdir, 'mismatch.psl')
        shutil.copy2(psl_src, psl_copy)
        mypddf = hapsolo.CreateBlatAlignmentDataStructure(psl_copy)

        aln_names = set(mypddf['qName']).union(set(mypddf['tName']))

        conv, unmatched = hapsolo.build_conversion_dict(canonical, aln_names)

        if len(aln_names - canonical) > 0:
            self.assertTrue(len(conv) > 0,
                            'Expected conversions but got none. '
                            'aln_names=' + str(aln_names) + ' canonical=' + str(canonical))

            mypddf['qName'] = mypddf['qName'].replace(conv)
            mypddf['tName'] = mypddf['tName'].replace(conv)

            new_aln_names = set(mypddf['qName']).union(set(mypddf['tName']))
            unresolved = new_aln_names - canonical
            self.assertEqual(len(unresolved), 0,
                             'Unresolved names after conversion: ' + str(unresolved))

    def test_busco_name_conversion(self):
        """BUSCO results with pipe-names should be remapped to match FASTA."""
        fasta = os.path.join(FIXTURES_DIR, 'test_assembly.fasta')
        canonical = set(hapsolo.CalculateContigSizes(fasta).keys())

        busco_dir = os.path.join(FIXTURES_DIR, 'busco_mismatch')
        b2c, c2b = hapsolo.importBuscos(busco_dir)

        busco_names = set(c2b.keys())
        conv, unmatched = hapsolo.build_conversion_dict(canonical, busco_names)

        if len(busco_names - canonical) > 0:
            self.assertTrue(len(conv) > 0,
                            'Expected BUSCO name conversions but got none. '
                            'busco_names=' + str(busco_names) + ' canonical=' + str(canonical))

            new_c2b = dict()
            for name in c2b:
                new_name = conv.get(name, name)
                new_c2b[new_name] = c2b[name]

            remapped_names = set(new_c2b.keys())
            unresolved = remapped_names - canonical
            self.assertEqual(len(unresolved), 0,
                             'Unresolved BUSCO names after conversion: ' + str(unresolved))


class TestCalculatePctAlign(unittest.TestCase):
    """Test the alignment percentage calculation helper."""

    def test_normal(self):
        self.assertAlmostEqual(hapsolo.CalculatePctAlign(50, 100), 0.5)

    def test_zero_total(self):
        self.assertEqual(hapsolo.CalculatePctAlign(50, 0), 0.0)

    def test_full(self):
        self.assertAlmostEqual(hapsolo.CalculatePctAlign(100, 100), 1.0)


if __name__ == '__main__':
    unittest.main()
