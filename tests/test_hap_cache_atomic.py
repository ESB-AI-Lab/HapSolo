#!/usr/bin/env python
"""The .hap alignment cache must only ever exist complete.

Every later run reuses a non-empty .hap without re-reading the alignment, so a partial cache from
an interrupted run (the thorny skate cache of 2026-07-12 held 1,502 of 207,119 alignments)
silently corrupts all later training. The cache is now written to a temp file and renamed on
success.
"""
import os
import shutil
import sys
import tempfile
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TESTS_DIR)
FIXTURES_DIR = os.path.join(TESTS_DIR, 'fixtures')
sys.path.insert(0, PROJECT_DIR)

from hapsolo.alignment import create_paf_alignment, create_psl_alignment  # noqa: E402


class TestAtomicHapCache(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='hapsolo_hap_')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    N_RECORDS = 50

    def _records(self):
        # Every record passes the default prefilter: QPct 0.8, PID 0.9, QRPct 1.0, qLen 5000.
        return ['c%d\t5000\t0\t4000\t+\tc%d\t5000\t0\t4000\t3600\t4000\t60' % (i, i + 1)
                for i in range(self.N_RECORDS)]

    def _paf(self, name='aln.paf', corrupt_after=None):
        src = self._records()
        if corrupt_after is not None:
            src = src[:corrupt_after] + ['truncated\trecord'] + src[corrupt_after:]
        path = os.path.join(self.tmp, name)
        with open(path, 'w') as fh:
            fh.write('\n'.join(src) + '\n')
        return path

    def _leftovers(self):
        return [f for f in os.listdir(self.tmp) if f.endswith('.tmp')]

    def test_complete_parse_writes_cache_and_no_temp(self):
        paf = self._paf()
        df = create_paf_alignment(paf, 1000, 0.2, 0.2, 0.2)
        self.assertEqual(len(df), self.N_RECORDS)
        hap = paf.replace('.paf', '.hap')
        self.assertTrue(os.path.exists(hap))
        with open(hap) as fh:
            self.assertEqual(sum(1 for _ in fh), len(df))
        self.assertEqual(self._leftovers(), [])

    def test_interrupted_parse_leaves_no_cache(self):
        paf = self._paf(corrupt_after=25)
        with self.assertRaises(ValueError):
            create_paf_alignment(paf, 1000, 0.2, 0.2, 0.2)
        self.assertFalse(os.path.exists(paf.replace('.paf', '.hap')))
        self.assertEqual(self._leftovers(), [])

    def test_rerun_after_interruption_reparses(self):
        paf = self._paf(corrupt_after=25)
        with self.assertRaises(ValueError):
            create_paf_alignment(paf, 1000, 0.2, 0.2, 0.2)
        with open(paf, 'w') as fh:
            fh.write('\n'.join(self._records()) + '\n')
        # A partial cache would hold 25 records and be loaded instead of the repaired PAF.
        self.assertEqual(len(create_paf_alignment(paf, 1000, 0.2, 0.2, 0.2)), self.N_RECORDS)

    def test_psl_complete_parse_has_no_temp(self):
        psl = os.path.join(self.tmp, 'aln.psl')
        shutil.copy(os.path.join(FIXTURES_DIR, 'test_alignment.psl'), psl)
        create_psl_alignment(psl, 1000, 0.2, 0.2, 0.2)
        self.assertTrue(os.path.exists(psl.replace('.psl', '.hap')))
        self.assertEqual(self._leftovers(), [])


if __name__ == '__main__':
    unittest.main()
