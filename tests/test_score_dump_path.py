#!/usr/bin/env python
"""Score dumps (.scores/.deltascores) must never overwrite the input assembly.

Before 2026-10-01 the dump paths were built with asm.replace('.fasta', ...), so an input named
*.fa or *.fna was overwritten with scores after the primary/secondary FASTAs were written.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TESTS_DIR)
FIXTURES_DIR = os.path.join(TESTS_DIR, 'fixtures')
sys.path.insert(0, PROJECT_DIR)

import hapsolo  # noqa: E402
from hapsolo.__main__ import score_dump_path  # noqa: E402

# Run the subprocess against the same package these tests imported (repo, install or container).
PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(hapsolo.__file__)))


class TestScoreDumpPath(unittest.TestCase):

    def test_fasta_name_keeps_historical_naming(self):
        self.assertEqual(score_dump_path('asm_new.fasta', 'T', '.scores'), 'asm_new_T.scores')
        self.assertEqual(score_dump_path('dir/asm.fasta', 'T', '.deltascores'), 'dir/asm_T.deltascores')

    def test_other_fasta_extensions_are_stripped(self):
        for ext in ('.fa', '.fna', '.fas', '.FA'):
            self.assertEqual(score_dump_path('asm' + ext, 'T', '.scores'), 'asm_T.scores')

    def test_unknown_extension_is_appended_not_replaced(self):
        self.assertEqual(score_dump_path('asm.txt', 'T', '.scores'), 'asm.txt_T.scores')
        self.assertEqual(score_dump_path('asm', 'T', '.scores'), 'asm_T.scores')

    def test_never_returns_input_path(self):
        for name in ('a.fa', 'a.fna', 'a.fasta', 'a', 'a.fasta.gz', 'x/a.fa'):
            for suffix in ('.scores', '.deltascores'):
                self.assertNotEqual(os.path.abspath(score_dump_path(name, 'T', suffix)),
                                    os.path.abspath(name))


class TestInputNotOverwritten(unittest.TestCase):
    """End-to-end: run the optimizer on a *.fa input and check the input is untouched."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='hapsolo_fa_')
        shutil.copy(os.path.join(FIXTURES_DIR, 'test_assembly_paf.fasta'), os.path.join(self.tmp, 'asm.fa'))
        shutil.copy(os.path.join(FIXTURES_DIR, 'test_alignment.paf'), os.path.join(self.tmp, 'aln.paf'))
        shutil.copytree(os.path.join(FIXTURES_DIR, 'busco'), os.path.join(self.tmp, 'busco'))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fa_input_md5_unchanged_and_scores_written(self):
        path = os.path.join(self.tmp, 'asm.fa')
        before = hashlib.md5(open(path, 'rb').read()).hexdigest()
        env = dict(os.environ, PYTHONPATH=PACKAGE_PARENT)
        res = subprocess.run([sys.executable, '-m', 'hapsolo', '-i', 'asm.fa', '--paf', 'aln.paf',
                              '-b', 'busco', '--mode', '0', '-n', '3', '-t', '1', '--outdir', 'out'],
                             cwd=self.tmp, env=env, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr[-2000:])
        self.assertEqual(hashlib.md5(open(path, 'rb').read()).hexdigest(), before)
        dumps = sorted(f for f in os.listdir(self.tmp) if f.endswith(('.scores', '.deltascores')))
        self.assertEqual(len(dumps), 2, os.listdir(self.tmp))
        self.assertTrue(all(f.startswith('asm_') for f in dumps), dumps)


if __name__ == '__main__':
    unittest.main()
