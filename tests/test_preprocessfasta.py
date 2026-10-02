#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integration tests for preprocessfasta.py.
Runs the script on synthetic FASTA files and validates output.
"""
import os
import sys
import shutil
import subprocess
import tempfile
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TESTS_DIR)
FIXTURES_DIR = os.path.join(TESTS_DIR, 'fixtures')
SCRIPT_MODULE = [sys.executable, '-m', 'hapsolo.preprocess']


class TestPreprocessFasta(unittest.TestCase):

    def setUp(self):
        """Create a temporary working directory for each test."""
        self.workdir = tempfile.mkdtemp(prefix='hapsolo_test_')

    def tearDown(self):
        """Clean up temporary working directory."""
        if os.path.exists(self.workdir):
            shutil.rmtree(self.workdir)

    def _make_env(self):
        """Build env with PYTHONPATH pointing to the project root."""
        env = os.environ.copy()
        env['PYTHONPATH'] = PROJECT_DIR + os.pathsep + env.get('PYTHONPATH', '')
        return env

    def _run_preprocess(self, fasta_file, extra_args=None):
        """Run hapsolo.preprocess and return (returncode, stdout, stderr)."""
        # Copy input to workdir
        input_copy = os.path.join(self.workdir, os.path.basename(fasta_file))
        shutil.copy2(fasta_file, input_copy)
        cmd = SCRIPT_MODULE + ['-i', os.path.basename(fasta_file)]
        if extra_args:
            cmd.extend(extra_args)
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=self.workdir, env=self._make_env()
        )
        stdout, stderr = proc.communicate()
        return proc.returncode, stdout.decode('utf-8', errors='replace'), stderr.decode('utf-8', errors='replace')

    def test_clean_headers_passthrough(self):
        """FASTA with clean headers should produce matching output."""
        fasta = os.path.join(FIXTURES_DIR, 'test_assembly.fasta')
        rc, stdout, stderr = self._run_preprocess(fasta)
        self.assertEqual(rc, 0, 'preprocessfasta failed: ' + stderr)

        # Check _new.fasta exists
        new_fasta = os.path.join(self.workdir, 'test_assembly_new.fasta')
        self.assertTrue(os.path.exists(new_fasta))

        # Check contigs directory has individual files
        contigs_dir = os.path.join(self.workdir, 'contigs')
        self.assertTrue(os.path.isdir(contigs_dir))
        contig_files = [f for f in os.listdir(contigs_dir) if f.endswith('.fasta')]
        self.assertEqual(len(contig_files), 3)

        # Check name mapping file exists
        mapping_file = os.path.join(contigs_dir, 'name_mapping.tsv')
        self.assertTrue(os.path.exists(mapping_file))

    def test_special_chars_sanitized(self):
        """FASTA headers with pipes should be sanitized to underscores."""
        fasta = os.path.join(FIXTURES_DIR, 'test_assembly_specialchars.fasta')
        rc, stdout, stderr = self._run_preprocess(fasta)
        self.assertEqual(rc, 0, 'preprocessfasta failed: ' + stderr)

        new_fasta = os.path.join(self.workdir, 'test_assembly_specialchars_new.fasta')
        self.assertTrue(os.path.exists(new_fasta))

        # Read output headers
        headers = []
        with open(new_fasta) as f:
            for line in f:
                if line.startswith('>'):
                    headers.append(line.strip()[1:])

        # No pipes should remain
        for h in headers:
            self.assertNotIn('|', h, 'Pipe character not sanitized in header: ' + h)

        # All 3 contigs should be present
        self.assertEqual(len(headers), 3)

    def test_name_mapping_file_content(self):
        """Name mapping file should map original -> sanitized names."""
        fasta = os.path.join(FIXTURES_DIR, 'test_assembly_specialchars.fasta')
        rc, stdout, stderr = self._run_preprocess(fasta)
        self.assertEqual(rc, 0)

        mapping_file = os.path.join(self.workdir, 'contigs', 'name_mapping.tsv')
        mappings = {}
        with open(mapping_file) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    mappings[parts[0]] = parts[1]

        # Original names had pipes
        self.assertIn('tig00000001|arrow', mappings)
        self.assertIn('tig00000002|arrow', mappings)
        self.assertIn('tig00000003|arrow', mappings)

        # Sanitized names should not have pipes
        for orig, sanitized in mappings.items():
            self.assertNotIn('|', sanitized)

    def test_unique_headers_after_sanitization(self):
        """All output headers must be unique."""
        fasta = os.path.join(FIXTURES_DIR, 'test_assembly_specialchars.fasta')
        rc, stdout, stderr = self._run_preprocess(fasta)
        self.assertEqual(rc, 0)

        new_fasta = os.path.join(self.workdir, 'test_assembly_specialchars_new.fasta')
        headers = []
        with open(new_fasta) as f:
            for line in f:
                if line.startswith('>'):
                    headers.append(line.strip()[1:])

        self.assertEqual(len(headers), len(set(headers)),
                         'Duplicate headers found: ' + str(headers))

    def test_sequence_integrity(self):
        """Sequence content should be preserved after preprocessing."""
        fasta = os.path.join(FIXTURES_DIR, 'test_assembly.fasta')
        rc, stdout, stderr = self._run_preprocess(fasta)
        self.assertEqual(rc, 0)

        # Read original sequences
        orig_seqs = {}
        current = None
        with open(fasta) as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    current = line[1:].split()[0]
                    orig_seqs[current] = ''
                elif current and line:
                    orig_seqs[current] += line

        # Read preprocessed sequences
        new_fasta = os.path.join(self.workdir, 'test_assembly_new.fasta')
        new_seqs = {}
        current = None
        with open(new_fasta) as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    current = line[1:]
                    new_seqs[current] = ''
                elif current and line:
                    new_seqs[current] += line

        # Same number of sequences
        self.assertEqual(len(orig_seqs), len(new_seqs))

        # Sequence lengths should match
        orig_lens = sorted(len(s) for s in orig_seqs.values())
        new_lens = sorted(len(s) for s in new_seqs.values())
        self.assertEqual(orig_lens, new_lens)

    def test_maxcontig_flag(self):
        """The -m flag should control which contigs get individual files."""
        # Create a FASTA with one large and one small contig
        test_fasta = os.path.join(self.workdir, 'size_test.fasta')
        with open(test_fasta, 'w') as f:
            f.write('>small_contig\n')
            f.write('ATCG' * 10 + '\n')  # 40 bp
            f.write('>large_contig\n')
            f.write('ATCG' * 500 + '\n')  # 2000 bp

        cmd = SCRIPT_MODULE + ['-i', 'size_test.fasta', '-m', '1']
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=self.workdir, env=self._make_env()
        )
        stdout, stderr = proc.communicate()
        self.assertEqual(proc.returncode, 0, stderr.decode('utf-8', errors='replace'))

        # -m 1 means 1 Mb max. Both contigs are < 1 Mb, so both should be written
        contigs_dir = os.path.join(self.workdir, 'contigs')
        contig_files = [f for f in os.listdir(contigs_dir) if f.endswith('.fasta')]
        self.assertEqual(len(contig_files), 2)


if __name__ == '__main__':
    unittest.main()
