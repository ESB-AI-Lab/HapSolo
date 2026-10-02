#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for hapsolo_cli.py — the unified CLI wrapper.

Focuses on argument parsing, subcommand routing, path construction,
and error handling. All subprocess calls are mocked so no external
tools (minimap2, miniprot, blat) are needed.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TESTS_DIR)

sys.path.insert(0, PROJECT_DIR)
import hapsolo_cli


# ── resolve_python ─────────────────────────────────────────────────────────

class TestResolvePython(unittest.TestCase):
    """Tests for resolve_python()."""

    def test_default_returns_sys_executable(self):
        """When --python is not specified, resolve_python returns sys.executable."""
        args = MagicMock()
        args.python = None
        result = hapsolo_cli.resolve_python(args)
        self.assertEqual(result, sys.executable)

    def test_python_not_set_attribute(self):
        """When args has no 'python' attribute at all, returns sys.executable."""
        args = MagicMock(spec=[])  # no attributes
        result = hapsolo_cli.resolve_python(args)
        self.assertEqual(result, sys.executable)

    @patch('hapsolo_cli.shutil.which')
    def test_python_found_on_path(self, mock_which):
        """When --python is given and found on PATH, return the resolved path."""
        mock_which.return_value = '/usr/bin/python3.10'
        args = MagicMock()
        args.python = 'python3.10'
        result = hapsolo_cli.resolve_python(args)
        self.assertEqual(result, '/usr/bin/python3.10')
        mock_which.assert_called_once_with('python3.10')

    @patch('hapsolo_cli.shutil.which')
    def test_python_not_found_exits(self, mock_which):
        """When --python is given but not found, sys.exit(1) is called."""
        mock_which.return_value = None
        args = MagicMock()
        args.python = 'python3.99'
        with self.assertRaises(SystemExit) as ctx:
            hapsolo_cli.resolve_python(args)
        self.assertEqual(ctx.exception.code, 1)


# ── check_tool ─────────────────────────────────────────────────────────────

class TestCheckTool(unittest.TestCase):
    """Tests for check_tool()."""

    @patch('hapsolo_cli.shutil.which')
    def test_tool_available(self, mock_which):
        mock_which.return_value = '/usr/bin/minimap2'
        self.assertTrue(hapsolo_cli.check_tool('minimap2'))

    @patch('hapsolo_cli.shutil.which')
    def test_tool_not_available(self, mock_which):
        mock_which.return_value = None
        self.assertFalse(hapsolo_cli.check_tool('nonexistent_tool'))


# ── run_cmd ────────────────────────────────────────────────────────────────

class TestRunCmd(unittest.TestCase):
    """Tests for run_cmd()."""

    @patch('hapsolo_cli.subprocess.run')
    def test_success(self, mock_run):
        """Successful command returns the CompletedProcess."""
        mock_run.return_value = MagicMock(returncode=0)
        result = hapsolo_cli.run_cmd(['echo', 'hello'])
        self.assertEqual(result.returncode, 0)
        mock_run.assert_called_once()
        self.assertEqual(mock_run.call_args[0][0], ['echo', 'hello'])

    @patch('hapsolo_cli.subprocess.run')
    def test_failure_exits(self, mock_run):
        """Failed command calls sys.exit with the exit code."""
        mock_run.return_value = MagicMock(returncode=42)
        with self.assertRaises(SystemExit) as ctx:
            hapsolo_cli.run_cmd(['false'])
        self.assertEqual(ctx.exception.code, 42)

    @patch('hapsolo_cli.subprocess.run')
    def test_description_is_printed(self, mock_run):
        """run_cmd with description should not crash (smoke test)."""
        mock_run.return_value = MagicMock(returncode=0)
        result = hapsolo_cli.run_cmd(['echo'], description='Test step')
        self.assertEqual(result.returncode, 0)


# ── Argument parsing ──────────────────────────────────────────────────────

class TestParserNoSubcommand(unittest.TestCase):
    """Tests for invoking with no subcommand."""

    def test_no_args_exits(self):
        """Running with no arguments should print help and exit(0)."""
        with self.assertRaises(SystemExit) as ctx:
            with patch('sys.argv', ['hapsolo']):
                hapsolo_cli.main()
        self.assertEqual(ctx.exception.code, 0)

    def test_invalid_subcommand_exits(self):
        """An unrecognized subcommand should cause an error exit."""
        with self.assertRaises(SystemExit) as ctx:
            with patch('sys.argv', ['hapsolo', 'bogus']):
                hapsolo_cli.main()
        self.assertNotEqual(ctx.exception.code, 0)


class TestPreprocessParsing(unittest.TestCase):
    """Argument parsing for the 'preprocess' subcommand."""

    def test_missing_input_exits(self):
        """preprocess without -i should fail."""
        with self.assertRaises(SystemExit) as ctx:
            with patch('sys.argv', ['hapsolo', 'preprocess']):
                hapsolo_cli.main()
        self.assertNotEqual(ctx.exception.code, 0)

    @patch('hapsolo_cli.run_cmd')
    def test_valid_args(self, mock_run_cmd):
        """preprocess with -i should succeed."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', 'preprocess', '-i', 'asm.fasta']):
            hapsolo_cli.main()
        mock_run_cmd.assert_called_once()

    @patch('hapsolo_cli.run_cmd')
    def test_maxcontig_passed(self, mock_run_cmd):
        """preprocess with -m should include maxcontig in the command."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', 'preprocess', '-i', 'asm.fasta', '-m', '5']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        # Find the -m that comes after the module arg (not python -m)
        m_indices = [i for i, c in enumerate(cmd) if c == '-m']
        self.assertTrue(len(m_indices) >= 2, 'Expected -m for module and -m for maxcontig')
        # Last -m should be the maxcontig flag with value '5'
        self.assertEqual(cmd[m_indices[-1] + 1], '5')


class TestAlignParsing(unittest.TestCase):
    """Argument parsing for the 'align' subcommand."""

    def test_missing_input_exits(self):
        """align without -i should fail."""
        with self.assertRaises(SystemExit) as ctx:
            with patch('sys.argv', ['hapsolo', 'align']):
                hapsolo_cli.main()
        self.assertNotEqual(ctx.exception.code, 0)

    def test_invalid_aligner_exits(self):
        """align with an invalid --aligner should fail."""
        with self.assertRaises(SystemExit) as ctx:
            with patch('sys.argv', ['hapsolo', 'align', '-i', 'asm.fasta',
                                    '--aligner', 'bowtie2']):
                hapsolo_cli.main()
        self.assertNotEqual(ctx.exception.code, 0)


class TestSearchParsing(unittest.TestCase):
    """Argument parsing for the 'search' subcommand."""

    def test_missing_input_exits(self):
        """search without -i should fail."""
        with self.assertRaises(SystemExit) as ctx:
            with patch('sys.argv', ['hapsolo', 'search', '-l', 'lineage/']):
                hapsolo_cli.main()
        self.assertNotEqual(ctx.exception.code, 0)

    def test_missing_lineage_exits(self):
        """search without -l should fail."""
        with self.assertRaises(SystemExit) as ctx:
            with patch('sys.argv', ['hapsolo', 'search', '-i', 'asm.fasta']):
                hapsolo_cli.main()
        self.assertNotEqual(ctx.exception.code, 0)


class TestTrainParsing(unittest.TestCase):
    """Argument parsing for the 'train' subcommand."""

    def test_missing_input_exits(self):
        """train without -i should fail."""
        with self.assertRaises(SystemExit) as ctx:
            with patch('sys.argv', ['hapsolo', 'train', '-b', 'busco/']):
                hapsolo_cli.main()
        self.assertNotEqual(ctx.exception.code, 0)

    def test_missing_buscos_exits(self):
        """train without -b should fail."""
        with self.assertRaises(SystemExit) as ctx:
            with patch('sys.argv', ['hapsolo', 'train', '-i', 'asm.fasta',
                                    '--paf', 'align.paf']):
                hapsolo_cli.main()
        self.assertNotEqual(ctx.exception.code, 0)

    def test_invalid_mode_exits(self):
        """train with invalid --mode should fail."""
        with self.assertRaises(SystemExit) as ctx:
            with patch('sys.argv', ['hapsolo', 'train', '-i', 'asm.fasta',
                                    '-b', 'busco/', '--paf', 'a.paf',
                                    '--mode', '5']):
                hapsolo_cli.main()
        self.assertNotEqual(ctx.exception.code, 0)


class TestClassifyParsing(unittest.TestCase):
    """Argument parsing for the 'classify' subcommand."""

    def test_missing_input_exits(self):
        """classify without -i should fail."""
        with self.assertRaises(SystemExit) as ctx:
            with patch('sys.argv', ['hapsolo', 'classify', '-b', 'busco/']):
                hapsolo_cli.main()
        self.assertNotEqual(ctx.exception.code, 0)

    def test_missing_buscos_exits(self):
        """classify without -b should fail."""
        with self.assertRaises(SystemExit) as ctx:
            with patch('sys.argv', ['hapsolo', 'classify', '-i', 'asm.fasta',
                                    '--paf', 'align.paf']):
                hapsolo_cli.main()
        self.assertNotEqual(ctx.exception.code, 0)


# ── Subcommand routing and path construction ───────────────────────────────

class TestPreprocessCommand(unittest.TestCase):
    """Verify preprocess subcommand constructs the right subprocess call."""

    @patch('hapsolo_cli.run_cmd')
    def test_basic_command(self, mock_run_cmd):
        """preprocess should call preprocessfasta.py with -i."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', 'preprocess', '-i', 'assembly.fasta']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        # Should call python -m hapsolo.preprocess ... -i assembly.fasta
        self.assertTrue(any('hapsolo.preprocess' in c for c in cmd))
        self.assertIn('-i', cmd)
        self.assertIn('assembly.fasta', cmd)

    @patch('hapsolo_cli.run_cmd')
    def test_maxcontig_included(self, mock_run_cmd):
        """preprocess -m 5 should include -m 5 in the command."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', 'preprocess', '-i', 'asm.fasta', '-m', '20']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        self.assertIn('-m', cmd)
        self.assertIn('20', cmd)

    @patch('hapsolo_cli.run_cmd')
    def test_maxcontig_not_included_when_omitted(self, mock_run_cmd):
        """preprocess without -m should NOT include -m for maxcontig."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', 'preprocess', '-i', 'asm.fasta']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        # Only one -m should be present (the python -m for module execution)
        m_count = cmd.count('-m')
        self.assertEqual(m_count, 1, 'Expected only python -m, not maxcontig -m')


class TestAlignCommand(unittest.TestCase):
    """Verify align subcommand constructs the right subprocess call."""

    @patch('hapsolo_cli.check_tool', return_value=True)
    @patch('hapsolo_cli.run_cmd')
    def test_minimap2_default(self, mock_run_cmd, mock_check):
        """align defaults to minimap2 and constructs the right command."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', 'align', '-i', 'asm.fasta',
                                '-t', '4', '--no-gzip']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        self.assertIn('minimap2', cmd)
        self.assertIn('-t', cmd)
        idx = cmd.index('-t')
        self.assertEqual(cmd[idx + 1], '4')
        # Both query and target should be the input file
        self.assertEqual(cmd.count('asm.fasta'), 2)

    @patch('hapsolo_cli.check_tool', return_value=True)
    @patch('hapsolo_cli.run_cmd')
    def test_minimap2_output_flag(self, mock_run_cmd, mock_check):
        """align -o custom.paf should use that filename."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', 'align', '-i', 'asm.fasta',
                                '-o', 'custom.paf', '--no-gzip']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        self.assertIn('-o', cmd)
        idx = cmd.index('-o')
        self.assertEqual(cmd[idx + 1], 'custom.paf')

    @patch('hapsolo_cli.check_tool', return_value=True)
    @patch('hapsolo_cli.run_cmd')
    def test_blat_aligner(self, mock_run_cmd, mock_check):
        """align --aligner blat should call blat instead of minimap2."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', 'align', '-i', 'asm.fasta',
                                '--aligner', 'blat', '--no-gzip']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        self.assertIn('blat', cmd)
        self.assertNotIn('minimap2', cmd)

    @patch('hapsolo_cli.check_tool', return_value=False)
    def test_minimap2_not_found_exits(self, mock_check):
        """align should exit if minimap2 is not on PATH."""
        with self.assertRaises(SystemExit) as ctx:
            with patch('sys.argv', ['hapsolo', 'align', '-i', 'asm.fasta']):
                hapsolo_cli.main()
        self.assertNotEqual(ctx.exception.code, 0)

    @patch('hapsolo_cli.check_tool', return_value=False)
    def test_blat_not_found_exits(self, mock_check):
        """align --aligner blat should exit if blat is not on PATH."""
        with self.assertRaises(SystemExit) as ctx:
            with patch('sys.argv', ['hapsolo', 'align', '-i', 'asm.fasta',
                                    '--aligner', 'blat']):
                hapsolo_cli.main()
        self.assertNotEqual(ctx.exception.code, 0)

    @patch('hapsolo_cli.check_tool', return_value=True)
    @patch('hapsolo_cli.run_cmd')
    def test_minimap2_paper_params(self, mock_run_cmd, mock_check):
        """align should include the HapSolo paper minimap2 parameters."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', 'align', '-i', 'asm.fasta', '--no-gzip']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        # Check a few of the signature paper params
        self.assertIn('-P', cmd)
        self.assertIn('-N', cmd)
        idx = cmd.index('-N')
        self.assertEqual(cmd[idx + 1], '50')
        self.assertIn('--paf-no-hit', cmd)

    @patch('hapsolo_cli.check_tool', return_value=True)
    @patch('hapsolo_cli.run_cmd')
    def test_minimap2_default_output_name(self, mock_run_cmd, mock_check):
        """align without -o should auto-name the output file."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', 'align', '-i', 'genome.fasta', '--no-gzip']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        self.assertIn('-o', cmd)
        idx = cmd.index('-o')
        self.assertEqual(cmd[idx + 1], 'genome_self_align.paf')

    @patch('hapsolo_cli.check_tool', return_value=True)
    @patch('hapsolo_cli.run_cmd')
    def test_gzip_compression_step(self, mock_run_cmd, mock_check):
        """align without --no-gzip should issue a gzip command."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', 'align', '-i', 'asm.fasta']):
            hapsolo_cli.main()
        # run_cmd should be called at least twice: alignment + gzip
        self.assertGreaterEqual(mock_run_cmd.call_count, 2)
        gzip_call = mock_run_cmd.call_args_list[-1]
        gzip_cmd = gzip_call[0][0]
        self.assertIn('gzip', gzip_cmd)


class TestSearchCommand(unittest.TestCase):
    """Verify search subcommand constructs the right subprocess call."""

    @patch('hapsolo_cli.check_tool', return_value=True)
    @patch('hapsolo_cli.run_cmd')
    def test_basic_command(self, mock_run_cmd, mock_check):
        """search should call search_orthologs.py with correct args."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', 'search', '-i', 'asm.fasta',
                                '-l', 'diptera_odb10/', '-t', '8']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        self.assertTrue(any('hapsolo.search' in c for c in cmd))
        self.assertIn('-i', cmd)
        self.assertIn('asm.fasta', cmd)
        self.assertIn('-l', cmd)
        self.assertIn('diptera_odb10/', cmd)
        self.assertIn('-t', cmd)
        self.assertIn('8', cmd)

    @patch('hapsolo_cli.check_tool', return_value=True)
    @patch('hapsolo_cli.run_cmd')
    def test_custom_output_dir(self, mock_run_cmd, mock_check):
        """search -o mydir should pass -o mydir to the script."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', 'search', '-i', 'asm.fasta',
                                '-l', 'lineage/', '-o', 'mydir']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        self.assertIn('-o', cmd)
        self.assertIn('mydir', cmd)

    @patch('hapsolo_cli.check_tool', return_value=True)
    @patch('hapsolo_cli.run_cmd')
    def test_contig_dir_passed(self, mock_run_cmd, mock_check):
        """search --contig-dir should forward the flag."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', 'search', '-i', 'asm.fasta',
                                '-l', 'lineage/', '--contig-dir', 'mycontigs']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        self.assertIn('--contig-dir', cmd)
        self.assertIn('mycontigs', cmd)

    @patch('hapsolo_cli.check_tool', return_value=False)
    def test_miniprot_not_found_exits(self, mock_check):
        """search should exit if miniprot is not on PATH."""
        with self.assertRaises(SystemExit) as ctx:
            with patch('sys.argv', ['hapsolo', 'search', '-i', 'asm.fasta',
                                    '-l', 'lineage/']):
                hapsolo_cli.main()
        self.assertNotEqual(ctx.exception.code, 0)


class TestTrainCommand(unittest.TestCase):
    """Verify train subcommand constructs the right subprocess call."""

    @patch('hapsolo_cli.run_cmd')
    def test_basic_paf_command(self, mock_run_cmd):
        """train with --paf should call python -m hapsolo with -a flag."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', 'train', '-i', 'asm.fasta',
                                '-b', 'busco/', '--paf', 'align.paf']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        self.assertIn('-m', cmd)
        self.assertIn('hapsolo', cmd)
        self.assertIn('-a', cmd)
        self.assertIn('align.paf', cmd)
        self.assertIn('-i', cmd)
        self.assertIn('asm.fasta', cmd)
        self.assertIn('-b', cmd)
        self.assertIn('busco/', cmd)

    @patch('hapsolo_cli.run_cmd')
    def test_basic_psl_command(self, mock_run_cmd):
        """train with --psl should call python -m hapsolo with -p flag."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', 'train', '-i', 'asm.fasta',
                                '-b', 'busco/', '--psl', 'align.psl']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        self.assertIn('-p', cmd)
        self.assertIn('align.psl', cmd)

    @patch('hapsolo_cli.run_cmd')
    def test_no_alignment_exits(self, mock_run_cmd):
        """train without --paf or --psl should exit with error."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with self.assertRaises(SystemExit):
            with patch('sys.argv', ['hapsolo', 'train', '-i', 'asm.fasta',
                                    '-b', 'busco/']):
                hapsolo_cli.main()

    @patch('hapsolo_cli.run_cmd')
    def test_optional_params_included(self, mock_run_cmd):
        """train with optional params should forward them."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', 'train', '-i', 'asm.fasta',
                                '-b', 'busco/', '--paf', 'a.paf',
                                '-n', '500', '-t', '4', '--mode', '2',
                                '-B', '3', '--min-contig', '5000',
                                '-S', '2.0', '-D', '1.5', '-F', '0.5', '-M', '0.8']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        # Check mode
        self.assertIn('--mode', cmd)
        idx = cmd.index('--mode')
        self.assertEqual(cmd[idx + 1], '2')
        # Check iterations
        self.assertIn('-n', cmd)
        idx = cmd.index('-n')
        self.assertEqual(cmd[idx + 1], '500')
        # Check threads
        self.assertIn('-t', cmd)
        idx = cmd.index('-t')
        self.assertEqual(cmd[idx + 1], '4')
        # Check bestn
        self.assertIn('-B', cmd)
        idx = cmd.index('-B')
        self.assertEqual(cmd[idx + 1], '3')
        # Check min contig
        self.assertIn('--min', cmd)
        idx = cmd.index('--min')
        self.assertEqual(cmd[idx + 1], '5000')
        # Check theta weights
        self.assertIn('-S', cmd)
        self.assertIn('-D', cmd)
        self.assertIn('-F', cmd)
        self.assertIn('-M', cmd)

    @patch('hapsolo_cli.run_cmd')
    def test_optional_params_omitted(self, mock_run_cmd):
        """train without optional params should not include them."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', 'train', '-i', 'asm.fasta',
                                '-b', 'busco/', '--paf', 'a.paf']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        self.assertNotIn('-B', cmd)
        self.assertNotIn('--min', cmd)
        self.assertNotIn('-S', cmd)
        self.assertNotIn('-D', cmd)
        self.assertNotIn('-F', cmd)
        self.assertNotIn('-M', cmd)
        self.assertNotIn('--gpu', cmd)
        self.assertNotIn('--totaliters', cmd)
        self.assertNotIn('--agents', cmd)

    @patch('hapsolo_cli.run_cmd')
    def test_gpu_params_forwarded(self, mock_run_cmd):
        """train --gpu --totaliters --agents should forward all three."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', 'train', '-i', 'asm.fasta',
                                '-b', 'busco/', '--paf', 'a.paf', '--mode', '3',
                                '--gpu', '--totaliters', '100000', '--agents', '128']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        self.assertIn('--gpu', cmd)
        self.assertEqual(cmd[cmd.index('--totaliters') + 1], '100000')
        self.assertEqual(cmd[cmd.index('--agents') + 1], '128')

    @patch('hapsolo_cli.run_cmd')
    def test_totaliters_without_gpu_forwarded(self, mock_run_cmd):
        """--totaliters also applies to CPU runs, so it is forwarded without --gpu."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', 'train', '-i', 'asm.fasta',
                                '-b', 'busco/', '--paf', 'a.paf',
                                '-t', '8', '--totaliters', '50000']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        self.assertNotIn('--gpu', cmd)
        self.assertEqual(cmd[cmd.index('--totaliters') + 1], '50000')

    @patch('hapsolo_cli.run_cmd')
    def test_gpu_command_accepted_by_optimizer_parser(self, mock_run_cmd):
        """The forwarded command must parse with python -m hapsolo's real argparse."""
        from hapsolo.__main__ import build_parser
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', 'train', '-i', 'asm.fasta',
                                '-b', 'busco/', '--paf', 'a.paf', '--mode', '3',
                                '--gpu', '--totaliters', '100000', '--agents', '128']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        parsed = build_parser().parse_args(cmd[cmd.index('hapsolo') + 1:])
        self.assertTrue(parsed.gpu)
        self.assertEqual(parsed.totaliters, 100000)
        self.assertEqual(parsed.agents, 128)
        self.assertEqual(parsed.mode, 3)

    @patch('hapsolo_cli.run_cmd')
    def test_default_mode_is_zero(self, mock_run_cmd):
        """train without --mode should default to mode 0."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', 'train', '-i', 'asm.fasta',
                                '-b', 'busco/', '--paf', 'a.paf']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        idx = cmd.index('--mode')
        self.assertEqual(cmd[idx + 1], '0')

    @patch('hapsolo_cli.run_cmd')
    def test_default_iterations_1000(self, mock_run_cmd):
        """train without -n should default to 1000 iterations."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', 'train', '-i', 'asm.fasta',
                                '-b', 'busco/', '--paf', 'a.paf']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        idx = cmd.index('-n')
        self.assertEqual(cmd[idx + 1], '1000')


class TestClassifyCommand(unittest.TestCase):
    """Verify classify subcommand constructs the right subprocess call."""

    @patch('hapsolo_cli.run_cmd')
    def test_basic_paf_command(self, mock_run_cmd):
        """classify with --paf should call python -m hapsolo --mode 1."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', 'classify', '-i', 'asm.fasta',
                                '-b', 'busco/', '--paf', 'align.paf']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        self.assertIn('-m', cmd)
        self.assertIn('hapsolo', cmd)
        self.assertIn('--mode', cmd)
        idx = cmd.index('--mode')
        self.assertEqual(cmd[idx + 1], '1')
        self.assertIn('-a', cmd)
        self.assertIn('align.paf', cmd)

    @patch('hapsolo_cli.run_cmd')
    def test_basic_psl_command(self, mock_run_cmd):
        """classify with --psl should use -p flag."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', 'classify', '-i', 'asm.fasta',
                                '-b', 'busco/', '--psl', 'align.psl']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        self.assertIn('-p', cmd)
        self.assertIn('align.psl', cmd)

    @patch('hapsolo_cli.run_cmd')
    def test_no_alignment_exits(self, mock_run_cmd):
        """classify without --paf or --psl should exit with error."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with self.assertRaises(SystemExit):
            with patch('sys.argv', ['hapsolo', 'classify', '-i', 'asm.fasta',
                                    '-b', 'busco/']):
                hapsolo_cli.main()

    @patch('hapsolo_cli.run_cmd')
    def test_threshold_params_passed(self, mock_run_cmd):
        """classify with -P, -Q, -R should forward them."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', 'classify', '-i', 'asm.fasta',
                                '-b', 'busco/', '--paf', 'a.paf',
                                '-P', '0.8', '-Q', '0.6', '-R', '0.9']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        self.assertIn('-P', cmd)
        idx = cmd.index('-P')
        self.assertEqual(cmd[idx + 1], '0.8')
        self.assertIn('-Q', cmd)
        idx = cmd.index('-Q')
        self.assertEqual(cmd[idx + 1], '0.6')
        self.assertIn('-R', cmd)
        idx = cmd.index('-R')
        self.assertEqual(cmd[idx + 1], '0.9')

    @patch('hapsolo_cli.run_cmd')
    def test_threshold_params_omitted(self, mock_run_cmd):
        """classify without threshold flags should not include them."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', 'classify', '-i', 'asm.fasta',
                                '-b', 'busco/', '--paf', 'a.paf']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        self.assertNotIn('-P', cmd)
        self.assertNotIn('-Q', cmd)
        self.assertNotIn('-R', cmd)

    @patch('hapsolo_cli.run_cmd')
    def test_min_contig_passed(self, mock_run_cmd):
        """classify with --min-contig should forward it."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', 'classify', '-i', 'asm.fasta',
                                '-b', 'busco/', '--paf', 'a.paf',
                                '--min-contig', '2000']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        self.assertIn('--min', cmd)
        idx = cmd.index('--min')
        self.assertEqual(cmd[idx + 1], '2000')

    @patch('hapsolo_cli.run_cmd')
    def test_classify_always_mode_1(self, mock_run_cmd):
        """classify always passes --mode 1 (not configurable)."""
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', 'classify', '-i', 'asm.fasta',
                                '-b', 'busco/', '--paf', 'a.paf']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        idx = cmd.index('--mode')
        self.assertEqual(cmd[idx + 1], '1')


# ── Global --python flag ──────────────────────────────────────────────────

class TestGlobalPythonFlag(unittest.TestCase):
    """Verify that the global --python flag is forwarded to sub-scripts."""

    @patch('hapsolo_cli.shutil.which')
    @patch('hapsolo_cli.run_cmd')
    def test_python_flag_used_in_preprocess(self, mock_run_cmd, mock_which):
        """--python python3.10 should use that interpreter for preprocess."""
        mock_which.return_value = '/usr/bin/python3.10'
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', '--python', 'python3.10',
                                'preprocess', '-i', 'asm.fasta']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        self.assertEqual(cmd[0], '/usr/bin/python3.10')

    @patch('hapsolo_cli.shutil.which')
    @patch('hapsolo_cli.run_cmd')
    def test_python_flag_used_in_train(self, mock_run_cmd, mock_which):
        """--python should be used as the interpreter for train."""
        mock_which.return_value = '/opt/python3.11/bin/python3'
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', '--python', 'python3.11',
                                'train', '-i', 'asm.fasta', '-b', 'busco/',
                                '--paf', 'a.paf']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        self.assertEqual(cmd[0], '/opt/python3.11/bin/python3')

    @patch('hapsolo_cli.check_tool', return_value=True)
    @patch('hapsolo_cli.shutil.which')
    @patch('hapsolo_cli.run_cmd')
    def test_python_flag_used_in_search(self, mock_run_cmd, mock_which,
                                        mock_check):
        """--python should be used as the interpreter for search."""
        mock_which.return_value = '/usr/bin/python3.10'
        mock_run_cmd.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['hapsolo', '--python', 'python3.10',
                                'search', '-i', 'asm.fasta', '-l', 'lineage/']):
            hapsolo_cli.main()
        cmd = mock_run_cmd.call_args[0][0]
        self.assertEqual(cmd[0], '/usr/bin/python3.10')


if __name__ == '__main__':
    unittest.main()
