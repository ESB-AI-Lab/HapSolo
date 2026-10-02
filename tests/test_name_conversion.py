#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for sanitize_name and build_conversion_dict functions.
"""
import unittest

from hapsolo.names import sanitize_name, build_conversion_dict


class TestSanitizeName(unittest.TestCase):

    def test_pipe_replaced(self):
        self.assertEqual(sanitize_name('tig00000001|arrow'), 'tig00000001_arrow')

    def test_spaces_replaced(self):
        self.assertEqual(sanitize_name('tig 001 scaffold'), 'tig_001_scaffold')

    def test_multiple_special_chars(self):
        self.assertEqual(sanitize_name('contig#1@v2(test)'), 'contig_1_v2_test_')

    def test_dots_preserved(self):
        self.assertEqual(sanitize_name('tig001.pilon.v2'), 'tig001.pilon.v2')

    def test_already_clean(self):
        self.assertEqual(sanitize_name('tig00000001'), 'tig00000001')

    def test_slashes_replaced(self):
        self.assertEqual(sanitize_name('scaffold/001'), 'scaffold_001')

    def test_hyphens_replaced(self):
        self.assertEqual(sanitize_name('contig-001-v2'), 'contig_001_v2')

    def test_empty_string(self):
        self.assertEqual(sanitize_name(''), '')


class TestBuildConversionDict(unittest.TestCase):

    def test_exact_match_no_conversion(self):
        """Names that already match should not appear in conversion dict."""
        canonical = {'tig001', 'tig002', 'tig003'}
        external = {'tig001', 'tig002', 'tig003'}
        conv, unmatched = build_conversion_dict(canonical, external)
        self.assertEqual(len(conv), 0)
        self.assertEqual(len(unmatched), 0)

    def test_sanitized_match_pipe(self):
        """Names with pipes should match sanitized canonical names."""
        canonical = {'tig00000001_arrow', 'tig00000002_arrow'}
        external = {'tig00000001|arrow', 'tig00000002|arrow'}
        conv, unmatched = build_conversion_dict(canonical, external)
        self.assertEqual(conv['tig00000001|arrow'], 'tig00000001_arrow')
        self.assertEqual(conv['tig00000002|arrow'], 'tig00000002_arrow')
        self.assertEqual(len(unmatched), 0)

    def test_sanitized_match_various_chars(self):
        """Names with various special chars should match sanitized versions."""
        canonical = {'contig_1_v2'}
        external = {'contig-1-v2'}
        conv, unmatched = build_conversion_dict(canonical, external)
        self.assertEqual(conv['contig-1-v2'], 'contig_1_v2')

    def test_prefix_match_truncated(self):
        """Truncated names should match via prefix matching."""
        canonical = {'tig00000001_arrow_pilon'}
        external = {'tig00000001'}
        conv, unmatched = build_conversion_dict(canonical, external)
        self.assertEqual(conv['tig00000001'], 'tig00000001_arrow_pilon')

    def test_prefix_match_reverse(self):
        """Full external names should match truncated canonical via prefix."""
        canonical = {'tig00000001'}
        external = {'tig00000001_arrow_pilon'}
        conv, unmatched = build_conversion_dict(canonical, external)
        self.assertEqual(conv['tig00000001_arrow_pilon'], 'tig00000001')

    def test_ambiguous_prefix_unmatched(self):
        """Ambiguous prefix matches should be reported as unmatched."""
        canonical = {'tig001_a', 'tig001_b'}
        external = {'tig001'}
        conv, unmatched = build_conversion_dict(canonical, external)
        self.assertIn('tig001', unmatched)
        self.assertNotIn('tig001', conv)

    def test_completely_unmatched(self):
        """Names with no match at all should be in unmatched set."""
        canonical = {'tig001', 'tig002'}
        external = {'scaffold_99'}
        conv, unmatched = build_conversion_dict(canonical, external)
        self.assertIn('scaffold_99', unmatched)

    def test_mixed_match_and_mismatch(self):
        """Mix of exact, sanitized, and unmatched names."""
        canonical = {'tig001', 'tig002_arrow', 'tig003'}
        external = {'tig001', 'tig002|arrow', 'unknown_contig'}
        conv, unmatched = build_conversion_dict(canonical, external)
        # tig001 is exact match, no conversion
        self.assertNotIn('tig001', conv)
        # tig002|arrow sanitizes to tig002_arrow
        self.assertEqual(conv['tig002|arrow'], 'tig002_arrow')
        # unknown_contig has no match
        self.assertIn('unknown_contig', unmatched)

    def test_sanitized_ambiguity_skipped(self):
        """When two canonical names sanitize identically, both are skipped."""
        # These both sanitize to 'contig_1'
        canonical = {'contig-1', 'contig_1'}
        external = {'contig|1'}
        conv, unmatched = build_conversion_dict(canonical, external)
        # contig|1 sanitizes to contig_1, but that's ambiguous
        # It should match 'contig_1' exactly though... no wait,
        # canonical has both 'contig-1' and 'contig_1'. 'contig-1' sanitizes to 'contig_1'.
        # 'contig_1' sanitizes to 'contig_1'. So sanitized_lookup['contig_1'] = None (ambiguous).
        # But 'contig_1' is still in canonical_set, so exact match check happens first.
        # For 'contig|1': sanitize -> 'contig_1', lookup is None (ambiguous).
        # Then prefix match tries, but both sanitized canons are 'contig_1' -> None.
        # So it should be unmatched.
        self.assertIn('contig|1', unmatched)

    def test_empty_inputs(self):
        """Empty inputs should produce empty outputs."""
        conv, unmatched = build_conversion_dict(set(), set())
        self.assertEqual(len(conv), 0)
        self.assertEqual(len(unmatched), 0)

    def test_external_subset(self):
        """External names that are a subset of canonical should all match."""
        canonical = {'tig001', 'tig002', 'tig003'}
        external = {'tig001'}
        conv, unmatched = build_conversion_dict(canonical, external)
        self.assertEqual(len(conv), 0)
        self.assertEqual(len(unmatched), 0)


if __name__ == '__main__':
    unittest.main()
