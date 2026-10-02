#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for UniquePriorityQueue, RandomWalkOptimizer, SteepestDescentOptimizer, and evaluate_thresholds."""
import os
import sys
import re
import unittest
import pandas as pd

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TESTS_DIR)
FIXTURES_DIR = os.path.join(TESTS_DIR, 'fixtures')

if 'hapsolo' not in sys.modules:
    sys.argv = [
        'hapsolo.py',
        '-i', os.path.join(FIXTURES_DIR, 'test_assembly.fasta'),
        '--paf', os.path.join(FIXTURES_DIR, 'test_alignment.paf'),
        '-b', os.path.join(FIXTURES_DIR, 'busco')
    ]
    sys.path.insert(0, PROJECT_DIR)
    import hapsolo

import hapsolo


# ── UniquePriorityQueue ────────────────────────────────────────────────────

class TestUniquePriorityQueue(unittest.TestCase):

    def test_add_and_sort(self):
        upq = hapsolo.UniquePriorityQueue(5)
        upq.add([0.5, {'a', 'b'}, {'c'}, {}, [0.5, 0.5, 0.5]])
        upq.add([0.3, {'a', 'b', 'c'}, set(), {}, [0.3, 0.3, 0.3]])
        upq.add([0.7, {'a'}, {'b', 'c'}, {}, [0.7, 0.7, 0.7]])
        self.assertEqual(len(upq), 3)
        self.assertAlmostEqual(upq[0][0], 0.3)
        self.assertAlmostEqual(upq[1][0], 0.5)
        self.assertAlmostEqual(upq[2][0], 0.7)

    def test_dedup_keeps_lower_score(self):
        upq = hapsolo.UniquePriorityQueue(5)
        upq.add([0.5, {'a', 'b'}, {'c'}, {}, [0.5, 0.5, 0.5]])
        upq.add([0.3, {'a', 'b'}, {'c'}, {}, [0.3, 0.3, 0.3]])
        self.assertEqual(len(upq), 1)
        self.assertAlmostEqual(upq[0][0], 0.3)

    def test_dedup_higher_score_first(self):
        upq = hapsolo.UniquePriorityQueue(5)
        upq.add([0.3, {'a', 'b'}, {'c'}, {}, [0.3, 0.3, 0.3]])
        upq.add([0.5, {'a', 'b'}, {'c'}, {}, [0.5, 0.5, 0.5]])
        self.assertEqual(len(upq), 1)
        self.assertAlmostEqual(upq[0][0], 0.3)

    def test_max_size(self):
        upq = hapsolo.UniquePriorityQueue(2)
        upq.add([0.5, {'a'}, {'b'}, {}, []])
        upq.add([0.3, {'b'}, {'a'}, {}, []])
        upq.add([0.1, {'c'}, {'d'}, {}, []])
        self.assertEqual(len(upq), 2)
        self.assertAlmostEqual(upq[0][0], 0.1)
        self.assertAlmostEqual(upq[1][0], 0.3)

    def test_should_add_when_full(self):
        upq = hapsolo.UniquePriorityQueue(2)
        upq.add([0.5, {'a'}, set(), {}, []])
        upq.add([0.3, {'b'}, set(), {}, []])
        self.assertTrue(upq.should_add(0.4))
        self.assertFalse(upq.should_add(0.6))

    def test_should_add_when_not_full(self):
        upq = hapsolo.UniquePriorityQueue(5)
        upq.add([0.5, {'a'}, set(), {}, []])
        self.assertTrue(upq.should_add(999.0))

    def test_empty_queue(self):
        upq = hapsolo.UniquePriorityQueue(3)
        self.assertEqual(len(upq), 0)
        self.assertTrue(upq.should_add(999.0))

    def test_max_size_one_with_dedup(self):
        upq = hapsolo.UniquePriorityQueue(1)
        upq.add([0.5, {'a', 'b'}, {'c'}, {}, [0.7, 0.7, 0.7]])
        upq.add([0.3, {'a', 'b'}, {'c'}, {}, [0.8, 0.8, 0.8]])
        self.assertEqual(len(upq), 1)
        self.assertAlmostEqual(upq[0][0], 0.3)

    def test_items_property_returns_list(self):
        upq = hapsolo.UniquePriorityQueue(3)
        upq.add([0.5, {'a'}, set(), {}, []])
        self.assertIsInstance(upq.items, list)
        self.assertEqual(len(upq.items), 1)

    def test_multiple_dedup_groups(self):
        upq = hapsolo.UniquePriorityQueue(5)
        upq.add([0.5, {'a'}, set(), {}, []])
        upq.add([0.3, {'a'}, set(), {}, []])
        upq.add([0.7, {'b'}, set(), {}, []])
        upq.add([0.4, {'b'}, set(), {}, []])
        self.assertEqual(len(upq), 2)
        self.assertAlmostEqual(upq[0][0], 0.3)
        self.assertAlmostEqual(upq[1][0], 0.4)


# ── backward-compatible wrapper ────────────────────────────────────────────

class TestUniquepriorityqueueWrapper(unittest.TestCase):

    def test_matches_class_behavior(self):
        saved = hapsolo.bestnscores
        try:
            hapsolo.bestnscores = 5
            pqlist = [[0.5, {'c1', 'c2'}, {'c3'}, {}, [0.7, 0.7, 0.7]]]
            new_val = [0.3, {'c1', 'c2', 'c3'}, set(), {}, [0.8, 0.8, 0.8]]
            result = hapsolo.uniquepriorityqueue(pqlist, new_val)
            self.assertAlmostEqual(result[0][0], 0.3)
            self.assertEqual(len(result), 2)
        finally:
            hapsolo.bestnscores = saved

    def test_dedup_through_wrapper(self):
        saved = hapsolo.bestnscores
        try:
            hapsolo.bestnscores = 5
            pqlist = [[0.5, {'c1', 'c2'}, {'c3'}, {}, [0.7, 0.7, 0.7]]]
            new_val = [0.3, {'c1', 'c2'}, {'c3'}, {}, [0.8, 0.8, 0.8]]
            result = hapsolo.uniquepriorityqueue(pqlist, new_val)
            self.assertEqual(len(result), 1)
            self.assertAlmostEqual(result[0][0], 0.3)
        finally:
            hapsolo.bestnscores = saved


# ── RandomWalkOptimizer ────────────────────────────────────────────────────

class TestRandomWalkOptimizer(unittest.TestCase):

    def test_all_out_of_bounds_resets(self):
        opt = hapsolo.RandomWalkOptimizer(0.2, 0.2, 0.2, 0.01, 10, 1e-6)
        deltas = [0.0] * 20
        pid, qpct, qrpct = opt.step(1.5, 1.5, 1.5, 1, deltas)
        self.assertGreaterEqual(pid, 0.2)
        self.assertLessEqual(pid, 1.0)
        self.assertGreaterEqual(qpct, 0.2)
        self.assertLessEqual(qpct, 1.0)
        self.assertGreaterEqual(qrpct, 0.2)
        self.assertLessEqual(qrpct, 1.0)

    def test_single_pid_out_of_bounds(self):
        opt = hapsolo.RandomWalkOptimizer(0.2, 0.2, 0.2, 0.01, 10, 1e-6)
        deltas = [0.1] * 20
        pid, qpct, qrpct = opt.step(1.5, 0.5, 0.5, 5, deltas)
        self.assertGreaterEqual(pid, 0.2)
        self.assertLessEqual(pid, 1.0)

    def test_single_qpct_out_of_bounds(self):
        opt = hapsolo.RandomWalkOptimizer(0.2, 0.2, 0.2, 0.01, 10, 1e-6)
        deltas = [0.1] * 20
        pid, qpct, qrpct = opt.step(0.5, 1.5, 0.5, 5, deltas)
        self.assertGreaterEqual(qpct, 0.2)
        self.assertLessEqual(qpct, 1.0)

    def test_single_qrpct_out_of_bounds(self):
        opt = hapsolo.RandomWalkOptimizer(0.2, 0.2, 0.2, 0.01, 10, 1e-6)
        deltas = [0.1] * 20
        pid, qpct, qrpct = opt.step(0.5, 0.5, 1.5, 5, deltas)
        self.assertGreaterEqual(qrpct, 0.2)
        self.assertLessEqual(qrpct, 1.0)

    def test_plateau_resets_all(self):
        opt = hapsolo.RandomWalkOptimizer(0.2, 0.2, 0.2, 0.01, 3, 1e-6)
        deltas = [0.0] * 20
        pid, qpct, qrpct = opt.step(0.5, 0.5, 0.5, 5, deltas)
        self.assertGreaterEqual(pid, 0.2)
        self.assertLessEqual(pid, 1.0)
        self.assertGreaterEqual(qpct, 0.2)
        self.assertLessEqual(qpct, 1.0)
        self.assertGreaterEqual(qrpct, 0.2)
        self.assertLessEqual(qrpct, 1.0)

    def test_no_plateau_when_deltas_vary(self):
        opt = hapsolo.RandomWalkOptimizer(0.2, 0.2, 0.2, 0.0001, 10, 1e-6)
        deltas = [0.1, 0.05, 0.02, 0.01, 0.005] + [0.0] * 15
        pid, qpct, qrpct = opt.step(0.5, 0.5, 0.5, 3, deltas)
        changed = (pid != 0.5) or (qpct != 0.5) or (qrpct != 0.5)
        self.assertTrue(changed)

    def test_normal_step_produces_change(self):
        opt = hapsolo.RandomWalkOptimizer(0.2, 0.2, 0.2, 0.01, 10, 1e-6)
        deltas = [0.1, 0.05, 0.02]
        pid, qpct, qrpct = opt.step(0.5, 0.5, 0.5, 3, deltas)
        self.assertTrue((pid != 0.5) or (qpct != 0.5) or (qrpct != 0.5))

    def test_two_out_of_bounds_combos(self):
        opt = hapsolo.RandomWalkOptimizer(0.2, 0.2, 0.2, 0.01, 10, 1e-6)
        deltas = [0.1] * 20
        # QPct and QRPct out of bounds
        pid, qpct, qrpct = opt.step(0.5, 1.5, 1.5, 5, deltas)
        self.assertGreaterEqual(qpct, 0.2)
        self.assertLessEqual(qpct, 1.0)
        self.assertGreaterEqual(qrpct, 0.2)
        self.assertLessEqual(qrpct, 1.0)

    def test_respects_min_bounds(self):
        opt = hapsolo.RandomWalkOptimizer(0.3, 0.4, 0.5, 0.01, 10, 1e-6)
        deltas = [0.0] * 20
        for _ in range(50):
            pid, qpct, qrpct = opt.step(1.5, 1.5, 1.5, 5, deltas)
            self.assertGreaterEqual(pid, 0.3)
            self.assertGreaterEqual(qpct, 0.4)
            self.assertGreaterEqual(qrpct, 0.5)


# ── SteepestDescentOptimizer ──────────────────────────────────────────────

class TestSteepestDescentOptimizer(unittest.TestCase):

    def _make_eval_fn(self, cost_map=None, default_cost=1.0):
        """Build an evaluate_fn that returns costs from a map or a default."""
        call_log = []
        def eval_fn(pid, qpct, qrpct):
            call_log.append((round(pid, 6), round(qpct, 6), round(qrpct, 6)))
            if cost_map:
                key = (round(pid, 6), round(qpct, 6), round(qrpct, 6))
                cost = cost_map.get(key, default_cost)
            else:
                cost = default_cost
            return cost, {'c1'}, {'c2'}, {'S': 10, 'D': 1, 'C': 11, 'F': 0, 'M': 0}
        return eval_fn, call_log

    def test_directions_has_eight_entries(self):
        self.assertEqual(len(hapsolo.SteepestDescentOptimizer.DIRECTIONS), 8)

    def test_all_directions_are_octant_combos(self):
        for d in hapsolo.SteepestDescentOptimizer.DIRECTIONS:
            self.assertEqual(len(d), 3)
            for v in d:
                self.assertIn(v, (+1, -1))

    def test_directions_are_unique(self):
        dirs = hapsolo.SteepestDescentOptimizer.DIRECTIONS
        self.assertEqual(len(dirs), len(set(dirs)))

    def test_picks_lowest_cost_neighbor(self):
        step = 0.01
        center = (0.5, 0.5, 0.5)
        best_dir = hapsolo.SteepestDescentOptimizer.DIRECTIONS[3]
        best_neighbor = (center[0] + best_dir[0]*step,
                         center[1] + best_dir[1]*step,
                         center[2] + best_dir[2]*step)
        cost_map = {(round(best_neighbor[0], 6),
                     round(best_neighbor[1], 6),
                     round(best_neighbor[2], 6)): 0.1}
        eval_fn, _ = self._make_eval_fn(cost_map=cost_map, default_cost=5.0)
        opt = hapsolo.SteepestDescentOptimizer(0.2, 0.2, 0.2, step, 10, 1e-6, eval_fn)
        deltas = [0.1] * 20
        pid, qpct, qrpct = opt.step(*center, 5, deltas)
        self.assertAlmostEqual(pid, best_neighbor[0], places=6)
        self.assertAlmostEqual(qpct, best_neighbor[1], places=6)
        self.assertAlmostEqual(qrpct, best_neighbor[2], places=6)

    def test_evaluates_all_eight_neighbors(self):
        eval_fn, call_log = self._make_eval_fn(default_cost=2.0)
        opt = hapsolo.SteepestDescentOptimizer(0.2, 0.2, 0.2, 0.01, 10, 1e-6, eval_fn)
        deltas = [0.1] * 20
        opt.step(0.5, 0.5, 0.5, 5, deltas)
        self.assertEqual(len(call_log), 8)

    def test_skips_out_of_bounds_neighbors(self):
        eval_fn, call_log = self._make_eval_fn(default_cost=2.0)
        opt = hapsolo.SteepestDescentOptimizer(0.2, 0.2, 0.2, 0.05, 10, 1e-6, eval_fn)
        deltas = [0.1] * 20
        opt.step(0.97, 0.97, 0.97, 5, deltas)
        for p, q, r in call_log:
            self.assertGreaterEqual(p, 0.2)
            self.assertLessEqual(p, 1.0)
            self.assertGreaterEqual(q, 0.2)
            self.assertLessEqual(q, 1.0)
            self.assertGreaterEqual(r, 0.2)
            self.assertLessEqual(r, 1.0)
        self.assertLess(len(call_log), 8)

    def test_caches_last_result(self):
        eval_fn, _ = self._make_eval_fn(default_cost=2.5)
        opt = hapsolo.SteepestDescentOptimizer(0.2, 0.2, 0.2, 0.01, 10, 1e-6, eval_fn)
        self.assertIsNone(opt.last_result)
        deltas = [0.1] * 20
        opt.step(0.5, 0.5, 0.5, 5, deltas)
        self.assertIsNotNone(opt.last_result)
        cost, contigs, purged, scores = opt.last_result
        self.assertEqual(cost, 2.5)
        self.assertIsInstance(contigs, set)
        self.assertIsInstance(scores, dict)

    def test_last_result_cleared_on_entry(self):
        eval_fn, _ = self._make_eval_fn(default_cost=1.0)
        opt = hapsolo.SteepestDescentOptimizer(0.2, 0.2, 0.2, 0.01, 10, 1e-6, eval_fn)
        opt.last_result = "stale"
        deltas = [0.1] * 20
        opt.step(0.5, 0.5, 0.5, 5, deltas)
        self.assertNotEqual(opt.last_result, "stale")

    def test_plateau_resets_all(self):
        eval_fn, call_log = self._make_eval_fn(default_cost=1.0)
        opt = hapsolo.SteepestDescentOptimizer(0.2, 0.2, 0.2, 0.01, 3, 1e-6, eval_fn)
        deltas = [0.0] * 20
        pid, qpct, qrpct = opt.step(0.5, 0.5, 0.5, 5, deltas)
        self.assertEqual(len(call_log), 0)
        self.assertIsNone(opt.last_result)
        self.assertGreaterEqual(pid, 0.2)
        self.assertLessEqual(pid, 1.0)
        self.assertGreaterEqual(qpct, 0.2)
        self.assertLessEqual(qpct, 1.0)
        self.assertGreaterEqual(qrpct, 0.2)
        self.assertLessEqual(qrpct, 1.0)

    def test_no_plateau_when_deltas_vary(self):
        eval_fn, call_log = self._make_eval_fn(default_cost=1.0)
        opt = hapsolo.SteepestDescentOptimizer(0.2, 0.2, 0.2, 0.01, 10, 1e-6, eval_fn)
        deltas = [0.5, 0.3, 0.1, 0.05] + [0.0] * 16
        opt.step(0.5, 0.5, 0.5, 3, deltas)
        self.assertGreater(len(call_log), 0)

    def test_all_out_of_bounds_resets(self):
        eval_fn, call_log = self._make_eval_fn(default_cost=1.0)
        opt = hapsolo.SteepestDescentOptimizer(0.2, 0.2, 0.2, 0.01, 10, 1e-6, eval_fn)
        deltas = [0.1] * 20
        pid, qpct, qrpct = opt.step(1.5, 1.5, 1.5, 1, deltas)
        self.assertEqual(len(call_log), 0)
        self.assertGreaterEqual(pid, 0.2)
        self.assertLessEqual(pid, 1.0)
        self.assertGreaterEqual(qpct, 0.2)
        self.assertLessEqual(qpct, 1.0)
        self.assertGreaterEqual(qrpct, 0.2)
        self.assertLessEqual(qrpct, 1.0)

    def test_single_pid_out_of_bounds(self):
        eval_fn, call_log = self._make_eval_fn(default_cost=1.0)
        opt = hapsolo.SteepestDescentOptimizer(0.2, 0.2, 0.2, 0.01, 10, 1e-6, eval_fn)
        deltas = [0.1] * 20
        pid, qpct, qrpct = opt.step(1.5, 0.5, 0.5, 5, deltas)
        self.assertGreaterEqual(pid, 0.2)
        self.assertLessEqual(pid, 1.0)
        self.assertEqual(len(call_log), 0)

    def test_two_out_of_bounds(self):
        eval_fn, call_log = self._make_eval_fn(default_cost=1.0)
        opt = hapsolo.SteepestDescentOptimizer(0.2, 0.2, 0.2, 0.01, 10, 1e-6, eval_fn)
        deltas = [0.1] * 20
        pid, qpct, qrpct = opt.step(0.5, 1.5, 1.5, 5, deltas)
        self.assertGreaterEqual(qpct, 0.2)
        self.assertLessEqual(qpct, 1.0)
        self.assertGreaterEqual(qrpct, 0.2)
        self.assertLessEqual(qrpct, 1.0)
        self.assertEqual(len(call_log), 0)

    def test_all_neighbors_oob_fallback(self):
        eval_fn, call_log = self._make_eval_fn(default_cost=1.0)
        opt = hapsolo.SteepestDescentOptimizer(0.99, 0.99, 0.99, 0.1, 10, 1e-6, eval_fn)
        deltas = [0.1] * 20
        pid, qpct, qrpct = opt.step(0.995, 0.995, 0.995, 5, deltas)
        self.assertEqual(len(call_log), 0)
        self.assertIsNone(opt.last_result)
        self.assertGreaterEqual(pid, 0.99)
        self.assertLessEqual(pid, 1.0)
        self.assertGreaterEqual(qpct, 0.99)
        self.assertLessEqual(qpct, 1.0)
        self.assertGreaterEqual(qrpct, 0.99)
        self.assertLessEqual(qrpct, 1.0)

    def test_respects_min_bounds(self):
        eval_fn, _ = self._make_eval_fn(default_cost=1.0)
        opt = hapsolo.SteepestDescentOptimizer(0.3, 0.4, 0.5, 0.01, 10, 1e-6, eval_fn)
        deltas = [0.0] * 20
        for _ in range(50):
            pid, qpct, qrpct = opt.step(1.5, 1.5, 1.5, 5, deltas)
            self.assertGreaterEqual(pid, 0.3)
            self.assertGreaterEqual(qpct, 0.4)
            self.assertGreaterEqual(qrpct, 0.5)

    def test_best_result_matches_returned_position(self):
        step = 0.01
        target_dir = (-1, -1, -1)
        center = (0.5, 0.5, 0.5)
        target = (center[0] + target_dir[0]*step,
                  center[1] + target_dir[1]*step,
                  center[2] + target_dir[2]*step)
        cost_map = {(round(target[0], 6),
                     round(target[1], 6),
                     round(target[2], 6)): 0.01}
        eval_fn, _ = self._make_eval_fn(cost_map=cost_map, default_cost=10.0)
        opt = hapsolo.SteepestDescentOptimizer(0.2, 0.2, 0.2, step, 10, 1e-6, eval_fn)
        deltas = [0.1] * 20
        pid, qpct, qrpct = opt.step(*center, 5, deltas)
        self.assertIsNotNone(opt.last_result)
        self.assertAlmostEqual(opt.last_result[0], 0.01, places=6)


# ── SteepestDescentOptimizer + hillclimbing integration ──────────────────

class TestSteepestDescentHillclimbing(unittest.TestCase):

    def setUp(self):
        hapsolo.myMinContigSize = 1000
        hapsolo.mypddf = pd.DataFrame({
            'qName': ['c1', 'c1', 'c2'],
            'tName': ['c2', 'c3', 'c3'],
            'qSize': [5000, 5000, 3000],
            'QPct':  [0.9, 0.8, 0.7],
            'PID':   [0.9, 0.8, 0.7],
            'QRAlignLenPct': [0.5, 0.6, 0.4],
        })
        hapsolo.allcontigsset = {'c1', 'c2', 'c3'}
        hapsolo.qrycontigset = {'c1', 'c2', 'c3'}
        hapsolo.missingrefcontigset = set()
        hapsolo.smallcontigset = set()
        hapsolo.busco2contigdict = {
            'B1': {'C': ['c1'], 'S': [], 'D': [], 'F': [], 'M': []},
            'B2': {'C': ['c2', 'c3'], 'S': [], 'D': [], 'F': [], 'M': []},
            'B3': {'C': [], 'S': [], 'D': [], 'F': ['c1'], 'M': []},
        }
        hapsolo.contigs2buscodict = {
            'c1': {'C': ['B1'], 'S': [], 'D': [], 'F': ['B3'], 'M': []},
            'c2': {'C': ['B2'], 'S': [], 'D': [], 'F': [], 'M': []},
            'c3': {'C': ['B2'], 'S': [], 'D': [], 'F': [], 'M': []},
        }
        hapsolo.thetaS = 1.0
        hapsolo.thetaD = 1.0
        hapsolo.thetaM = 1.0
        hapsolo.thetaF = 0.0
        hapsolo.bestnscores = 1
        hapsolo.maxzeros = 10
        hapsolo.mode = 2
        hapsolo.myMinPID = 0.2
        hapsolo.myMinQPctMin = 0.2
        hapsolo.myMinQRPctMin = 0.2

    def test_mode2_returns_expected_structure(self):
        job_args = [0, 5, 0.0001, 0.5, 0.5, 0.5]
        result = hapsolo.hillclimbing(job_args)
        bestnscoreslist, costfxn, costfxndelta = result
        self.assertIsInstance(bestnscoreslist, list)
        self.assertEqual(len(costfxn), 5)
        self.assertEqual(len(costfxndelta), 5)

    def test_mode2_best_entry_has_valid_score(self):
        job_args = [0, 10, 0.0001, 0.5, 0.5, 0.5]
        result = hapsolo.hillclimbing(job_args)
        bestnscoreslist = result[0]
        self.assertGreater(len(bestnscoreslist), 0)
        self.assertGreater(bestnscoreslist[0][0], 0)

    def test_mode2_upq_dedup(self):
        hapsolo.bestnscores = 3
        job_args = [0, 20, 0.0001, 0.5, 0.5, 0.5]
        result = hapsolo.hillclimbing(job_args)
        bestnscoreslist = result[0]
        contig_sets = [frozenset(entry[1]) for entry in bestnscoreslist]
        self.assertEqual(len(contig_sets), len(set(contig_sets)))

    def test_mode2_produces_results_comparable_to_mode0(self):
        hapsolo.mode = 0
        job_args = [0, 20, 0.0001, 0.5, 0.5, 0.5]
        result0 = hapsolo.hillclimbing(job_args)
        cost0 = result0[0][0][0]

        hapsolo.mode = 2
        job_args = [0, 20, 0.0001, 0.5, 0.5, 0.5]
        result2 = hapsolo.hillclimbing(job_args)
        cost2 = result2[0][0][0]

        self.assertGreater(cost0, 0)
        self.assertGreater(cost2, 0)
        self.assertLess(cost2, 50000000.0)

    def test_mode2_cost_not_worse_than_initial(self):
        job_args = [0, 15, 0.0001, 0.5, 0.5, 0.5]
        result = hapsolo.hillclimbing(job_args)
        best_cost = result[0][0][0]
        initial_cost, _, _, _ = hapsolo.evaluate_thresholds(0.5, 0.5, 0.5, 3)
        self.assertLessEqual(best_cost, initial_cost)


# ── BaseOptimizer interface ────────────────────────────────────────────────

class TestBaseOptimizer(unittest.TestCase):

    def test_step_not_implemented(self):
        opt = hapsolo.BaseOptimizer(0.2, 0.2, 0.2)
        with self.assertRaises(NotImplementedError):
            opt.step(0.5, 0.5, 0.5, 0, [])


# ── evaluate_thresholds ───────────────────────────────────────────────────

class TestEvaluateThresholds(unittest.TestCase):

    def setUp(self):
        hapsolo.myMinContigSize = 1000
        hapsolo.mypddf = pd.DataFrame({
            'qName': ['c1', 'c1', 'c2'],
            'tName': ['c2', 'c3', 'c3'],
            'qSize': [5000, 5000, 3000],
            'QPct':  [0.9, 0.8, 0.7],
            'PID':   [0.9, 0.8, 0.7],
            'QRAlignLenPct': [0.5, 0.6, 0.4],
        })
        hapsolo.allcontigsset = {'c1', 'c2', 'c3'}
        hapsolo.qrycontigset = {'c1', 'c2', 'c3'}
        hapsolo.missingrefcontigset = set()
        hapsolo.smallcontigset = set()
        hapsolo.busco2contigdict = {
            'B1': {'C': ['c1'], 'S': [], 'D': [], 'F': [], 'M': []},
            'B2': {'C': ['c2', 'c3'], 'S': [], 'D': [], 'F': [], 'M': []},
            'B3': {'C': [], 'S': [], 'D': [], 'F': ['c1'], 'M': []},
        }
        hapsolo.contigs2buscodict = {
            'c1': {'C': ['B1'], 'S': [], 'D': [], 'F': ['B3'], 'M': []},
            'c2': {'C': ['B2'], 'S': [], 'D': [], 'F': [], 'M': []},
            'c3': {'C': ['B2'], 'S': [], 'D': [], 'F': [], 'M': []},
        }
        hapsolo.thetaS = 1.0
        hapsolo.thetaD = 1.0
        hapsolo.thetaM = 1.0
        hapsolo.thetaF = 0.0

    def test_returns_four_tuple(self):
        cost, contigs, purged, scores = hapsolo.evaluate_thresholds(0.5, 0.5, 0.3, 3)
        self.assertIsInstance(cost, float)
        self.assertIsInstance(contigs, set)
        self.assertIsInstance(purged, set)
        self.assertIsInstance(scores, dict)

    def test_scores_have_busco_keys(self):
        _, _, _, scores = hapsolo.evaluate_thresholds(0.5, 0.5, 0.3, 3)
        for key in ['S', 'D', 'C', 'F', 'M']:
            self.assertIn(key, scores)

    def test_zero_singles_gives_penalty(self):
        hapsolo.busco2contigdict = {
            'B1': {'C': [], 'S': [], 'D': [], 'F': [], 'M': []},
        }
        hapsolo.contigs2buscodict = {
            'c1': {'C': [], 'S': [], 'D': [], 'F': [], 'M': []},
            'c2': {'C': [], 'S': [], 'D': [], 'F': [], 'M': []},
            'c3': {'C': [], 'S': [], 'D': [], 'F': [], 'M': []},
        }
        cost, _, _, _ = hapsolo.evaluate_thresholds(0.5, 0.5, 0.3, 1)
        self.assertEqual(cost, 50000000.0)

    def test_contigs_and_purged_are_disjoint(self):
        cost, contigs, purged, _ = hapsolo.evaluate_thresholds(0.5, 0.5, 0.3, 3)
        self.assertEqual(len(contigs & purged), 0)

    def test_contigs_union_purged_equals_all(self):
        cost, contigs, purged, _ = hapsolo.evaluate_thresholds(0.5, 0.5, 0.3, 3)
        self.assertEqual(contigs | purged, hapsolo.allcontigsset - {''})


# ── hillclimbing integration with new components ──────────────────────────

class TestHillclimbingIntegration(unittest.TestCase):

    def setUp(self):
        hapsolo.myMinContigSize = 1000
        hapsolo.mypddf = pd.DataFrame({
            'qName': ['c1', 'c1', 'c2'],
            'tName': ['c2', 'c3', 'c3'],
            'qSize': [5000, 5000, 3000],
            'QPct':  [0.9, 0.8, 0.7],
            'PID':   [0.9, 0.8, 0.7],
            'QRAlignLenPct': [0.5, 0.6, 0.4],
        })
        hapsolo.allcontigsset = {'c1', 'c2', 'c3'}
        hapsolo.qrycontigset = {'c1', 'c2', 'c3'}
        hapsolo.missingrefcontigset = set()
        hapsolo.smallcontigset = set()
        hapsolo.busco2contigdict = {
            'B1': {'C': ['c1'], 'S': [], 'D': [], 'F': [], 'M': []},
            'B2': {'C': ['c2', 'c3'], 'S': [], 'D': [], 'F': [], 'M': []},
            'B3': {'C': [], 'S': [], 'D': [], 'F': ['c1'], 'M': []},
        }
        hapsolo.contigs2buscodict = {
            'c1': {'C': ['B1'], 'S': [], 'D': [], 'F': ['B3'], 'M': []},
            'c2': {'C': ['B2'], 'S': [], 'D': [], 'F': [], 'M': []},
            'c3': {'C': ['B2'], 'S': [], 'D': [], 'F': [], 'M': []},
        }
        hapsolo.thetaS = 1.0
        hapsolo.thetaD = 1.0
        hapsolo.thetaM = 1.0
        hapsolo.thetaF = 0.0
        hapsolo.bestnscores = 1
        hapsolo.maxzeros = 10
        hapsolo.mode = 0
        hapsolo.myMinPID = 0.2
        hapsolo.myMinQPctMin = 0.2
        hapsolo.myMinQRPctMin = 0.2

    def test_returns_expected_structure(self):
        job_args = [0, 5, 0.0001, 0.5, 0.5, 0.5]
        result = hapsolo.hillclimbing(job_args)
        bestnscoreslist, costfxn, costfxndelta = result
        self.assertIsInstance(bestnscoreslist, list)
        self.assertEqual(len(costfxn), 5)
        self.assertEqual(len(costfxndelta), 5)

    def test_best_entry_has_valid_score(self):
        job_args = [0, 10, 0.0001, 0.5, 0.5, 0.5]
        result = hapsolo.hillclimbing(job_args)
        bestnscoreslist = result[0]
        self.assertGreater(len(bestnscoreslist), 0)
        self.assertGreater(bestnscoreslist[0][0], 0)

    def test_mode1_single_evaluation(self):
        hapsolo.mode = 1
        job_args = [0, 1, 0.0001, 0.7, 0.7, 0.7]
        result = hapsolo.hillclimbing(job_args)
        bestnscoreslist = result[0]
        self.assertEqual(len(bestnscoreslist), 1)

    def test_upq_dedup_in_hillclimbing(self):
        hapsolo.bestnscores = 3
        job_args = [0, 20, 0.0001, 0.5, 0.5, 0.5]
        result = hapsolo.hillclimbing(job_args)
        bestnscoreslist = result[0]
        contig_sets = [frozenset(entry[1]) for entry in bestnscoreslist]
        self.assertEqual(len(contig_sets), len(set(contig_sets)))


def _gpu_available():
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@unittest.skipUnless(_gpu_available(), 'requires CuPy and a CUDA GPU')
class TestGpuNativePlateauRestart(unittest.TestCase):
    """hillclimbing_gpu_native must restart walkers on a plateau, like the CPU classes.

    On a flat cost landscape every recorded delta is 0, so a walker's first plateau is at
    iteration maxzeros and it then restarts on every iteration: iterations - maxzeros restarts.
    """

    def _setup_flat(self, mode, maxzeros=10):
        import pandas as pd
        from hapsolo import optimizers
        # PID 0.1 is below every searchable threshold (>= 0.2), so no contig is ever removed
        # and every candidate has the same cost.
        df = pd.DataFrame({'qName': ['c1', 'c2'], 'tName': ['c2', 'c1'], 'qSize': [5000, 5000],
                           'QPct': [0.9, 0.9], 'PID': [0.1, 0.1], 'QRAlignLenPct': [1.0, 1.0]})
        b2c = {'B1': {'C': ['c1'], 'S': [], 'D': [], 'F': [], 'M': []},
               'B2': {'C': ['c2'], 'S': [], 'D': [], 'F': [], 'M': []}}
        c2b = {'c1': {'C': ['B1'], 'S': [], 'D': [], 'F': [], 'M': []},
               'c2': {'C': ['B2'], 'S': [], 'D': [], 'F': [], 'M': []}}
        allc = {'c1', 'c2'}
        optimizers.setup(df, allc, set(), set(), allc, b2c, c2b, 1.0, 1.0, 0.0, 1.0,
                         mode, 1, 0.2, 0.2, 0.2, 0.0001, maxzeros, 0.0001, use_gpu=True)
        return optimizers

    def _run(self, mode, iterations, n_walkers=4, maxzeros=10):
        import contextlib
        import io
        optimizers = self._setup_flat(mode, maxzeros)
        jobs = [[w, iterations, 0.0001, 0.5, 0.5, 0.5] for w in range(n_walkers)]
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            results = optimizers.hillclimbing_gpu_native(jobs)
        m = re.search(r'plateau restarts per walker: mean ([\d.]+), min (\d+), max (\d+)', err.getvalue())
        self.assertIsNotNone(m, err.getvalue())
        return results, float(m.group(1)), int(m.group(2)), int(m.group(3))

    def test_random_walk_restarts_on_plateau(self):
        _, mean, lo, hi = self._run(mode=0, iterations=50)
        self.assertEqual((lo, hi), (50 - 10, 50 - 10))

    def test_simulated_annealing_restarts_on_plateau(self):
        _, mean, lo, hi = self._run(mode=3, iterations=50)
        self.assertEqual((lo, hi), (50 - 10, 50 - 10))

    def test_maxzeros_controls_first_restart(self):
        _, _, lo, hi = self._run(mode=3, iterations=50, maxzeros=20)
        self.assertEqual((lo, hi), (50 - 20, 50 - 20))

    def test_no_restart_before_maxzeros(self):
        _, _, lo, hi = self._run(mode=0, iterations=10)
        self.assertEqual((lo, hi), (0, 0))

    def test_random_walk_steps_forward_only(self):
        """GPU mode 0 must never lower a threshold, like RandomWalkOptimizer.

        The only cost improvement is at PID <= 0.3 (removing c2, which duplicates B1); walkers
        start at PID 0.3002 with restarts disabled, so a forward-only walk can never reach it,
        while a walk that can step down finds it within a few iterations.
        """
        import pandas as pd
        from hapsolo import optimizers
        df = pd.DataFrame({'qName': ['c2'], 'tName': ['c1'], 'qSize': [5000],
                           'QPct': [0.99], 'PID': [0.3], 'QRAlignLenPct': [1.0]})
        b2c = {'B1': {'C': ['c1', 'c2'], 'S': [], 'D': [], 'F': [], 'M': []},
               'B2': {'C': ['c3'], 'S': [], 'D': [], 'F': [], 'M': []}}
        c2b = {'c1': {'C': ['B1'], 'S': [], 'D': [], 'F': [], 'M': []},
               'c2': {'C': ['B1'], 'S': [], 'D': [], 'F': [], 'M': []},
               'c3': {'C': ['B2'], 'S': [], 'D': [], 'F': [], 'M': []}}
        allc = {'c1', 'c2', 'c3'}
        optimizers.setup(df, allc, {'c1', 'c3'}, set(), {'c2'}, b2c, c2b, 1.0, 1.0, 0.0, 1.0,
                         0, 1, 0.2, 0.2, 0.2, 0.0001, 10 ** 6, 0.0001, use_gpu=True)
        start_cost = optimizers.evaluate_thresholds(0.3002, 0.5, 0.5, 2)[0]
        removed_cost = optimizers.evaluate_thresholds(0.25, 0.5, 0.5, 2)[0]
        self.assertLess(removed_cost, start_cost)  # the landscape is as described
        jobs = [[w, 200, 0.0001, 0.3002, 0.5, 0.5] for w in range(16)]
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()):
            results = optimizers.hillclimbing_gpu_native(jobs)
        for items, _, _ in results:
            self.assertAlmostEqual(items[0][0], start_cost)
            self.assertGreaterEqual(items[0][4][0], 0.3002 - 1e-6)

    def test_cost_history_recorded(self):
        """Native GPU path returns real per-iteration costs (it used to return all zeros)."""
        import contextlib
        import io
        import pandas as pd
        from hapsolo import optimizers
        # c1 and c2 both carry B1 (duplicated), so the cost is non-zero; PID 0.3 removes c2.
        df = pd.DataFrame({'qName': ['c2'], 'tName': ['c1'], 'qSize': [5000],
                           'QPct': [0.99], 'PID': [0.3], 'QRAlignLenPct': [1.0]})
        b2c = {'B1': {'C': ['c1', 'c2'], 'S': [], 'D': [], 'F': [], 'M': []},
               'B2': {'C': ['c3'], 'S': [], 'D': [], 'F': [], 'M': []}}
        c2b = {'c1': {'C': ['B1'], 'S': [], 'D': [], 'F': [], 'M': []},
               'c2': {'C': ['B1'], 'S': [], 'D': [], 'F': [], 'M': []},
               'c3': {'C': ['B2'], 'S': [], 'D': [], 'F': [], 'M': []}}
        allc = {'c1', 'c2', 'c3'}
        optimizers.setup(df, allc, {'c1', 'c3'}, set(), {'c2'}, b2c, c2b, 1.0, 1.0, 0.0, 1.0,
                         3, 1, 0.2, 0.2, 0.2, 0.0001, 10, 0.0001, use_gpu=True)
        jobs = [[w, 30, 0.0001, 0.5, 0.5, 0.5] for w in range(5)]
        with contextlib.redirect_stderr(io.StringIO()):
            results = optimizers.hillclimbing_gpu_native(jobs)
        for items, costs, deltas in results:
            self.assertEqual(len(costs), 30)
            self.assertEqual(len(deltas), 30)
            self.assertTrue(any(c != 0 for c in costs))
            self.assertAlmostEqual(deltas[0], costs[0])
            for i in range(1, 30):
                self.assertAlmostEqual(deltas[i], costs[i - 1] - costs[i])
            self.assertAlmostEqual(min(costs), items[0][0])

    def test_results_in_bounds_and_one_entry_per_walker(self):
        results, _, _, _ = self._run(mode=3, iterations=40, n_walkers=6)
        self.assertEqual(len(results), 6)
        for items, _, _ in results:
            p, q, r = items[0][4]
            for v in (p, q, r):
                self.assertTrue(0.2 <= v <= 1.0, v)


# ── SimulatedAnnealingOptimizer ──────────────────────────────────────────────

class TestSimulatedAnnealingOptimizer(unittest.TestCase):

    def _make(self, initial_temp=1.0, cooling_rate=0.99, min_temp=1e-6):
        return hapsolo.SimulatedAnnealingOptimizer(
            0.2, 0.2, 0.2, 0.01, 10, 1e-6,
            initial_temp, cooling_rate, min_temp)

    def test_all_out_of_bounds_resets(self):
        opt = self._make()
        deltas = [0.0] * 20
        pid, qpct, qrpct = opt.step(1.5, 1.5, 1.5, 1, deltas)
        self.assertGreaterEqual(pid, 0.2)
        self.assertLessEqual(pid, 1.0)
        self.assertGreaterEqual(qpct, 0.2)
        self.assertLessEqual(qpct, 1.0)
        self.assertGreaterEqual(qrpct, 0.2)
        self.assertLessEqual(qrpct, 1.0)

    def test_single_pid_out_of_bounds(self):
        opt = self._make()
        deltas = [0.1] * 20
        pid, qpct, qrpct = opt.step(1.5, 0.5, 0.5, 5, deltas)
        self.assertGreaterEqual(pid, 0.2)
        self.assertLessEqual(pid, 1.0)

    def test_normal_step_produces_change(self):
        opt = self._make()
        deltas = [0.1, 0.05, 0.02]
        pid, qpct, qrpct = opt.step(0.5, 0.5, 0.5, 3, deltas)
        self.assertTrue((pid != 0.5) or (qpct != 0.5) or (qrpct != 0.5))

    def test_temperature_decreases(self):
        opt = self._make(initial_temp=10.0, cooling_rate=0.9)
        deltas = [0.1] * 20
        opt.step(0.5, 0.5, 0.5, 1, deltas)
        t1 = opt.temperature
        opt.step(0.5, 0.5, 0.5, 2, deltas)
        t2 = opt.temperature
        self.assertLess(t2, t1)

    def test_temperature_floors_at_min(self):
        opt = self._make(initial_temp=1e-5, cooling_rate=0.001, min_temp=1e-6)
        deltas = [0.1] * 200
        for i in range(100):
            opt.step(0.5, 0.5, 0.5, i, deltas)
        self.assertGreaterEqual(opt.temperature, 1e-6)

    def test_accept_always_for_improvement(self):
        opt = self._make(initial_temp=0.0001, min_temp=1e-6)
        for _ in range(100):
            self.assertTrue(opt.accept(0.5))

    def test_accept_rejects_at_zero_temp(self):
        opt = self._make(initial_temp=1e-6, cooling_rate=0.001, min_temp=1e-6)
        deltas = [0.1] * 20
        for _ in range(100):
            opt.step(0.5, 0.5, 0.5, 1, deltas)
        rejections = sum(1 for _ in range(100) if not opt.accept(-0.1))
        self.assertEqual(rejections, 100)

    def test_accept_probabilistic_at_high_temp(self):
        opt = self._make(initial_temp=100.0, cooling_rate=0.9999)
        accepts = sum(1 for _ in range(1000) if opt.accept(-0.001))
        self.assertGreater(accepts, 900)

    def test_plateau_reheats(self):
        opt = self._make(initial_temp=10.0, cooling_rate=0.5)
        deltas = [0.0] * 20
        # Steps 0-9: no plateau (i < maxzeros=10), normal cooling
        for i in range(10):
            opt.step(0.5, 0.5, 0.5, i, deltas)
        t_before_reheat = opt.temperature
        self.assertLess(t_before_reheat, 0.1)  # cooled significantly
        # Step 10: first plateau detected, reheat without immediate re-cool
        opt.step(0.5, 0.5, 0.5, 10, deltas)
        self.assertAlmostEqual(opt.temperature, 10.0)  # full reheat
        self.assertEqual(opt._reheat_count, 1)

    def test_plateau_reheat_decays(self):
        """Successive plateaus produce progressively weaker reheats."""
        # maxzeros=5 so gaps of 5 non-zero deltas break the window
        opt = hapsolo.SimulatedAnnealingOptimizer(
            0.2, 0.2, 0.2, 0.01, 5, 1e-6,
            10.0, 0.9, 1e-6, reheat_decay=0.5)
        deltas = [0.0] * 200
        # First plateau at i=5 (indices 1-5 all zero)
        opt.step(0.5, 0.5, 0.5, 5, deltas)
        self.assertAlmostEqual(opt.temperature, 10.0)
        self.assertEqual(opt._reheat_count, 1)
        # Break with 5 non-zero deltas (must be >= maxzeros)
        for i in range(6, 11):
            deltas[i] = 1.0
            opt.step(0.5, 0.5, 0.5, i, deltas)
        # 5 zero deltas to trigger second plateau
        for i in range(11, 15):
            opt.step(0.5, 0.5, 0.5, i, deltas)
        opt.step(0.5, 0.5, 0.5, 15, deltas)
        self.assertAlmostEqual(opt.temperature, 5.0)  # 10.0 * 0.5^1
        self.assertEqual(opt._reheat_count, 2)
        # Break again with 5 non-zero
        for i in range(16, 21):
            deltas[i] = 1.0
            opt.step(0.5, 0.5, 0.5, i, deltas)
        # 5 zeros for third plateau
        for i in range(21, 25):
            opt.step(0.5, 0.5, 0.5, i, deltas)
        opt.step(0.5, 0.5, 0.5, 25, deltas)
        self.assertAlmostEqual(opt.temperature, 2.5)  # 10.0 * 0.5^2
        self.assertEqual(opt._reheat_count, 3)

    def test_respects_min_bounds(self):
        opt = hapsolo.SimulatedAnnealingOptimizer(
            0.3, 0.4, 0.5, 0.01, 10, 1e-6, 1.0, 0.99, 1e-6)
        deltas = [0.0] * 20
        for _ in range(50):
            pid, qpct, qrpct = opt.step(1.5, 1.5, 1.5, 5, deltas)
            self.assertGreaterEqual(pid, 0.3)
            self.assertGreaterEqual(qpct, 0.4)
            self.assertGreaterEqual(qrpct, 0.5)

    def test_two_out_of_bounds_combos(self):
        opt = self._make()
        deltas = [0.1] * 20
        pid, qpct, qrpct = opt.step(0.5, 1.5, 1.5, 5, deltas)
        self.assertGreaterEqual(qpct, 0.2)
        self.assertLessEqual(qpct, 1.0)
        self.assertGreaterEqual(qrpct, 0.2)
        self.assertLessEqual(qrpct, 1.0)

    def test_pid_and_qpct_out_of_bounds(self):
        opt = self._make()
        deltas = [0.1] * 20
        pid, qpct, qrpct = opt.step(1.5, 1.5, 0.5, 5, deltas)
        self.assertGreaterEqual(pid, 0.2)
        self.assertLessEqual(pid, 1.0)
        self.assertGreaterEqual(qpct, 0.2)
        self.assertLessEqual(qpct, 1.0)

    def test_pid_and_qrpct_out_of_bounds(self):
        opt = self._make()
        deltas = [0.1] * 20
        pid, qpct, qrpct = opt.step(1.5, 0.5, 1.5, 5, deltas)
        self.assertGreaterEqual(pid, 0.2)
        self.assertLessEqual(pid, 1.0)
        self.assertGreaterEqual(qrpct, 0.2)
        self.assertLessEqual(qrpct, 1.0)

    def test_single_qpct_out_of_bounds(self):
        opt = self._make()
        deltas = [0.1] * 20
        pid, qpct, qrpct = opt.step(0.5, 1.5, 0.5, 5, deltas)
        self.assertGreaterEqual(qpct, 0.2)
        self.assertLessEqual(qpct, 1.0)

    def test_single_qrpct_out_of_bounds(self):
        opt = self._make()
        deltas = [0.1] * 20
        pid, qpct, qrpct = opt.step(0.5, 0.5, 1.5, 5, deltas)
        self.assertGreaterEqual(qrpct, 0.2)
        self.assertLessEqual(qrpct, 1.0)


class TestSimulatedAnnealingHillclimbing(unittest.TestCase):

    def setUp(self):
        hapsolo.myMinContigSize = 1000
        hapsolo.mypddf = pd.DataFrame({
            'qName': ['c1', 'c1', 'c2'],
            'tName': ['c2', 'c3', 'c3'],
            'qSize': [5000, 5000, 3000],
            'QPct':  [0.9, 0.8, 0.7],
            'PID':   [0.9, 0.8, 0.7],
            'QRAlignLenPct': [0.5, 0.6, 0.4],
        })
        hapsolo.allcontigsset = {'c1', 'c2', 'c3'}
        hapsolo.qrycontigset = {'c1', 'c2', 'c3'}
        hapsolo.missingrefcontigset = set()
        hapsolo.smallcontigset = set()
        hapsolo.busco2contigdict = {
            'B1': {'C': ['c1'], 'S': [], 'D': [], 'F': [], 'M': []},
            'B2': {'C': ['c2', 'c3'], 'S': [], 'D': [], 'F': [], 'M': []},
            'B3': {'C': [], 'S': [], 'D': [], 'F': ['c1'], 'M': []},
        }
        hapsolo.contigs2buscodict = {
            'c1': {'C': ['B1'], 'S': [], 'D': [], 'F': ['B3'], 'M': []},
            'c2': {'C': ['B2'], 'S': [], 'D': [], 'F': [], 'M': []},
            'c3': {'C': ['B2'], 'S': [], 'D': [], 'F': [], 'M': []},
        }
        hapsolo.thetaS = 1.0
        hapsolo.thetaD = 1.0
        hapsolo.thetaM = 1.0
        hapsolo.thetaF = 0.0
        hapsolo.bestnscores = 1
        hapsolo.maxzeros = 10
        hapsolo.mode = 3
        hapsolo.myMinPID = 0.2
        hapsolo.myMinQPctMin = 0.2
        hapsolo.myMinQRPctMin = 0.2

    def test_mode3_returns_expected_structure(self):
        job_args = [0, 10, 0.0001, 0.5, 0.5, 0.5]
        result = hapsolo.hillclimbing(job_args)
        bestnscoreslist, costfxn, costfxndelta = result
        self.assertIsInstance(bestnscoreslist, list)
        self.assertEqual(len(costfxn), 10)
        self.assertEqual(len(costfxndelta), 10)

    def test_mode3_best_entry_has_valid_score(self):
        job_args = [0, 20, 0.0001, 0.5, 0.5, 0.5]
        result = hapsolo.hillclimbing(job_args)
        bestnscoreslist = result[0]
        self.assertGreater(len(bestnscoreslist), 0)
        self.assertGreater(bestnscoreslist[0][0], 0)

    def test_mode3_cost_history_populated(self):
        job_args = [0, 10, 0.0001, 0.5, 0.5, 0.5]
        result = hapsolo.hillclimbing(job_args)
        costfxn = result[1]
        self.assertTrue(any(c != 0 for c in costfxn))


if __name__ == '__main__':
    unittest.main()
