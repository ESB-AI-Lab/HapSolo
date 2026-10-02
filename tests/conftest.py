#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared pytest configuration for HapSolo tests.

Provides automatic global state isolation so individual test classes
don't need to manually save/restore hapsolo's module-level globals.
"""
import os
import sys
from copy import deepcopy
import pytest
import pandas as pd

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TESTS_DIR)
FIXTURES_DIR = os.path.join(TESTS_DIR, 'fixtures')

# Ensure hapsolo package is importable
sys.path.insert(0, PROJECT_DIR)
import hapsolo  # noqa: F401
from hapsolo import optimizers as _optimizers


# ── Global state snapshot/restore ────────────────────────────────────────────
# These are ALL the mutable globals in hapsolo.__init__ that functions can
# read or write.  Snapshot once before each test, restore after.

_GLOBAL_NAMES = [
    'busco2contigdict',
    'contigs2buscodict',
    'missingrefcontigset',
    'qrycontigset',
    'allcontigsset',
    'smallcontigset',
    'myContigsDict',
    'myscerrorlog',
    'myMinContigSize',
    'myMinPID',
    'myMinQPctMin',
    'myMinQRPctMin',
    'thetaS',
    'thetaD',
    'thetaM',
    'thetaF',
    'bestnscores',
    'maxzeros',
    'mode',
    'threads',
    'iterations',
    'stepsize',
    'resolution',
    'use_gpu',
    '_cost_formula_fn',
]

# Optimizer module-level state that needs snapshot/restore
_OPT_NAMES = [
    '_df', '_all_contigs', '_missing_ref', '_small_contigs', '_query_contigs',
    '_busco2contig', '_contigs2busco',
    '_theta_s', '_theta_d', '_theta_f', '_theta_m',
    '_mode', '_bestnscores', '_min_pid', '_min_qpct', '_min_qrpct',
    '_stepsize', '_maxzeros', '_resolution',
    '_scoring_data',
    '_use_gpu', '_gpu_converted',
    '_gpu_scoring_data', '_gpu_all_mask', '_gpu_missing_ref_mask', '_gpu_small_mask',
    '_gpu_n_contigs', '_gpu_id_to_contig', '_gpu_contig_to_id',
    '_gpu_pid_arr', '_gpu_qpct_arr', '_gpu_qrpct_arr', '_gpu_qnameid_arr',
    '_cost_formula_fn', '_cost_formula_str',
]


def _snapshot():
    """Deep-copy every mutable global to protect against nested mutations."""
    snap = {}
    for name in _GLOBAL_NAMES:
        val = getattr(hapsolo, name)
        if isinstance(val, (dict, set, list)):
            snap[name] = deepcopy(val)
        else:
            snap[name] = val
    snap['mypddf'] = hapsolo.mypddf.copy() if len(hapsolo.mypddf) > 0 else pd.DataFrame()
    # Also snapshot optimizer module state
    opt_snap = {}
    for name in _OPT_NAMES:
        val = getattr(_optimizers, name)
        if isinstance(val, (dict, set, list)):
            opt_snap[name] = deepcopy(val)
        elif isinstance(val, pd.DataFrame):
            opt_snap[name] = val.copy() if len(val) > 0 else pd.DataFrame()
        else:
            opt_snap[name] = val
    snap['_opt'] = opt_snap
    return snap


def _restore(snap):
    """Write snapshot values back to the modules."""
    opt_snap = snap.pop('_opt', {})
    for name, val in snap.items():
        setattr(hapsolo, name, val)
    for name, val in opt_snap.items():
        setattr(_optimizers, name, val)


@pytest.fixture(autouse=True)
def isolate_hapsolo_globals():
    """Auto-fixture: saves all hapsolo globals before each test, restores after."""
    snap = _snapshot()
    yield
    _restore(snap)
