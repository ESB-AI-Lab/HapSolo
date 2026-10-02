import sys
from math import exp, log
from random import randint, uniform

import numpy as np

from hapsolo.alignment import reduce_asm
from hapsolo.scoring import cost_function
from hapsolo.scoring_data import ScoringData
from hapsolo.utils import CalculateInverseProportion

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ---------------------------------------------------------------------------
# Module-level shared state for multiprocessing fork inheritance.
# __main__.py calls setup() before spawning the pool; forked workers inherit
# these values via copy-on-write.
# ---------------------------------------------------------------------------
_df = None
_all_contigs = set()
_missing_ref = set()
_small_contigs = set()
_query_contigs = set()
_busco2contig = {}
_contigs2busco = {}
_theta_s = 1.0
_theta_d = 1.0
_theta_f = 0.0
_theta_m = 1.0
_mode = 0
_bestnscores = 1
_min_pid = 0.2
_min_qpct = 0.2
_min_qrpct = 0.2
_stepsize = 0.0001
_maxzeros = 10
_resolution = 0.0001
_scoring_data = None
_use_gpu = False
_gpu_converted = False
_sa_initial_temp = None
_sa_cooling_rate = None
_sa_min_temp = 1e-6
_cost_formula_fn = None
_cost_formula_str = None

# CPU numpy arrays for fast filtering (populated by setup)
_np_pid_arr = None
_np_qpct_arr = None
_np_qrpct_arr = None
_np_qnameid_arr = None
_np_all_mask = None
_np_missing_ref_mask = None
_np_small_mask = None
_np_n_contigs = 0
_np_id_to_contig = None
_np_contig_to_id = None
_np_converted = False

# GPU-specific state (populated by _ensure_gpu after fork)
_gpu_scoring_data = None
_gpu_all_mask = None
_gpu_missing_ref_mask = None
_gpu_small_mask = None
_gpu_n_contigs = 0
_gpu_id_to_contig = None
_gpu_contig_to_id = None
_gpu_scatter_matrix = None
_gpu_pid_arr = None
_gpu_qpct_arr = None
_gpu_qrpct_arr = None
_gpu_qnameid_arr = None


def setup(df, all_contigs, missing_ref, small_contigs, query_contigs,
          busco2contig, contigs2busco,
          theta_s, theta_d, theta_f, theta_m,
          mode, bestnscores, min_pid, min_qpct, min_qrpct,
          stepsize, maxzeros, resolution, use_gpu=False,
          sa_initial_temp=None, sa_cooling_rate=None, sa_min_temp=1e-6,
          cost_formula_fn=None, cost_formula_str=None):
    """Set module-level state before multiprocessing pool creation."""
    global _df, _all_contigs, _missing_ref, _small_contigs, _query_contigs
    global _busco2contig, _contigs2busco
    global _theta_s, _theta_d, _theta_f, _theta_m
    global _mode, _bestnscores, _min_pid, _min_qpct, _min_qrpct
    global _stepsize, _maxzeros, _resolution
    global _scoring_data
    global _use_gpu, _gpu_converted
    global _sa_initial_temp, _sa_cooling_rate, _sa_min_temp
    global _cost_formula_fn, _cost_formula_str
    _df = df
    _all_contigs = all_contigs
    _missing_ref = missing_ref
    _small_contigs = small_contigs
    _query_contigs = query_contigs
    _busco2contig = busco2contig
    _contigs2busco = contigs2busco
    _theta_s = theta_s
    _theta_d = theta_d
    _theta_f = theta_f
    _theta_m = theta_m
    _mode = mode
    _bestnscores = bestnscores
    _min_pid = min_pid
    _min_qpct = min_qpct
    _min_qrpct = min_qrpct
    _stepsize = stepsize
    _maxzeros = maxzeros
    _resolution = resolution
    _use_gpu = use_gpu
    _gpu_converted = False
    _sa_initial_temp = sa_initial_temp
    _sa_cooling_rate = sa_cooling_rate
    _sa_min_temp = sa_min_temp
    _cost_formula_fn = cost_formula_fn
    _cost_formula_str = cost_formula_str
    _scoring_data = ScoringData.from_busco_dicts(busco2contig, contigs2busco)

    global _np_pid_arr, _np_qpct_arr, _np_qrpct_arr, _np_qnameid_arr
    global _np_all_mask, _np_missing_ref_mask, _np_small_mask
    global _np_n_contigs, _np_id_to_contig, _np_contig_to_id, _np_converted

    sorted_contigs = sorted(all_contigs)
    _np_contig_to_id = {name: i for i, name in enumerate(sorted_contigs)}
    _np_id_to_contig = sorted_contigs
    _np_n_contigs = len(sorted_contigs)

    qname_ids = df['qName'].map(_np_contig_to_id)
    valid = qname_ids.notna()
    _np_pid_arr = df.loc[valid, 'PID'].values.astype(np.float32)
    _np_qpct_arr = df.loc[valid, 'QPct'].values.astype(np.float32)
    _np_qrpct_arr = df.loc[valid, 'QRAlignLenPct'].values.astype(np.float32)
    _np_qnameid_arr = qname_ids[valid].values.astype(np.int32)

    _np_all_mask = np.zeros(_np_n_contigs, dtype=bool)
    for name in all_contigs:
        cid = _np_contig_to_id.get(name)
        if cid is not None:
            _np_all_mask[cid] = True

    _np_missing_ref_mask = np.zeros(_np_n_contigs, dtype=bool)
    for name in missing_ref:
        cid = _np_contig_to_id.get(name)
        if cid is not None:
            _np_missing_ref_mask[cid] = True

    _np_small_mask = np.zeros(_np_n_contigs, dtype=bool)
    for name in small_contigs:
        cid = _np_contig_to_id.get(name)
        if cid is not None:
            _np_small_mask[cid] = True

    _np_converted = True


class UniquePriorityQueue:
    """Bounded priority queue that deduplicates entries with identical contig sets.

    Entries are [score, contig_set, purged_set, busco_scores, thresholds].
    Lower score = better.
    """

    def __init__(self, max_size):
        self.max_size = max_size
        self._items = []

    def add(self, entry):
        items = self._items[:]
        items.append(entry)
        items.sort(key=lambda x: x[0])
        changed = True
        while changed:
            changed = False
            size_groups = {}
            for i, item in enumerate(items):
                sz = len(item[1])
                if sz not in size_groups:
                    size_groups[sz] = []
                size_groups[sz].append(i)
            for indices in size_groups.values():
                if len(indices) < 2:
                    continue
                for j in range(len(indices)):
                    for k in range(j + 1, len(indices)):
                        if items[indices[j]][1] == items[indices[k]][1]:
                            items.pop(indices[k])
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
        self._items = items[:self.max_size]

    def should_add(self, score):
        return len(self._items) < self.max_size or score <= self._items[-1][0]

    @property
    def items(self):
        return self._items

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


def uniquepriorityqueue(pqlist, myvalue):
    """Backward-compatible wrapper around UniquePriorityQueue."""
    upq = UniquePriorityQueue(_bestnscores)
    for item in pqlist:
        upq.add(item)
    upq.add(myvalue)
    return upq.items


class BaseOptimizer:
    """Base class for threshold optimizers."""

    def __init__(self, min_pid, min_qpct, min_qrpct):
        self.min_pid = min_pid
        self.min_qpct = min_qpct
        self.min_qrpct = min_qrpct

    def step(self, pid, qpct, qrpct, iteration, cost_deltas):
        raise NotImplementedError


class RandomWalkOptimizer(BaseOptimizer):
    """Random walk optimizer with plateau detection and boundary resets."""

    def __init__(self, min_pid, min_qpct, min_qrpct, step_size, maxzeros, resolution):
        super().__init__(min_pid, min_qpct, min_qrpct)
        self.step_size = step_size
        self.maxzeros = maxzeros
        self.resolution = resolution
        self._steps = [0, 0, 0]

    def step(self, pid, qpct, qrpct, iteration, cost_deltas):
        if iteration >= 2 and cost_deltas[iteration - 1] < 0.0 and cost_deltas[iteration - 2] < 0.0:
            while self._steps[0] != 0 or self._steps[1] != 0:
                for j in range(3):
                    self._steps[j] = self.step_size * randint(0, len(self._steps) - 1)

        plateau = (iteration >= self.maxzeros and
                   all(abs(cost_deltas[k]) <= self.resolution
                       for k in range(iteration - self.maxzeros + 1, iteration + 1)))

        out_pid = pid > 1.0
        out_qpct = qpct > 1.0
        out_qrpct = qrpct > 1.0

        if (out_pid and out_qpct and out_qrpct) or plateau:
            pid = uniform(self.min_pid, 1.0)
            qpct = uniform(self.min_qpct, 1.0)
            qrpct = uniform(self.min_qrpct, 1.0)
        elif out_qpct and out_qrpct:
            qpct = uniform(self.min_qpct, 1.0)
            qrpct = uniform(self.min_qrpct, 1.0)
        elif out_pid and out_qpct:
            pid = uniform(self.min_pid, 1.0)
            qpct = uniform(self.min_qpct, 1.0)
        elif out_pid and out_qrpct:
            pid = uniform(self.min_pid, 1.0)
            qrpct = uniform(self.min_qrpct, 1.0)
        elif out_pid:
            pid = uniform(self.min_pid, 1.0)
        elif out_qpct:
            qpct = uniform(self.min_qpct, 1.0)
        elif out_qrpct:
            qrpct = uniform(self.min_qrpct, 1.0)
        else:
            self._steps[randint(0, 2)] = self.step_size
            while True:
                for j in range(3):
                    self._steps[j] = self.step_size * randint(0, len(self._steps) - 1)
                if self._steps[0] != 0.0 or self._steps[1] != 0.0 or self._steps[2] != 0.0:
                    break
            pid += self._steps[0]
            qpct += self._steps[1]
            qrpct += self._steps[2]

        return pid, qpct, qrpct


class SteepestDescentOptimizer(BaseOptimizer):
    """Steepest-descent 8-neighbor optimizer with plateau detection and boundary resets."""

    DIRECTIONS = [
        (+1, +1, +1), (+1, +1, -1), (+1, -1, +1), (+1, -1, -1),
        (-1, +1, +1), (-1, +1, -1), (-1, -1, +1), (-1, -1, -1),
    ]

    def __init__(self, min_pid, min_qpct, min_qrpct, step_size, maxzeros,
                 resolution, evaluate_fn):
        super().__init__(min_pid, min_qpct, min_qrpct)
        self.step_size = step_size
        self.maxzeros = maxzeros
        self.resolution = resolution
        self._evaluate = evaluate_fn
        self.last_result = None

    def step(self, pid, qpct, qrpct, iteration, cost_deltas):
        self.last_result = None

        plateau = (iteration >= self.maxzeros and
                   all(abs(cost_deltas[k]) <= self.resolution
                       for k in range(iteration - self.maxzeros + 1, iteration + 1)))

        out_pid = pid > 1.0
        out_qpct = qpct > 1.0
        out_qrpct = qrpct > 1.0

        if (out_pid and out_qpct and out_qrpct) or plateau:
            return uniform(self.min_pid, 1.0), uniform(self.min_qpct, 1.0), uniform(self.min_qrpct, 1.0)
        if out_qpct and out_qrpct:
            return pid, uniform(self.min_qpct, 1.0), uniform(self.min_qrpct, 1.0)
        if out_pid and out_qpct:
            return uniform(self.min_pid, 1.0), uniform(self.min_qpct, 1.0), qrpct
        if out_pid and out_qrpct:
            return uniform(self.min_pid, 1.0), qpct, uniform(self.min_qrpct, 1.0)
        if out_pid:
            return uniform(self.min_pid, 1.0), qpct, qrpct
        if out_qpct:
            return pid, uniform(self.min_qpct, 1.0), qrpct
        if out_qrpct:
            return pid, qpct, uniform(self.min_qrpct, 1.0)

        s = self.step_size
        best_cost = float('inf')
        best_pos = (pid, qpct, qrpct)
        best_eval = None

        for dp, dq, dr in self.DIRECTIONS:
            np_ = pid + dp * s
            nq = qpct + dq * s
            nr = qrpct + dr * s
            if not (self.min_pid <= np_ <= 1.0 and
                    self.min_qpct <= nq <= 1.0 and
                    self.min_qrpct <= nr <= 1.0):
                continue
            cost, contigs, purged, scores = self._evaluate(np_, nq, nr)
            if cost < best_cost:
                best_cost = cost
                best_pos = (np_, nq, nr)
                best_eval = (cost, contigs, purged, scores)

        if best_eval is not None:
            self.last_result = best_eval
            return best_pos

        return uniform(self.min_pid, 1.0), uniform(self.min_qpct, 1.0), uniform(self.min_qrpct, 1.0)


class SimulatedAnnealingOptimizer(BaseOptimizer):
    """Simulated annealing optimizer for threshold search.

    Dedicated to the memory of Dr. Richard Lathrop, who advised on
    the development of this optimization approach.
    """

    def __init__(self, min_pid, min_qpct, min_qrpct, step_size,
                 maxzeros, resolution, initial_temp, cooling_rate, min_temp,
                 reheat_decay=0.5):
        super().__init__(min_pid, min_qpct, min_qrpct)
        self.step_size = step_size
        self.maxzeros = maxzeros
        self.resolution = resolution
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.reheat_decay = reheat_decay
        self._temp = initial_temp
        self._reheat_count = 0
        self._steps = [0, 0, 0]

    def step(self, pid, qpct, qrpct, iteration, cost_deltas):
        """Propose a candidate position via random perturbation."""
        plateau = (iteration >= self.maxzeros and
                   all(abs(cost_deltas[k]) <= self.resolution
                       for k in range(iteration - self.maxzeros + 1, iteration + 1)))

        if plateau:
            reheat_temp = self.initial_temp * (self.reheat_decay ** self._reheat_count)
            self._temp = max(self.min_temp, reheat_temp)
            self._reheat_count += 1
            return (uniform(self.min_pid, 1.0),
                    uniform(self.min_qpct, 1.0),
                    uniform(self.min_qrpct, 1.0))

        out_pid = pid > 1.0
        out_qpct = qpct > 1.0
        out_qrpct = qrpct > 1.0

        if out_pid and out_qpct and out_qrpct:
            self._temp = max(self.min_temp, self._temp * self.cooling_rate)
            return (uniform(self.min_pid, 1.0),
                    uniform(self.min_qpct, 1.0),
                    uniform(self.min_qrpct, 1.0))
        if out_qpct and out_qrpct:
            self._temp = max(self.min_temp, self._temp * self.cooling_rate)
            return pid, uniform(self.min_qpct, 1.0), uniform(self.min_qrpct, 1.0)
        if out_pid and out_qpct:
            self._temp = max(self.min_temp, self._temp * self.cooling_rate)
            return uniform(self.min_pid, 1.0), uniform(self.min_qpct, 1.0), qrpct
        if out_pid and out_qrpct:
            self._temp = max(self.min_temp, self._temp * self.cooling_rate)
            return uniform(self.min_pid, 1.0), qpct, uniform(self.min_qrpct, 1.0)
        if out_pid:
            self._temp = max(self.min_temp, self._temp * self.cooling_rate)
            return uniform(self.min_pid, 1.0), qpct, qrpct
        if out_qpct:
            self._temp = max(self.min_temp, self._temp * self.cooling_rate)
            return pid, uniform(self.min_qpct, 1.0), qrpct
        if out_qrpct:
            self._temp = max(self.min_temp, self._temp * self.cooling_rate)
            return pid, qpct, uniform(self.min_qrpct, 1.0)

        while True:
            for j in range(3):
                self._steps[j] = self.step_size * randint(0, 2)
            if self._steps[0] != 0.0 or self._steps[1] != 0.0 or self._steps[2] != 0.0:
                break

        self._temp = max(self.min_temp, self._temp * self.cooling_rate)
        return pid + self._steps[0], qpct + self._steps[1], qrpct + self._steps[2]

    @property
    def temperature(self):
        return self._temp

    def accept(self, cost_delta):
        """Metropolis criterion: accept if improved or with probability exp(delta/T).

        cost_delta = old_cost - new_cost (positive = improvement).
        """
        if cost_delta >= 0:
            return True
        T = self._temp
        if T <= self.min_temp:
            return False
        prob = exp(cost_delta / T)
        return uniform(0, 1) < prob


def _auto_calibrate_temperature(total_buscos, n_samples=10):
    """Estimate initial SA temperature from cost landscape variance.

    Runs n_samples random evaluations, computes the standard deviation of
    consecutive cost deltas, and sets T0 so that a 1-sigma worsening has
    ~80% acceptance probability: T0 = -sigma / ln(0.8).
    """
    costs = []
    for _ in range(n_samples):
        p = uniform(_min_pid, 1.0)
        q = uniform(_min_qpct, 1.0)
        r = uniform(_min_qrpct, 1.0)
        cost, _, _, _ = evaluate_thresholds(p, q, r, total_buscos)
        costs.append(cost)
    deltas = [abs(costs[i] - costs[i - 1]) for i in range(1, len(costs))]
    if not deltas:
        return 1.0
    mean_d = sum(deltas) / len(deltas)
    sigma = (sum((d - mean_d) ** 2 for d in deltas) / len(deltas)) ** 0.5
    if sigma <= 0:
        return 1.0
    return -sigma / log(0.8)


def _evaluate_cost_gpu(pid, qpct, qrpct, total_buscos):
    """GPU fast path: compute cost and scores without materializing contig sets.

    Uses raw cupy arrays for filtering (single fused boolean op). Returns
    (cost, scores, good_mask). The mask stays
    on GPU; call _materialize_contigs(good_mask) only when needed.
    """
    import cupy as cp

    qrpct_max = CalculateInverseProportion(qrpct)
    hit_mask = ((_gpu_pid_arr >= pid) &
                (_gpu_qpct_arr >= qpct) &
                (_gpu_qrpct_arr >= qrpct) &
                (_gpu_qrpct_arr <= qrpct_max))

    if cp.any(hit_mask):
        removed_ids = cp.unique(_gpu_qnameid_arr[hit_mask])
        removal_mask = cp.zeros(_gpu_n_contigs, dtype=bool)
        removal_mask[removed_ids] = True
    else:
        removal_mask = cp.zeros(_gpu_n_contigs, dtype=bool)

    good_mask = (_gpu_all_mask & ~removal_mask) | _gpu_missing_ref_mask
    good_mask = good_mask & ~_gpu_small_mask

    scores = _gpu_scoring_data.calculate_scores_from_mask(good_mask)

    if scores['S'] == 0:
        cost = 50000000.0
    else:
        cost = cost_function(scores['M'], scores['S'], scores['D'], scores['F'],
                             total_buscos, _theta_s, _theta_d, _theta_f, _theta_m,
                             formula_fn=_cost_formula_fn)

    return cost, scores, good_mask


def _materialize_contigs(good_mask):
    """Transfer GPU mask to CPU contig sets. Only call when result is needed."""
    import cupy as cp
    good_ids = cp.where(good_mask)[0].get()
    good_contigs = set(_gpu_id_to_contig[i] for i in good_ids)
    purged_contigs = _all_contigs - good_contigs - {''}
    return good_contigs, purged_contigs


def _evaluate_thresholds_gpu(pid, qpct, qrpct, total_buscos):
    """GPU path: cupy filtering + scoring with full materialization."""
    cost, scores, good_mask = _evaluate_cost_gpu(pid, qpct, qrpct, total_buscos)
    good_contigs, purged_contigs = _materialize_contigs(good_mask)
    return cost, good_contigs, purged_contigs, scores


def evaluate_thresholds(pid, qpct, qrpct, total_buscos):
    """Evaluate filter thresholds against module-level alignment data.

    Returns (cost, good_contigs, purged_contigs, busco_scores).
    Dispatches to GPU path when --gpu is enabled.
    """
    if _use_gpu and _gpu_converted:
        return _evaluate_thresholds_gpu(pid, qpct, qrpct, total_buscos)

    if _np_converted:
        qrpct_max = CalculateInverseProportion(qrpct)
        hit_mask = ((_np_pid_arr >= pid) &
                    (_np_qpct_arr >= qpct) &
                    (_np_qrpct_arr >= qrpct) &
                    (_np_qrpct_arr <= qrpct_max))
        removal_mask = np.zeros(_np_n_contigs, dtype=bool)
        if np.any(hit_mask):
            removed_ids = np.unique(_np_qnameid_arr[hit_mask])
            removal_mask[removed_ids] = True
        good_mask = (_np_all_mask & ~removal_mask) | _np_missing_ref_mask
        good_mask = good_mask & ~_np_small_mask
        mygoodcontigs = {_np_id_to_contig[i] for i in np.where(good_mask)[0]}
        purgedcontigs = _all_contigs - mygoodcontigs - {''}
        scores = _scoring_data.calculate_scores(mygoodcontigs)
    else:
        mygoodcontigs = reduce_asm(_df, pid, qpct, qrpct, _all_contigs)
        mygoodcontigs = mygoodcontigs.union(_missing_ref) - _small_contigs - {''}
        purgedcontigs = _all_contigs - mygoodcontigs - {''}
        scores = _scoring_data.calculate_scores(mygoodcontigs)

    if scores['S'] == 0:
        cost = 50000000.0
    else:
        cost = cost_function(scores['M'], scores['S'], scores['D'], scores['F'],
                             total_buscos, _theta_s, _theta_d, _theta_f, _theta_m,
                             formula_fn=_cost_formula_fn)
    return cost, mygoodcontigs, purgedcontigs, scores


def _ensure_gpu():
    """Lazy GPU setup — called once per worker process after fork.

    Builds integer encoding, extracts alignment columns as cupy arrays,
    creates cupy boolean masks, and builds ScoringData on GPU. CUDA
    initializes here, safely after fork.
    """
    global _gpu_converted
    global _gpu_scoring_data, _gpu_all_mask, _gpu_missing_ref_mask, _gpu_small_mask
    global _gpu_n_contigs, _gpu_id_to_contig, _gpu_contig_to_id
    global _gpu_pid_arr, _gpu_qpct_arr, _gpu_qrpct_arr, _gpu_qnameid_arr
    global _gpu_scatter_matrix
    if _use_gpu and not _gpu_converted:
        import cupy as cp
        from hapsolo.scoring_data import ScoringData

        sorted_contigs = sorted(_all_contigs)
        _gpu_contig_to_id = {name: i for i, name in enumerate(sorted_contigs)}
        _gpu_id_to_contig = sorted_contigs
        _gpu_n_contigs = len(sorted_contigs)

        qname_ids = _df['qName'].map(_gpu_contig_to_id)
        valid = qname_ids.notna()
        _gpu_pid_arr = cp.asarray(_df.loc[valid, 'PID'].values, dtype=cp.float32)
        _gpu_qpct_arr = cp.asarray(_df.loc[valid, 'QPct'].values, dtype=cp.float32)
        _gpu_qrpct_arr = cp.asarray(_df.loc[valid, 'QRAlignLenPct'].values, dtype=cp.float32)
        _gpu_qnameid_arr = cp.asarray(qname_ids[valid].values.astype('int32'))

        _gpu_all_mask = cp.zeros(_gpu_n_contigs, dtype=bool)
        for name in _all_contigs:
            cid = _gpu_contig_to_id.get(name)
            if cid is not None:
                _gpu_all_mask[cid] = True

        _gpu_missing_ref_mask = cp.zeros(_gpu_n_contigs, dtype=bool)
        for name in _missing_ref:
            cid = _gpu_contig_to_id.get(name)
            if cid is not None:
                _gpu_missing_ref_mask[cid] = True

        _gpu_small_mask = cp.zeros(_gpu_n_contigs, dtype=bool)
        for name in _small_contigs:
            cid = _gpu_contig_to_id.get(name)
            if cid is not None:
                _gpu_small_mask[cid] = True

        _gpu_scoring_data = ScoringData.from_busco_dicts(
            _busco2contig, _contigs2busco, _gpu_contig_to_id, _gpu_id_to_contig)
        _gpu_scoring_data.to_gpu()

        from cupyx.scipy import sparse as cusp
        A_aln = len(_gpu_qnameid_arr)
        _gpu_scatter_matrix = cusp.csr_matrix(
            (cp.ones(A_aln, dtype=cp.float32),
             (_gpu_qnameid_arr.astype(cp.int32),
              cp.arange(A_aln, dtype=cp.int32))),
            shape=(_gpu_n_contigs, A_aln))
        _gpu_scoring_data.prepare_batch_fast()

        _gpu_converted = True


def hillclimbing(job_args):
    """Run hill climbing optimization for one thread.

    job_args: [thread_id, num_iterations, resolution, pid, qpct, qrpct]
    Reads module-level state set by setup().
    """
    _ensure_gpu()

    mythread = job_args[0]
    numofiterations = job_args[1]
    res = job_args[2]
    myPID = job_args[3]
    myQPctMin = job_args[4]
    myQRPctMin = job_args[5]

    pbar = None
    if HAS_TQDM and _mode != 1:
        pbar = tqdm(
            total=numofiterations,
            position=mythread,
            desc='JOBID: ' + str(mythread),
            bar_format='{desc} [{bar:30}] {n_fmt}/{total_fmt} {postfix}',
            leave=True,
            dynamic_ncols=True,
            mininterval=0.2,
            miniters=1,
        )
        pbar.set_postfix_str(
            'PID: ' + ('%.4f' % myPID)
            + ' QPMin: ' + ('%.4f' % myQPctMin)
            + ' QRPMin: ' + ('%.4f' % myQRPctMin)
            + ' CostΔ ' + ('%+.4f' % 0.0)
            + ' Score: ' + ('%.4f' % 0.0))

    costfxn = [0.0] * numofiterations
    costfxndelta = [0.0] * numofiterations

    allmycontigs = _query_contigs.union(_missing_ref) - _small_contigs - {''}
    allcontigsbuscoscore = _scoring_data.calculate_scores(allmycontigs)
    totalbuscos = allcontigsbuscoscore['C'] + allcontigsbuscoscore['M'] + allcontigsbuscoscore['F']
    if allcontigsbuscoscore['S'] == 0:
        oldasmscorefxn = 5000.0
    else:
        oldasmscorefxn = cost_function(allcontigsbuscoscore['M'], allcontigsbuscoscore['S'],
                                       allcontigsbuscoscore['D'], allcontigsbuscoscore['F'],
                                       totalbuscos, _theta_s, _theta_d, _theta_f, _theta_m,
                                       formula_fn=_cost_formula_fn)

    upq = UniquePriorityQueue(_bestnscores)
    bestcontigset = allmycontigs.copy()
    bestpurgedset = _all_contigs - bestcontigset - {''}
    if _mode != 1:
        upq.add([oldasmscorefxn, bestcontigset, bestpurgedset,
                 allcontigsbuscoscore.copy(), [0.0, 0.0, 0.0]])

    _gpu_fast = _use_gpu and _gpu_converted

    cost, contigs, purged, scores = evaluate_thresholds(myPID, myQPctMin, myQRPctMin, totalbuscos)
    costfxn[0] = cost
    costfxndelta[0] = cost
    upq.add([cost, contigs, purged, scores, [myPID, myQPctMin, myQRPctMin]])

    if pbar is not None:
        pbar.set_postfix_str(
            'PID: ' + ('%.4f' % myPID)
            + ' QPMin: ' + ('%.4f' % myQPctMin)
            + ' QRPMin: ' + ('%.4f' % myQRPctMin)
            + ' CostΔ ' + ('%+.4f' % costfxndelta[0])
            + ' Score: ' + ('%.4f' % cost))
        pbar.update(1)

    if _mode == 1:
        if pbar is not None:
            pbar.close()
        return [upq.items, costfxn, costfxndelta]

    if _mode == 2:
        if _gpu_fast:
            def eval_fn(p, q, r):
                c, sc, mask = _evaluate_cost_gpu(p, q, r, totalbuscos)
                gc, pc = _materialize_contigs(mask)
                return c, gc, pc, sc
        else:
            eval_fn = lambda p, q, r: evaluate_thresholds(p, q, r, totalbuscos)
        optimizer = SteepestDescentOptimizer(_min_pid, _min_qpct, _min_qrpct,
                                             _stepsize, _maxzeros, res, eval_fn)
    elif _mode == 3:
        sa_temp = _sa_initial_temp
        if sa_temp is None:
            sa_temp = _auto_calibrate_temperature(totalbuscos)
        sa_cooling = _sa_cooling_rate
        if sa_cooling is None:
            sa_cooling = (_sa_min_temp / sa_temp) ** (1.0 / max(numofiterations, 1))
        optimizer = SimulatedAnnealingOptimizer(
            _min_pid, _min_qpct, _min_qrpct, _stepsize, _maxzeros, res,
            sa_temp, sa_cooling, _sa_min_temp)
    else:
        optimizer = RandomWalkOptimizer(_min_pid, _min_qpct, _min_qrpct,
                                        _stepsize, _maxzeros, res)

    for i in range(1, numofiterations):
        if _mode == 3:
            prev_pid, prev_qpct, prev_qrpct = myPID, myQPctMin, myQRPctMin

        myPID, myQPctMin, myQRPctMin = optimizer.step(
            myPID, myQPctMin, myQRPctMin, i, costfxndelta)

        if hasattr(optimizer, 'last_result') and optimizer.last_result is not None:
            cost, contigs, purged, scores = optimizer.last_result
        elif _gpu_fast:
            cost, scores, good_mask = _evaluate_cost_gpu(
                myPID, myQPctMin, myQRPctMin, totalbuscos)
            contigs, purged = None, None
        else:
            cost, contigs, purged, scores = evaluate_thresholds(
                myPID, myQPctMin, myQRPctMin, totalbuscos)

        if _mode == 3:
            delta = costfxn[i - 1] - cost
            if not optimizer.accept(delta):
                myPID, myQPctMin, myQRPctMin = prev_pid, prev_qpct, prev_qrpct
                cost = costfxn[i - 1]
                contigs, purged = None, None

        costfxn[i] = cost
        costfxndelta[i] = costfxn[i - 1] - costfxn[i]

        if pbar is not None:
            pbar.set_postfix_str(
                'PID: ' + ('%.4f' % myPID)
                + ' QPMin: ' + ('%.4f' % myQPctMin)
                + ' QRPMin: ' + ('%.4f' % myQRPctMin)
                + ' CostΔ ' + ('%+.4f' % costfxndelta[i])
                + ' Score: ' + ('%.4f' % cost))
            pbar.update(1)

        if upq.should_add(cost):
            if contigs is None:
                if _mode == 3 and costfxn[i] == costfxn[i - 1]:
                    pass
                elif _gpu_fast:
                    contigs, purged = _materialize_contigs(good_mask)
                else:
                    _, contigs, purged, _ = evaluate_thresholds(
                        myPID, myQPctMin, myQRPctMin, totalbuscos)
            if contigs is not None:
                upq.add([cost, contigs, purged, scores,
                         [myPID, myQPctMin, myQRPctMin]])

    if pbar is not None:
        pbar.close()
    return [upq.items, costfxn, costfxndelta]


def _detect_gpu_info():
    """Auto-detect GPU(s) and return device info list.

    Per NVIDIA CUDA docs, maxBlocksPerMultiprocessor × SMs = "full wave"
    (max simultaneous CUDA thread blocks). However, our walker count is
    a tensor dimension processed by CuPy's internal kernels, not a CUDA
    block count. The practical cap is GPU memory: the scoring tensor
    (N, n_ortho, n_contigs) uint8 dominates VRAM usage.

    default_agents = SM count (good balance of exploration vs convergence).
    max_agents = memory-based estimate (set later when data sizes are known).
    """
    try:
        import cupy as cp
        n_devices = cp.cuda.runtime.getDeviceCount()
        devices = []
        for i in range(n_devices):
            props = cp.cuda.runtime.getDeviceProperties(i)
            name = props['name']
            if isinstance(name, bytes):
                name = name.decode()
            sm_count = props['multiProcessorCount']
            major = props['major']
            cores_per_sm = 128 if major >= 8 else 64 if major == 7 else 128
            mem_bytes = props['totalGlobalMem']
            mem_gb = mem_bytes / (1024**3)
            try:
                max_blocks_per_sm = cp.cuda.runtime.deviceGetAttribute(106, i)
            except Exception:
                max_blocks_per_sm = 16
            devices.append({
                'id': i, 'name': name, 'sm_count': sm_count,
                'cuda_cores': sm_count * cores_per_sm,
                'memory_bytes': mem_bytes,
                'memory_gb': mem_gb,
                'compute_capability': (major, props['minor']),
                'max_blocks_per_sm': max_blocks_per_sm,
                'full_wave': sm_count * max_blocks_per_sm,
                'default_agents': sm_count,
            })
        return devices
    except Exception:
        return []


def _check_oob_plateau(pid, qpct, qrpct, iteration, cost_deltas_w):
    """Check OOB/plateau and return (needs_reset, new_pid, new_qpct, new_qrpct)."""
    plateau = (iteration >= _maxzeros and
               all(abs(cost_deltas_w[k]) <= _resolution
                   for k in range(iteration - _maxzeros + 1, iteration + 1)))
    out_pid = pid > 1.0
    out_qpct = qpct > 1.0
    out_qrpct = qrpct > 1.0

    if (out_pid and out_qpct and out_qrpct) or plateau:
        return True, uniform(_min_pid, 1.0), uniform(_min_qpct, 1.0), uniform(_min_qrpct, 1.0)
    if out_qpct and out_qrpct:
        return True, pid, uniform(_min_qpct, 1.0), uniform(_min_qrpct, 1.0)
    if out_pid and out_qpct:
        return True, uniform(_min_pid, 1.0), uniform(_min_qpct, 1.0), qrpct
    if out_pid and out_qrpct:
        return True, uniform(_min_pid, 1.0), qpct, uniform(_min_qrpct, 1.0)
    if out_pid:
        return True, uniform(_min_pid, 1.0), qpct, qrpct
    if out_qpct:
        return True, pid, uniform(_min_qpct, 1.0), qrpct
    if out_qrpct:
        return True, pid, qpct, uniform(_min_qrpct, 1.0)
    return False, pid, qpct, qrpct


def _estimate_batch_chunk_size():
    """Estimate max candidates per chunk based on GPU memory.

    The scoring tensor (N, n_ortho, n_contigs) uint8 dominates memory.
    Peak usage per candidate ~ n_ortho * n_contigs * 5 bytes (tensor + temps).
    Uses 60% of free GPU memory as budget.
    """
    import cupy as cp
    free_mem = cp.cuda.Device().mem_info[0]
    budget = int(free_mem * 0.6)
    n_ortho = _gpu_scoring_data.n_orthologs
    bytes_per_candidate = n_ortho * _gpu_n_contigs * 5
    if bytes_per_candidate == 0:
        return 10000
    return max(16, budget // bytes_per_candidate)


def _evaluate_batch_gpu(pids, qpcts, qrpcts, total_buscos):
    """Evaluate N threshold combos simultaneously on GPU.

    Broadcasting filter: (A,) alignment arrays against (N,) threshold arrays,
    producing (A,N) hit masks in one fused operation. Scatter-based removal
    mask construction, then batch scoring via ScoringData.calculate_scores_batch.

    Auto-chunks large batches to fit GPU memory (scoring tensor is
    (N, n_ortho, n_contigs) and can exceed VRAM for N > ~1000).

    pids, qpcts, qrpcts: cupy float32 arrays of shape (N,)
    Returns: (costs, singles, dups, frags, missings, good_masks)
        costs: cupy (N,) float64
        singles/dups/frags/missings: cupy (N,) int
        good_masks: cupy (N, n_contigs) bool
    """
    import cupy as cp

    qrpct_maxs = cp.exp(-cp.log2(qrpcts.astype(cp.float64))).astype(cp.float32)

    hit_mask = ((_gpu_pid_arr[:, None] >= pids[None, :]) &
                (_gpu_qpct_arr[:, None] >= qpcts[None, :]) &
                (_gpu_qrpct_arr[:, None] >= qrpcts[None, :]) &
                (_gpu_qrpct_arr[:, None] <= qrpct_maxs[None, :]))

    N = len(pids)
    removal_mask = cp.zeros((N, _gpu_n_contigs), dtype=bool)
    hit_a, hit_n = cp.nonzero(hit_mask)
    if len(hit_a) > 0:
        removal_mask[hit_n, _gpu_qnameid_arr[hit_a]] = True

    good_masks = (_gpu_all_mask[None, :] & ~removal_mask) | _gpu_missing_ref_mask[None, :]
    good_masks = good_masks & ~_gpu_small_mask[None, :]

    chunk_size = _estimate_batch_chunk_size()
    max_oom_retries = 3

    singles_parts, dups_parts, frags_parts, missings_parts = [], [], [], []
    oom_retries = 0
    start = 0
    while start < N:
        end = min(start + chunk_size, N)
        try:
            s, d, f, m = _gpu_scoring_data.calculate_scores_batch(good_masks[start:end])
            singles_parts.append(s)
            dups_parts.append(d)
            frags_parts.append(f)
            missings_parts.append(m)
            oom_retries = 0
            start = end
        except cp.cuda.memory.OutOfMemoryError:
            cp.get_default_memory_pool().free_all_blocks()
            oom_retries += 1
            if oom_retries > max_oom_retries:
                raise RuntimeError(
                    'GPU out of memory after ' + str(max_oom_retries)
                    + ' consecutive retries with chunk_size='
                    + str(chunk_size)
                    + '. Try reducing --agents or --totaliters.'
                )
            chunk_size = max(1, chunk_size // 2)
            print('GPU OOM: reducing chunk size to '
                  + str(chunk_size), flush=True)
    singles = cp.concatenate(singles_parts) if len(singles_parts) > 1 else singles_parts[0]
    dups = cp.concatenate(dups_parts) if len(dups_parts) > 1 else dups_parts[0]
    frags = cp.concatenate(frags_parts) if len(frags_parts) > 1 else frags_parts[0]
    missings = cp.concatenate(missings_parts) if len(missings_parts) > 1 else missings_parts[0]

    S_f64 = singles.astype(cp.float64)
    D_f64 = dups.astype(cp.float64)
    F_f64 = frags.astype(cp.float64)
    M_f64 = missings.astype(cp.float64)

    if _cost_formula_fn is not None:
        raw_costs = _cost_formula_fn(
            S=S_f64, D=D_f64, F=F_f64, M=M_f64,
            C=S_f64 + D_f64, n=cp.float64(total_buscos),
            theta_s=_theta_s, theta_d=_theta_d,
            theta_f=_theta_f, theta_m=_theta_m)
        costs = cp.where(singles > 0, raw_costs, cp.float64(50000000.0))
        costs = cp.where(cp.isfinite(costs), costs, cp.float64(50000000.0))
    else:
        costs = cp.where(
            singles > 0,
            (_theta_f * F_f64 + _theta_d * D_f64 + _theta_m * M_f64) /
            (_theta_s * S_f64),
            cp.float64(50000000.0)
        )

    return costs, singles, dups, frags, missings, good_masks


def _evaluate_batch_gpu_nosync(pids, qpcts, qrpcts, total_buscos):
    """Sync-free GPU batch evaluation using sparse scatter + matmul scoring.

    Replaces cp.nonzero (sync) with sparse matmul and the 3D scoring tensor
    with two matrix multiplies. No .get() or cp.where()[0] calls — all results
    stay on GPU.
    """
    import cupy as cp

    qrpct_maxs = cp.exp(-cp.log2(qrpcts.astype(cp.float64))).astype(cp.float32)

    hit_mask = ((_gpu_pid_arr[:, None] >= pids[None, :]) &
                (_gpu_qpct_arr[:, None] >= qpcts[None, :]) &
                (_gpu_qrpct_arr[:, None] >= qrpcts[None, :]) &
                (_gpu_qrpct_arr[:, None] <= qrpct_maxs[None, :]))

    hit_float = hit_mask.astype(cp.float32)
    removal_counts = _gpu_scatter_matrix @ hit_float
    removal_mask = removal_counts.T > 0

    good_masks = (_gpu_all_mask[None, :] & ~removal_mask) | _gpu_missing_ref_mask[None, :]
    good_masks = good_masks & ~_gpu_small_mask[None, :]

    singles, dups, frags, missings = _gpu_scoring_data.calculate_scores_batch_fast(good_masks)

    S_f64 = singles.astype(cp.float64)
    D_f64 = dups.astype(cp.float64)
    F_f64 = frags.astype(cp.float64)
    M_f64 = missings.astype(cp.float64)

    if _cost_formula_fn is not None:
        raw_costs = _cost_formula_fn(
            S=S_f64, D=D_f64, F=F_f64, M=M_f64,
            C=S_f64 + D_f64, n=cp.float64(total_buscos),
            theta_s=_theta_s, theta_d=_theta_d,
            theta_f=_theta_f, theta_m=_theta_m)
        costs = cp.where(singles > 0, raw_costs, cp.float64(50000000.0))
        costs = cp.where(cp.isfinite(costs), costs, cp.float64(50000000.0))
    else:
        costs = cp.where(
            singles > 0,
            (_theta_f * F_f64 + _theta_d * D_f64 + _theta_m * M_f64) /
            (_theta_s * S_f64),
            cp.float64(50000000.0))

    return costs, singles, dups, frags, missings


def hillclimbing_gpu_native(job_args_list):
    """GPU-native optimization — entire walker loop stays on GPU.

    All SA/random-walk logic uses CuPy operations with no CPU sync inside
    the loop. Only transfers final results at the end. Supports modes 0
    (random walk) and 3 (simulated annealing). Falls back to
    hillclimbing_gpu_batched for mode 2 (steepest descent).
    """
    import cupy as cp

    if _mode == 2:
        return hillclimbing_gpu_batched(job_args_list)

    _ensure_gpu()

    N = len(job_args_list)
    iterations = job_args_list[0][1]
    res = job_args_list[0][2]
    step_size = cp.float32(_stepsize)
    min_pid_f = cp.float32(_min_pid)
    min_qpct_f = cp.float32(_min_qpct)
    min_qrpct_f = cp.float32(_min_qrpct)

    pids = cp.array([float(j[3]) for j in job_args_list], dtype=cp.float32)
    qpcts = cp.array([float(j[4]) for j in job_args_list], dtype=cp.float32)
    qrpcts = cp.array([float(j[5]) for j in job_args_list], dtype=cp.float32)

    allmycontigs = _query_contigs.union(_missing_ref) - _small_contigs - {''}
    baseline_scores = _scoring_data.calculate_scores(allmycontigs)
    totalbuscos = baseline_scores['C'] + baseline_scores['M'] + baseline_scores['F']

    if _mode == 3:
        sa_temp = _sa_initial_temp
        if sa_temp is None:
            sa_temp = _auto_calibrate_temperature(totalbuscos)
        sa_cooling = _sa_cooling_rate
        if sa_cooling is None:
            sa_cooling = (_sa_min_temp / sa_temp) ** (1.0 / max(iterations, 1))
        temps = cp.full(N, sa_temp, dtype=cp.float64)
        reheats = cp.zeros(N, dtype=cp.float64)

    costs, S, D, F, M = _evaluate_batch_gpu_nosync(
        pids, qpcts, qrpcts, totalbuscos)

    # Plateau restart (and SA reheat), mirroring RandomWalkOptimizer/SimulatedAnnealingOptimizer.
    # The classes test |cost_delta| <= resolution over iterations i-maxzeros+1..i while
    # cost_delta[i] is still 0, i.e. over the previous maxzeros-1 recorded deltas. Without this
    # the walkers barely move (step 1e-4) and every run ends at its seeded starting basin.
    zero_run = cp.zeros(N, dtype=cp.int32)
    plateau_need = max(_maxzeros - 1, 1)
    n_restarts = cp.zeros(N, dtype=cp.int32)

    best_costs = costs.copy()
    best_pids = pids.copy()
    best_qpcts = qpcts.copy()
    best_qrpcts = qrpcts.copy()
    best_S = S.copy()
    best_D = D.copy()
    best_F = F.copy()
    best_M = M.copy()
    prev_costs = costs.copy()
    # Per-iteration current cost of every walker, kept on the device (as the CPU driver's costfxn),
    # so .scores/.deltascores are meaningful on this path too.
    cost_hist = cp.empty((iterations, N), dtype=cp.float64)
    cost_hist[0] = costs

    pbar = None
    sync_interval = 200
    if HAS_TQDM:
        pbar = tqdm(
            total=iterations,
            desc='GPU native (%d walkers)' % N,
            bar_format='{desc} [{bar:30}] {n_fmt}/{total_fmt}'
                       ' [{elapsed}<{remaining}] {postfix}',
            leave=True)
        pbar.update(1)

    for i in range(1, iterations):
        if _mode == 3:
            raw = cp.random.randint(0, 3, (N, 3)).astype(cp.float32) * step_size
            all_zero = (raw[:, 0] == 0) & (raw[:, 1] == 0) & (raw[:, 2] == 0)
            fix = cp.random.randint(0, 3, (N,))
            raw[:, 0] = cp.where(all_zero & (fix == 0), step_size, raw[:, 0])
            raw[:, 1] = cp.where(all_zero & (fix == 1), step_size, raw[:, 1])
            raw[:, 2] = cp.where(all_zero & (fix == 2), step_size, raw[:, 2])
        else:
            # Random forward walk: 0, 1 or 2 steps per threshold, as RandomWalkOptimizer and the
            # original HapSolo (until 2026-10-01 this path stepped -2..+2, unlike the CPU walk).
            raw = cp.random.randint(0, 3, (N, 3)).astype(cp.float32) * step_size
            all_zero = (raw[:, 0] == 0) & (raw[:, 1] == 0) & (raw[:, 2] == 0)
            fix = cp.random.randint(0, 3, (N,))
            raw[:, 0] = cp.where(all_zero & (fix == 0), step_size, raw[:, 0])
            raw[:, 1] = cp.where(all_zero & (fix == 1), step_size, raw[:, 1])
            raw[:, 2] = cp.where(all_zero & (fix == 2), step_size, raw[:, 2])

        proposed_p = pids + raw[:, 0]
        proposed_q = qpcts + raw[:, 1]
        proposed_r = qrpcts + raw[:, 2]

        plateau = (i >= _maxzeros) & (zero_run >= plateau_need)
        n_restarts += plateau
        # A plateau restarts all three coordinates; otherwise only out-of-bounds ones are redrawn.
        oob_p = (proposed_p > 1.0) | (proposed_p < min_pid_f) | plateau
        oob_q = (proposed_q > 1.0) | (proposed_q < min_qpct_f) | plateau
        oob_r = (proposed_r > 1.0) | (proposed_r < min_qrpct_f) | plateau
        proposed_p = cp.where(oob_p,
                              cp.random.uniform(float(_min_pid), 1.0, (N,)).astype(cp.float32),
                              proposed_p)
        proposed_q = cp.where(oob_q,
                              cp.random.uniform(float(_min_qpct), 1.0, (N,)).astype(cp.float32),
                              proposed_q)
        proposed_r = cp.where(oob_r,
                              cp.random.uniform(float(_min_qrpct), 1.0, (N,)).astype(cp.float32),
                              proposed_r)

        new_costs, nS, nD, nF, nM = _evaluate_batch_gpu_nosync(
            proposed_p, proposed_q, proposed_r, totalbuscos)

        if _mode == 3:
            # Reheat to T0 * 0.5**k on the k-th plateau of a walker, without cooling on that
            # iteration (SimulatedAnnealingOptimizer.step); all other iterations cool.
            reheat_temps = cp.maximum(cp.float64(_sa_min_temp), sa_temp * 0.5 ** reheats)
            temps = cp.where(plateau, reheat_temps,
                             cp.maximum(cp.float64(_sa_min_temp), temps * sa_cooling))
            reheats = reheats + plateau
            deltas = prev_costs - new_costs
            accept_prob = cp.exp(deltas / temps)
            rand_accept = cp.random.uniform(0, 1, (N,)).astype(cp.float64)
            accept = (deltas >= 0) | ((temps > _sa_min_temp) & (rand_accept < accept_prob))
            pids = cp.where(accept, proposed_p, pids)
            qpcts = cp.where(accept, proposed_q, qpcts)
            qrpcts = cp.where(accept, proposed_r, qrpcts)
            new_prev = cp.where(accept, new_costs, prev_costs)
        else:
            pids = proposed_p
            qpcts = proposed_q
            qrpcts = proposed_r
            new_prev = new_costs
        # A rejected SA move repeats the previous cost (delta 0), as in the CPU driver.
        zero_run = cp.where(cp.abs(prev_costs - new_prev) <= res, zero_run + 1, 0)
        prev_costs = new_prev
        cost_hist[i] = prev_costs

        improved = new_costs < best_costs
        best_costs = cp.where(improved, new_costs, best_costs)
        best_pids = cp.where(improved, proposed_p, best_pids)
        best_qpcts = cp.where(improved, proposed_q, best_qpcts)
        best_qrpcts = cp.where(improved, proposed_r, best_qrpcts)
        best_S = cp.where(improved, nS, best_S)
        best_D = cp.where(improved, nD, best_D)
        best_F = cp.where(improved, nF, best_F)
        best_M = cp.where(improved, nM, best_M)

        if pbar is not None and (i % sync_interval == 0 or i == iterations - 1):
            best_val = float(cp.min(best_costs))
            done = min(i + 1, iterations)
            pbar.n = done
            pbar.set_postfix_str('Best: %.6f' % best_val)
            pbar.refresh()

    if pbar is not None:
        pbar.close()
    restarts = n_restarts.get()
    print('GPU native: plateau restarts per walker: mean %.1f, min %d, max %d (of %d iterations)'
          % (restarts.mean(), restarts.min(), restarts.max(), iterations - 1), file=sys.stderr)

    hist = cost_hist.get()
    # cost_delta[0] is the first cost and cost_delta[i] = cost[i-1] - cost[i], as in hillclimbing().
    delta_hist = np.empty_like(hist)
    delta_hist[0] = hist[0]
    delta_hist[1:] = hist[:-1] - hist[1:]
    f_costs = best_costs.get()
    f_pids = best_pids.get()
    f_qpcts = best_qpcts.get()
    f_qrpcts = best_qrpcts.get()
    f_S = best_S.get()
    f_D = best_D.get()
    f_F = best_F.get()
    f_M = best_M.get()

    results = []
    for w in range(N):
        pid_w = float(f_pids[w])
        qpct_w = float(f_qpcts[w])
        qrpct_w = float(f_qrpcts[w])
        cost_w = float(f_costs[w])
        _, good_contigs, purged_contigs, _ = evaluate_thresholds(
            pid_w, qpct_w, qrpct_w, totalbuscos)
        scores = {'S': int(f_S[w]), 'D': int(f_D[w]),
                  'F': int(f_F[w]), 'M': int(f_M[w]),
                  'C': int(f_S[w]) + int(f_D[w])}
        upq = UniquePriorityQueue(_bestnscores)
        upq.add([cost_w, good_contigs, purged_contigs, scores,
                 [pid_w, qpct_w, qrpct_w]])
        results.append([upq.items, hist[:, w].tolist(), delta_hist[:, w].tolist()])

    return results


def hillclimbing_gpu_batched(job_args_list):
    """Single-process batched GPU hill climbing.

    Runs all walkers in one CUDA context, evaluating all threshold combos
    per iteration in a single batched GPU operation. For mode 2, all 8
    neighbors per walker are evaluated simultaneously (up to N*8 evals
    per batch).

    job_args_list: list of [thread_id, iterations, resolution, pid, qpct, qrpct]
    Returns: list of [upq_items, cost_history, cost_deltas] per walker
             (same format as pool.map(hillclimbing, ...))
    """
    import cupy as cp

    _ensure_gpu()

    N = len(job_args_list)
    iterations = job_args_list[0][1]
    res = job_args_list[0][2]

    walker_pids = [float(job_args_list[w][3]) for w in range(N)]
    walker_qpcts = [float(job_args_list[w][4]) for w in range(N)]
    walker_qrpcts = [float(job_args_list[w][5]) for w in range(N)]

    cost_history = [[0.0] * iterations for _ in range(N)]
    cost_deltas = [[0.0] * iterations for _ in range(N)]
    upqs = [UniquePriorityQueue(_bestnscores) for _ in range(N)]

    allmycontigs = _query_contigs.union(_missing_ref) - _small_contigs - {''}
    baseline_scores = _scoring_data.calculate_scores(allmycontigs)
    totalbuscos = baseline_scores['C'] + baseline_scores['M'] + baseline_scores['F']
    if baseline_scores['S'] == 0:
        baseline_cost = 5000.0
    else:
        baseline_cost = cost_function(baseline_scores['M'], baseline_scores['S'],
                                       baseline_scores['D'], baseline_scores['F'],
                                       totalbuscos, _theta_s, _theta_d, _theta_f, _theta_m,
                                       formula_fn=_cost_formula_fn)

    baseline_contigs = allmycontigs.copy()
    baseline_purged = _all_contigs - baseline_contigs - {''}
    for w in range(N):
        upqs[w].add([baseline_cost, baseline_contigs.copy(), baseline_purged.copy(),
                     baseline_scores.copy(), [0.0, 0.0, 0.0]])

    if _mode == 3:
        sa_temp = _sa_initial_temp
        if sa_temp is None:
            sa_temp = _auto_calibrate_temperature(totalbuscos)
        sa_cooling = _sa_cooling_rate
        if sa_cooling is None:
            sa_cooling = (_sa_min_temp / sa_temp) ** (1.0 / max(iterations, 1))
        walker_optimizers = [
            SimulatedAnnealingOptimizer(_min_pid, _min_qpct, _min_qrpct,
                                        _stepsize, _maxzeros, res,
                                        sa_temp, sa_cooling, _sa_min_temp)
            for _ in range(N)
        ]
    elif _mode == 0:
        walker_optimizers = [
            RandomWalkOptimizer(_min_pid, _min_qpct, _min_qrpct,
                                _stepsize, _maxzeros, res)
            for _ in range(N)
        ]

    best_ever = float('inf')

    pbar = None
    if HAS_TQDM:
        pbar = tqdm(
            total=iterations,
            desc='GPU batched (%d walkers)' % N,
            bar_format='{desc} [{bar:30}] {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}',
            leave=True,
        )

    batch_pids = cp.array(walker_pids, dtype=cp.float32)
    batch_qpcts = cp.array(walker_qpcts, dtype=cp.float32)
    batch_qrpcts = cp.array(walker_qrpcts, dtype=cp.float32)

    costs_gpu, S, D, F, M, masks = _evaluate_batch_gpu(
        batch_pids, batch_qpcts, batch_qrpcts, totalbuscos)
    costs_cpu = costs_gpu.get()
    S_cpu, D_cpu, F_cpu, M_cpu = S.get(), D.get(), F.get(), M.get()

    for w in range(N):
        cost = float(costs_cpu[w])
        cost_history[w][0] = cost
        cost_deltas[w][0] = cost
        scores = {'S': int(S_cpu[w]), 'D': int(D_cpu[w]), 'F': int(F_cpu[w]),
                  'M': int(M_cpu[w]), 'C': int(S_cpu[w]) + int(D_cpu[w])}
        good_contigs, purged_contigs = _materialize_contigs(masks[w])
        upqs[w].add([cost, good_contigs, purged_contigs, scores,
                     [walker_pids[w], walker_qpcts[w], walker_qrpcts[w]]])

    if pbar is not None:
        best_ever = min(best_ever, float(min(costs_cpu)))
        pbar.update(1)
        pbar.set_postfix_str('Best: %.6f' % best_ever)

    for i in range(1, iterations):
        if _mode == 2:
            candidates = []
            walker_ranges = {}

            for w in range(N):
                needs_reset, new_p, new_q, new_r = _check_oob_plateau(
                    walker_pids[w], walker_qpcts[w], walker_qrpcts[w],
                    i, cost_deltas[w])

                if needs_reset:
                    start = len(candidates)
                    candidates.append((new_p, new_q, new_r))
                    walker_ranges[w] = (start, start + 1, True)
                    walker_pids[w] = new_p
                    walker_qpcts[w] = new_q
                    walker_qrpcts[w] = new_r
                else:
                    start = len(candidates)
                    s = _stepsize
                    for dp, dq, dr in SteepestDescentOptimizer.DIRECTIONS:
                        np_ = walker_pids[w] + dp * s
                        nq = walker_qpcts[w] + dq * s
                        nr = walker_qrpcts[w] + dr * s
                        if (_min_pid <= np_ <= 1.0 and
                                _min_qpct <= nq <= 1.0 and
                                _min_qrpct <= nr <= 1.0):
                            candidates.append((np_, nq, nr))
                    end = len(candidates)
                    if end == start:
                        new_p = uniform(_min_pid, 1.0)
                        new_q = uniform(_min_qpct, 1.0)
                        new_r = uniform(_min_qrpct, 1.0)
                        candidates.append((new_p, new_q, new_r))
                        walker_ranges[w] = (start, start + 1, True)
                        walker_pids[w] = new_p
                        walker_qpcts[w] = new_q
                        walker_qrpcts[w] = new_r
                    else:
                        walker_ranges[w] = (start, end, False)

            batch_pids = cp.array([c[0] for c in candidates], dtype=cp.float32)
            batch_qpcts = cp.array([c[1] for c in candidates], dtype=cp.float32)
            batch_qrpcts = cp.array([c[2] for c in candidates], dtype=cp.float32)

            costs_gpu, S, D, F, M, masks = _evaluate_batch_gpu(
                batch_pids, batch_qpcts, batch_qrpcts, totalbuscos)
            costs_cpu = costs_gpu.get()
            S_cpu, D_cpu, F_cpu, M_cpu = S.get(), D.get(), F.get(), M.get()

            for w in range(N):
                start, end, is_reset = walker_ranges[w]
                if is_reset:
                    best_idx = start
                else:
                    best_idx = start
                    best_c = float(costs_cpu[start])
                    for c in range(start + 1, end):
                        if costs_cpu[c] < best_c:
                            best_c = costs_cpu[c]
                            best_idx = c
                    walker_pids[w] = float(batch_pids[best_idx])
                    walker_qpcts[w] = float(batch_qpcts[best_idx])
                    walker_qrpcts[w] = float(batch_qrpcts[best_idx])

                cost = float(costs_cpu[best_idx])
                cost_history[w][i] = cost
                cost_deltas[w][i] = cost_history[w][i - 1] - cost

                if upqs[w].should_add(cost):
                    scores = {'S': int(S_cpu[best_idx]), 'D': int(D_cpu[best_idx]),
                              'F': int(F_cpu[best_idx]), 'M': int(M_cpu[best_idx]),
                              'C': int(S_cpu[best_idx]) + int(D_cpu[best_idx])}
                    good_contigs, purged_contigs = _materialize_contigs(masks[best_idx])
                    upqs[w].add([cost, good_contigs, purged_contigs, scores,
                                 [walker_pids[w], walker_qpcts[w], walker_qrpcts[w]]])

        else:
            if _mode == 3:
                prev_pids = walker_pids[:]
                prev_qpcts = walker_qpcts[:]
                prev_qrpcts = walker_qrpcts[:]

            for w in range(N):
                new_p, new_q, new_r = walker_optimizers[w].step(
                    walker_pids[w], walker_qpcts[w], walker_qrpcts[w],
                    i, cost_deltas[w])
                walker_pids[w] = new_p
                walker_qpcts[w] = new_q
                walker_qrpcts[w] = new_r

            batch_pids = cp.array(walker_pids, dtype=cp.float32)
            batch_qpcts = cp.array(walker_qpcts, dtype=cp.float32)
            batch_qrpcts = cp.array(walker_qrpcts, dtype=cp.float32)

            costs_gpu, S, D, F, M, masks = _evaluate_batch_gpu(
                batch_pids, batch_qpcts, batch_qrpcts, totalbuscos)
            costs_cpu = costs_gpu.get()
            scores_transferred = False

            for w in range(N):
                cost = float(costs_cpu[w])

                if _mode == 3:
                    delta = cost_history[w][i - 1] - cost
                    if not walker_optimizers[w].accept(delta):
                        walker_pids[w] = prev_pids[w]
                        walker_qpcts[w] = prev_qpcts[w]
                        walker_qrpcts[w] = prev_qrpcts[w]
                        cost = cost_history[w][i - 1]

                cost_history[w][i] = cost
                cost_deltas[w][i] = cost_history[w][i - 1] - cost

                if upqs[w].should_add(cost):
                    if not scores_transferred:
                        S_cpu, D_cpu, F_cpu, M_cpu = S.get(), D.get(), F.get(), M.get()
                        scores_transferred = True
                    scores = {'S': int(S_cpu[w]), 'D': int(D_cpu[w]),
                              'F': int(F_cpu[w]), 'M': int(M_cpu[w]),
                              'C': int(S_cpu[w]) + int(D_cpu[w])}
                    good_contigs, purged_contigs = _materialize_contigs(masks[w])
                    upqs[w].add([cost, good_contigs, purged_contigs, scores,
                                 [walker_pids[w], walker_qpcts[w], walker_qrpcts[w]]])

        if pbar is not None:
            current_costs = [cost_history[w][i] for w in range(N)]
            best_ever = min(best_ever, min(current_costs))
            pbar.update(1)
            pbar.set_postfix_str('Best: %.6f' % best_ever)

    if pbar is not None:
        pbar.close()

    return [[upqs[w].items, cost_history[w], cost_deltas[w]] for w in range(N)]
