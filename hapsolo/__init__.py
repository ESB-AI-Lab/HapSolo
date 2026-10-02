"""HapSolo — haplotig reduction via hill-climbing optimization.

This package re-exports the public API for backward compatibility.
Tests and external consumers can use ``import hapsolo`` as before.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Re-exports from submodules (pure functions)
# ---------------------------------------------------------------------------
from hapsolo.state import HapSoloState
from hapsolo.scoring_data import ScoringData

from hapsolo.utils import (
    open_gzip,
    CalculatePctAlign,
    CalculateInverseProportion,
)
from hapsolo.names import (
    sanitize_name,
    build_conversion_dict,
)
from hapsolo.alignment import (
    reduce_asm,
    create_paf_alignment,
    create_psl_alignment,
    load_hap_file,
    _print_purge_breakdown,
    _open_with_progress,
)
from hapsolo.scoring import (
    import_orthologs,
    import_orthologs_tsv,
    load_orthologs,
    calculate_scores,
    cost_function,
    BUSCO_TYPES,
)
from hapsolo.optimizers import (
    BaseOptimizer,
    RandomWalkOptimizer,
    SteepestDescentOptimizer,
    SimulatedAnnealingOptimizer,
    UniquePriorityQueue,
    setup as setup_optimizers,
)
from hapsolo.cost_formula import (
    compile_cost_formula,
    validate_formula,
    DEFAULT_FORMULA,
)
from hapsolo.assembly import (
    CalculateContigSizes as _CalculateContigSizes,
    calculateasmstats as _calculateasmstats,
    WriteNewAssembly as _WriteNewAssembly,
)
from hapsolo.report import (
    write_report as _write_report,
    _compute_detailed_asm_stats as _detailed_asm_stats,
    _compute_gc_and_ns,
)
from hapsolo import optimizers as _optimizers

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ---------------------------------------------------------------------------
# Legacy module-level globals — tests set these directly via
#   ``hapsolo.mypddf = df``, etc.
# ---------------------------------------------------------------------------
mypddf = pd.DataFrame()
missingrefcontigset = set()
qrycontigset = set()
allcontigsset = set()
smallcontigset = set()
busco2contigdict = dict()
contigs2buscodict = dict()
myContigsDict = dict()
myscerrorlog = ''
buscotypes = ['C', 'S', 'D', 'F', 'M']
myMinContigSize = 1000
myMinPID = 0.2
myMinQPctMin = 0.2
myMinQRPctMin = 0.2
thetaS = 1.0
thetaD = 1.0
thetaF = 0.0
thetaM = 1.0
_cost_formula_fn = None
maxzeros = 10
stepsize = 0.0001
resolution = 0.0001
bestnscores = 1
mode = 0
iterations = 1000
threads = 1
dumpscores = True
use_gpu = False


# ---------------------------------------------------------------------------
# Legacy function wrappers
# These read module-level globals so old-style calls still work.
# ---------------------------------------------------------------------------

def CalculateContigSizes(asmFileName):
    """Legacy wrapper: returns just dict (original behavior).
    Also stores error_log in hapsolo.myscerrorlog for backward compat.
    New code should use hapsolo.assembly.CalculateContigSizes which returns (dict, error_log).
    """
    global myscerrorlog
    result = _CalculateContigSizes(asmFileName)
    if isinstance(result, tuple):
        myscerrorlog = result[1]
        return result[0]
    return result


def calculateasmstats(bestcontigset):
    """Legacy wrapper: reads contig_sizes from hapsolo.myContigsDict."""
    return _calculateasmstats(bestcontigset, myContigsDict)


def WriteNewAssembly(myasmFileName, newASMFileName, myGoodContigsSet):
    """Legacy wrapper: reads contig_sizes and error_log from module globals."""
    return _WriteNewAssembly(myasmFileName, newASMFileName, myGoodContigsSet,
                             myContigsDict, myscerrorlog)


def write_report(primary_fasta_path, contig_set, ortho_scores):
    """Legacy wrapper: reads contig_sizes from hapsolo.myContigsDict."""
    return _write_report(primary_fasta_path, contig_set, ortho_scores, myContigsDict)


def _compute_detailed_asm_stats(contig_set):
    """Legacy wrapper: reads contig_sizes from hapsolo.myContigsDict."""
    return _detailed_asm_stats(contig_set, myContigsDict)


def ReduceASM(myPID, myQPctMin, myQRPctMin):
    return reduce_asm(mypddf, myPID, myQPctMin, myQRPctMin, allcontigsset)


def calculateBuscos(mycontigslist, b2c, c2b):
    return calculate_scores(mycontigslist, b2c, c2b,
                            missingrefcontigset, smallcontigset)


def myLinearFxn(mbusco, sbusco, dbusco, fbusco, cbusco):
    return cost_function(mbusco, sbusco, dbusco, fbusco, cbusco,
                         thetaS, thetaD, thetaF, thetaM,
                         formula_fn=_cost_formula_fn)


def importBuscos(buscofileloc):
    global busco2contigdict, contigs2buscodict
    busco2contigdict, contigs2buscodict = import_orthologs(buscofileloc)
    return busco2contigdict, contigs2buscodict


def CreateMM2AlignmentDataStructure(alignmentfile):
    global mypddf
    mypddf = create_paf_alignment(alignmentfile, myMinContigSize,
                                  myMinPID, myMinQPctMin, myMinQRPctMin)
    return mypddf


def CreateBlatAlignmentDataStructure(alignmentfile):
    global mypddf
    mypddf = create_psl_alignment(alignmentfile, myMinContigSize,
                                  myMinPID, myMinQPctMin, myMinQRPctMin)
    return mypddf


def _sync_optimizer_state():
    """Sync module-level globals into the optimizers module for hillclimbing."""
    _optimizers.setup(
        df=mypddf,
        all_contigs=allcontigsset,
        missing_ref=missingrefcontigset,
        small_contigs=smallcontigset,
        query_contigs=qrycontigset,
        busco2contig=busco2contigdict,
        contigs2busco=contigs2buscodict,
        theta_s=thetaS, theta_d=thetaD, theta_f=thetaF, theta_m=thetaM,
        mode=mode,
        bestnscores=bestnscores,
        min_pid=myMinPID, min_qpct=myMinQPctMin, min_qrpct=myMinQRPctMin,
        stepsize=stepsize, maxzeros=maxzeros, resolution=resolution,
        use_gpu=use_gpu,
        cost_formula_fn=_cost_formula_fn,
    )


def evaluate_thresholds(pid, qpct, qrpct, total_buscos):
    """Legacy wrapper: syncs module globals into optimizers, then evaluates."""
    _sync_optimizer_state()
    return _optimizers.evaluate_thresholds(pid, qpct, qrpct, total_buscos)


def hillclimbing(job_args):
    """Legacy wrapper: syncs module globals into optimizers, then runs."""
    _sync_optimizer_state()
    return _optimizers.hillclimbing(job_args)


def uniquepriorityqueue(pqlist, myvalue):
    """Legacy wrapper: syncs bestnscores before calling."""
    _optimizers._bestnscores = bestnscores
    return _optimizers.uniquepriorityqueue(pqlist, myvalue)
