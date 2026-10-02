#!/usr/bin/env python3
"""HapSolo CLI entry point: python -m hapsolo"""

import argparse
import datetime
import os
import sys
from random import seed, uniform

import multiprocessing as mp

from hapsolo.assembly import CalculateContigSizes, calculateasmstats, WriteNewAssembly
from hapsolo.alignment import create_paf_alignment, create_psl_alignment, load_hap_file
from hapsolo.scoring import load_orthologs
from hapsolo.names import build_conversion_dict
from hapsolo.report import write_report
from hapsolo.utils import CalculateInverseProportion
from hapsolo import optimizers

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


FASTA_EXTENSIONS = ('.fasta', '.fa', '.fna', '.fas', '.fsa', '.faa', '.seq')


def score_dump_path(asm_path, timestamp, suffix):
    """Path for a .scores/.deltascores dump next to the input assembly.

    Names containing '.fasta' keep the historical naming (that substring replaced). Other names
    have a known FASTA extension stripped, or the suffix appended, so the dump can never take the
    input's own path (an input named *.fa used to be overwritten with scores).
    """
    tag = '_' + timestamp + suffix
    if '.fasta' in asm_path:
        path = asm_path.replace('.fasta', tag)
    else:
        root, ext = os.path.splitext(asm_path)
        path = (root if ext.lower() in FASTA_EXTENSIONS else asm_path) + tag
    if os.path.abspath(path) == os.path.abspath(asm_path):
        raise ValueError('Refusing to write scores over the input assembly: %s' % asm_path)
    return path


def build_parser():
    parser = argparse.ArgumentParser(
        description='Process alignments and orthologs for selecting reduced assembly candidates',
        epilog='-p/--psl and -a/--paf are mutually exclusive')
    parser.add_argument('-i', '--input', help='Input Fasta file', type=str, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-p', '--psl', help='BLAT PSL alignment file', type=str)
    group.add_argument('-a', '--paf', help='Minimap2 PAF alignment file', type=str)
    group.add_argument('--hap', help='Pre-cached HAP alignment file (from cache step)', type=str)
    parser.add_argument('--mode', help='Run mode: 0=random walk, 1=fixed thresholds, '
                        '2=steepest descent, 3=simulated annealing. Default=0',
                        type=int, required=False)
    parser.add_argument('-b', '--buscos', help='Ortholog data: directory (odb_output/) or cached TSV file', type=str, required=True)
    parser.add_argument('-m', '--maxzeros', help='Max consecutive zero-delta iterations. Default=10',
                        type=int, required=False)
    parser.add_argument('-t', '--threads', help='CPU threads (ignored with --gpu). Default=1', type=int, required=False)
    parser.add_argument('-n', '--niterations', help='Iterations per agent/thread. Default=1000', type=int, required=False)
    parser.add_argument('--totaliters', help='Total iterations across all agents/threads. '
                        'Auto-divides by agent/thread count. Takes precedence over -n.',
                        type=int, required=False)
    parser.add_argument('--agents', help='Number of GPU walkers (default: auto from CUDA max concurrent blocks). '
                        'Capped at hardware limit with warning.', type=int, required=False)
    parser.add_argument('-B', '--Bestn', help='Best N assemblies to return. Default=1', type=int, required=False)
    parser.add_argument('-S', '--thetaS', help='Weight for single orthologs. Default=1.0', type=float, required=False)
    parser.add_argument('-D', '--thetaD', help='Weight for duplicate orthologs. Default=1.0', type=float, required=False)
    parser.add_argument('-F', '--thetaF', help='Weight for fragmented orthologs. Default=0.0', type=float, required=False)
    parser.add_argument('-M', '--thetaM', help='Weight for missing orthologs. Default=1.0', type=float, required=False)
    parser.add_argument('-P', '--minPID', help='Min PID threshold. Default=0.2', type=float, required=False)
    parser.add_argument('-Q', '--minQ', help='Min QPct threshold. Default=0.2', type=float, required=False)
    parser.add_argument('-R', '--minQR', help='Min QRPct threshold. Default=0.2', type=float, required=False)
    parser.add_argument('--min', help='Min contig size for primary assembly. Default=1000', type=int, required=False)
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU acceleration via cupy (requires: pip install cupy-cuda12x)')
    parser.add_argument('--generate-hap', action='store_true',
                        help='Generate the .hap file from the alignment and exit. '
                        'Subsequent runs reuse the .hap file (skips PAF/PSL parsing).')
    parser.add_argument('--outdir', help='Output directory for assembly files (default: asms)',
                        type=str, required=False, default='asms')
    parser.add_argument('--sa-temp', help='SA initial temperature (default: auto-calibrate)',
                        type=float, required=False)
    parser.add_argument('--sa-cooling', help='SA cooling rate alpha (default: auto from iterations)',
                        type=float, required=False)
    parser.add_argument('--sa-min-temp', help='SA minimum temperature floor (default: 1e-6)',
                        type=float, required=False, default=1e-6)
    parser.add_argument('--formula',
                        help='Custom cost formula using variables S, D, F, M, C (=S+D), n (total). '
                        'Theta weights available as theta_s, theta_d, theta_f, theta_m. '
                        'Functions: abs, sqrt, log, log2, log10, exp, pow. '
                        'Example: --formula "(D**2 + M) / S"',
                        type=str, required=False, default=None)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    myasmFileName = args.input
    pslalignmentfile = args.psl
    pafalignmentfile = args.paf
    hapalignmentfile = args.hap
    buscofileloc = args.buscos
    maxzeros = args.maxzeros if args.maxzeros is not None else 10
    bestnscores = args.Bestn if args.Bestn is not None else 1

    thetaS = args.thetaS if args.thetaS is not None else 1.0
    if thetaS == 0.0:
        print('Warning: --thetaS cannot be 0 (division by zero in cost function). Using default 1.0')
        thetaS = 1.0
    thetaD = args.thetaD if args.thetaD is not None else 1.0
    thetaM = args.thetaM if args.thetaM is not None else 1.0
    thetaF = args.thetaF if args.thetaF is not None else 0.0
    outdir = args.outdir

    cost_formula_fn = None
    cost_formula_str = None
    if args.formula is not None:
        from hapsolo.cost_formula import compile_cost_formula
        try:
            cost_formula_fn = compile_cost_formula(args.formula)
            cost_formula_str = args.formula
            print('Custom cost formula: ' + args.formula)
        except (ValueError, SyntaxError) as e:
            print('Error in --formula: ' + str(e))
            quit(1)

    mode = args.mode if args.mode is not None else 0
    myMinPID = args.minPID
    myMinQPctMin = args.minQ
    myMinQRPctMin = args.minQR
    myMinContigSize = args.min

    if mode == 1:
        bestnscores = 1
        customMinPID = myMinPID if myMinPID is not None else 0.7
        customMinQPctMin = myMinQPctMin if myMinQPctMin is not None else 0.7
        customMinQRPctMin = myMinQRPctMin if myMinQRPctMin is not None else 0.7

    if myMinPID is None:
        myMinPID = 0.2
    if myMinQPctMin is None:
        myMinQPctMin = 0.2
    if myMinQRPctMin is None:
        myMinQRPctMin = 0.2
    elif myMinQRPctMin < 0.02:
        myMinQRPctMin = 0.02
        print('-R/--minQR set to a value less than 0.02. using 0.02 instead.')

    if myMinContigSize is None or myMinContigSize < 0:
        myMinContigSize = 1000

    # ── GPU / CPU setup ─────────────────────────────────────────────────
    use_gpu = args.gpu
    gpu_devices = []

    if use_gpu:
        try:
            import cupy
            print('GPU acceleration enabled (cupy ' + cupy.__version__ + ')')
        except ImportError:
            print('Error: --gpu requires cupy. Install with: pip install cupy-cuda12x')
            quit(1)
        gpu_devices = optimizers._detect_gpu_info()
        for dev in gpu_devices:
            print('  GPU %d: %s (%d CUDA cores, %d SMs, %.1f GB, CC %d.%d)' %
                  (dev['id'], dev['name'], dev['cuda_cores'], dev['sm_count'],
                   dev['memory_gb'], dev['compute_capability'][0], dev['compute_capability'][1]))

    if use_gpu and mode != 1:
        if args.threads is not None:
            print('Note: -t/--threads ignored with --gpu (use --agents for GPU walker count)')
        threads = 0
        iterations = 0
    elif mode != 1:
        threads = args.threads if args.threads is not None else 1
        if threads < 1:
            print('Invalid # of threads set. Please use a positive integer for threads')
            quit(1)
        if args.agents is not None:
            print('Note: --agents ignored without --gpu (use -t/--threads for CPU)')
        if args.totaliters is not None:
            iterations = max(1, -(-args.totaliters // threads))
            actual_total = iterations * threads
            print('CPU: %d threads x %d iters/thread = %d total evaluations (requested %d)' %
                  (threads, iterations, actual_total, args.totaliters))
        else:
            iterations = args.niterations if args.niterations is not None else 1000
    else:
        threads = 1
        iterations = 1

    dumpscores = True
    stepsize = 0.0001
    resolution = 0.0001

    seed(1)

    try:
        contig_sizes, error_log = CalculateContigSizes(myasmFileName)
    except (IOError, OSError) as e:
        print('Error reading assembly file: ' + str(e))
        quit(1)
    except (IndexError, ValueError) as e:
        print('Error parsing assembly file (malformed FASTA?): ' + str(e))
        quit(1)

    smallcontigset = set()
    for key in contig_sizes.keys():
        if contig_sizes[key][0] < myMinContigSize:
            smallcontigset.add(key)

    import time as _time
    _t_aln_start = _time.time()

    if hapalignmentfile is not None:
        mypddf = load_hap_file(hapalignmentfile)
    elif pafalignmentfile is not None:
        mypddf = create_paf_alignment(pafalignmentfile, myMinContigSize,
                                      myMinPID, myMinQPctMin, myMinQRPctMin)
    elif pslalignmentfile is not None:
        mypddf = create_psl_alignment(pslalignmentfile, myMinContigSize,
                                      myMinPID, myMinQPctMin, myMinQRPctMin)

    _t_aln_end = _time.time()
    print('>>> Timing: alignment loading = %.1fs' % (_t_aln_end - _t_aln_start))

    if args.generate_hap:
        print('HAP file generated. Exiting (--generate-hap mode).')
        return

    canonical_names = set(contig_sizes.keys())
    aln_names = set(mypddf['qName']).union(set(mypddf['tName']))
    aln_conversion, aln_unmatched = build_conversion_dict(canonical_names, aln_names)
    if aln_conversion:
        print(str(len(aln_conversion)) + ' alignment contig name(s) remapped to match assembly:')
        for old_name in sorted(aln_conversion.keys()):
            print('  ' + old_name + ' -> ' + aln_conversion[old_name])
        mypddf['qName'] = mypddf['qName'].replace(aln_conversion)
        mypddf['tName'] = mypddf['tName'].replace(aln_conversion)
    if aln_unmatched:
        print('Warning: ' + str(len(aln_unmatched)) + ' alignment contig name(s) could not be matched to assembly:')
        for name in sorted(aln_unmatched):
            print('  ' + name)

    qrycontigset = set(mypddf['qName'])
    missingrefcontigset = set(contig_sizes.keys()) - qrycontigset
    allcontigsset = set(contig_sizes.keys())

    _t_odb_start = _time.time()
    busco2contigdict, contigs2buscodict = load_orthologs(buscofileloc)
    _t_odb_end = _time.time()
    print('>>> Timing: ortholog loading = %.1fs' % (_t_odb_end - _t_odb_start))

    busco_names = set(contigs2buscodict.keys())
    busco_conversion, busco_unmatched = build_conversion_dict(canonical_names, busco_names)
    if busco_conversion:
        print(str(len(busco_conversion)) + ' BUSCO contig name(s) remapped to match assembly:')
        for old_name in sorted(busco_conversion.keys()):
            print('  ' + old_name + ' -> ' + busco_conversion[old_name])
        new_contigs2buscodict = dict()
        for name in contigs2buscodict:
            new_name = busco_conversion.get(name, name)
            new_contigs2buscodict[new_name] = contigs2buscodict[name]
        contigs2buscodict = new_contigs2buscodict
        buscotypes = ['C', 'S', 'D', 'F', 'M']
        for buscoid in busco2contigdict:
            for buscotype in buscotypes:
                busco2contigdict[buscoid][buscotype] = [busco_conversion.get(n, n)
                                                       for n in busco2contigdict[buscoid][buscotype]]
    if busco_unmatched:
        print('Warning: ' + str(len(busco_unmatched)) + ' BUSCO contig name(s) could not be matched to assembly:')
        for name in sorted(busco_unmatched):
            print('  ' + name)

    optimizers.setup(
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
        sa_initial_temp=args.sa_temp,
        sa_cooling_rate=args.sa_cooling,
        sa_min_temp=args.sa_min_temp,
        cost_formula_fn=cost_formula_fn,
        cost_formula_str=cost_formula_str,
    )

    # ── GPU agent count (deferred until data sizes are known) ──────────
    if use_gpu and mode != 1:
        n_ortho = len(busco2contigdict)
        n_contigs = len(allcontigsset)
        sm_count = gpu_devices[0]['sm_count']
        mem_bytes = gpu_devices[0]['memory_bytes']
        bytes_per_agent = n_ortho * n_contigs * 5
        if bytes_per_agent > 0:
            max_gpu_agents = max(1, int(mem_bytes * 0.6 / bytes_per_agent))
        else:
            max_gpu_agents = sm_count

        default_agents = sm_count
        agents = default_agents
        if args.agents is not None:
            if args.agents > max_gpu_agents:
                print('Warning: --agents %d exceeds GPU memory limit (~%d max for %d orthologs x %d contigs). Capped to %d.' %
                      (args.agents, max_gpu_agents, n_ortho, n_contigs, max_gpu_agents))
                agents = max_gpu_agents
            else:
                agents = args.agents

        if args.totaliters is not None:
            iterations = max(1, -(-args.totaliters // agents))
            actual_total = iterations * agents
            print('GPU: %d agents x %d iters/agent = %d total evaluations (requested %d)' %
                  (agents, iterations, actual_total, args.totaliters))
        elif args.niterations is not None:
            iterations = args.niterations
            print('GPU: %d agents x %d iters/agent = %d total evaluations' %
                  (agents, iterations, iterations * agents))
        else:
            iterations = 1000
            print('GPU: %d agents x %d iters/agent = %d total evaluations' %
                  (agents, iterations, iterations * agents))
        threads = agents

    # ── Optimization (modes 0 and 2) ────────────────────────────────────
    _t_opt_start = _time.time()
    job_args = list()
    if mode != 1:
        if use_gpu:
            for i in range(threads):
                job_args.append([i, iterations, resolution,
                                 uniform(myMinPID, 1), uniform(myMinQPctMin, 1), uniform(myMinQRPctMin, 1)])
            mylist = optimizers.hillclimbing_gpu_native(job_args)
            if HAS_TQDM:
                sys.stderr.write('\n')
                sys.stderr.flush()
        elif threads == 1:
            job_args = [0, iterations, resolution,
                        uniform(myMinPID, 1), uniform(myMinQPctMin, 1), uniform(myMinQRPctMin, 1)]
            result = optimizers.hillclimbing(job_args)
            mylist = [result]
            if HAS_TQDM:
                sys.stderr.write('\n')
                sys.stderr.flush()
        else:
            for i in range(threads):
                job_args.append([i, iterations, resolution,
                                 uniform(myMinPID, 1), uniform(myMinQPctMin, 1), uniform(myMinQRPctMin, 1)])
            if HAS_TQDM:
                pool = mp.Pool(processes=threads,
                               initializer=tqdm.set_lock,
                               initargs=(tqdm.get_lock(),))
            else:
                pool = mp.Pool(processes=threads)
            mylist = pool.map(optimizers.hillclimbing, job_args)
            pool.close()
            pool.join()
            if HAS_TQDM:
                sys.stderr.write('\n' * threads)
                sys.stderr.flush()

        _t_opt_end = _time.time()
        print('>>> Timing: optimization = %.1fs' % (_t_opt_end - _t_opt_start))
        print('>>> Timing: total = %.1fs (alignment=%.1fs + orthologs=%.1fs + optimization=%.1fs)' %
              (_t_opt_end - _t_aln_start, _t_aln_end - _t_aln_start,
               _t_odb_end - _t_odb_start, _t_opt_end - _t_opt_start))

        mybestnscoreslist = list()
        mybestnscoreslist.append(mylist[0][0][0])
        for i in range(0, len(mylist)):
            for j in range(0, min(bestnscores, len(mylist[i][0]))):
                mybestnscoreslist = optimizers.uniquepriorityqueue(mybestnscoreslist, mylist[i][0][j])
        for i in range(0, min(bestnscores, len(mybestnscoreslist))):
            for j in range(0, len(mybestnscoreslist[0][4])):
                mybestnscoreslist[i][4][j] = '%.4f' % mybestnscoreslist[i][4][j]
            asmbase = os.path.basename(myasmFileName).replace('.fasta', '')
            newasmfilename = (asmbase + '_' + str(myMinContigSize) + '_'
                              + str(mybestnscoreslist[i][4][0]) + '_'
                              + str(mybestnscoreslist[i][4][2]) + 'to'
                              + str('%.4f' % CalculateInverseProportion(float(mybestnscoreslist[i][4][2])))
                              + '_' + str(mybestnscoreslist[i][4][1]) + '_primary.fasta')
            print('Writing ' + newasmfilename + ' with score: ' + str(mybestnscoreslist[i][0]))
            WriteNewAssembly(myasmFileName, newasmfilename, mybestnscoreslist[i][1],
                             contig_sizes, error_log, outdir=outdir)
            WriteNewAssembly(myasmFileName, newasmfilename.replace('_primary.fasta', '_secondary.fasta'),
                             mybestnscoreslist[i][2], contig_sizes, error_log, outdir=outdir)
            write_report(outdir + '/' + newasmfilename, mybestnscoreslist[i][1],
                         mybestnscoreslist[i][3], contig_sizes)
        if dumpscores:
            timestamp = str(datetime.datetime.today()).replace(' ', '_').replace('-', '_').replace(':', '_').split('.')[0]
            with open(score_dump_path(myasmFileName, timestamp, '.scores'), 'w') as fout:
                for i in range(0, len(mylist)):
                    fout.write(str(mylist[i][1][0]))
                    for j in range(1, iterations):
                        fout.write(',' + str(mylist[i][1][j]))
                    fout.write('\n')
            with open(score_dump_path(myasmFileName, timestamp, '.deltascores'), 'w') as fout:
                for i in range(0, len(mylist)):
                    fout.write(str(mylist[i][2][0]))
                    for j in range(1, iterations):
                        fout.write(',' + str(mylist[i][2][j]))
                    fout.write('\n')
    elif mode == 1:
        job_args = [0, 1, resolution, customMinPID, customMinQPctMin, customMinQRPctMin]
        mylist = optimizers.hillclimbing(job_args)
        _t_opt_end = _time.time()
        print('>>> Timing: optimization = %.1fs' % (_t_opt_end - _t_opt_start))
        print('>>> Timing: total = %.1fs (alignment=%.1fs + orthologs=%.1fs + optimization=%.1fs)' %
              (_t_opt_end - _t_aln_start, _t_aln_end - _t_aln_start,
               _t_odb_end - _t_odb_start, _t_opt_end - _t_opt_start))
        mybestnscoreslist = mylist[0]
        for i in range(0, min(bestnscores, len(mybestnscoreslist))):
            for j in range(0, len(mybestnscoreslist[0][4])):
                mybestnscoreslist[i][4][j] = '%.4f' % mybestnscoreslist[i][4][j]
            asmbase = os.path.basename(myasmFileName).replace('.fasta', '')
            newasmfilename = (asmbase + '_' + str(myMinContigSize) + '_'
                              + str(mybestnscoreslist[i][4][0]) + '_'
                              + str(mybestnscoreslist[i][4][2]) + 'to'
                              + str('%.4f' % CalculateInverseProportion(float(mybestnscoreslist[i][4][2])))
                              + '_' + str(mybestnscoreslist[i][4][1]) + '_primary.fasta')
            print('Writing ' + newasmfilename + ' with score: ' + str(mybestnscoreslist[i][0]))
            WriteNewAssembly(myasmFileName, newasmfilename, mybestnscoreslist[i][1],
                             contig_sizes, error_log, outdir=outdir)
            WriteNewAssembly(myasmFileName, newasmfilename.replace('_primary.fasta', '_secondary.fasta'),
                             mybestnscoreslist[i][2], contig_sizes, error_log, outdir=outdir)
            write_report(outdir + '/' + newasmfilename, mybestnscoreslist[i][1],
                         mybestnscoreslist[i][3], contig_sizes)


if __name__ == '__main__':
    main()
