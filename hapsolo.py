#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Edwin
"""
import argparse, glob, gzip, os, datetime, sys, re
from math import exp, log, ceil
from random import seed, randint, uniform
import pandas as pd
import multiprocessing as mp

# Optional dependency: tqdm for per-thread progress bars.
# Falls back to silent mode if not installed.
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# usage haplotigreduction.py mypslfile.psl myfastafile.fasta buscoresults.tsv
parser = argparse.ArgumentParser(description='Process alignments and BUSCO"s for selecting reduced assembly candidates', epilog='-p/--psl and -a/--paf are mutually exclusive')
parser.add_argument('-i', '--input', help='Input Fasta file', type=str, required=True)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-p', '--psl', help='BLAT PSL alignnment file', type=str)
group.add_argument('-a', '--paf', help='Minimap2 PAF alignnment file. Note. paf file functionality is currently experimental', type=str)

#mode = parser.add_mutually_exclusive_group(required=True)
#mode.add_argument('-p', '--psl', help='BLAT PSL alignnment file', type=str)
#mode.add_argument('-p', '--psl', help='BLAT PSL alignnment file', type=str)
#mode.add_argument('-p', '--psl', help='BLAT PSL alignnment file', type=str)
parser.add_argument('--mode', help='HapSolo run mode. 0 = Random walking, 1 = No optimization with defaults, 2 = Optimized walking, Default = 0', type=int, required=False)

parser.add_argument('-b', '--buscos', help='Location BUSCO output directories. i.e. buscoN/', type=str, required=True)
parser.add_argument('-m', '--maxzeros', help='Max # of times cost function delta can consecutively be 0. Default = 10', type=int, required=False)
parser.add_argument('-t', '--threads', help='# of threads. Multiplies iterations by threads. Default = 1', type=int, required=False)
parser.add_argument('-n', '--niterations', help='# of total iterations to run per gradient descent. Default = 1000', type=int, required=False)
parser.add_argument('-B', '--Bestn', help='# of best candidate assemblies to return using gradient descent. Default = 1', type=int, required=False)
parser.add_argument('-S', '--thetaS', help='Weight for single BUSCOs in linear fxn. Default = 1.0', type=float, required=False)
parser.add_argument('-D', '--thetaD', help='Weight for duplicate BUSCOs in linear fxn. Default = 1.0', type=float, required=False)
parser.add_argument('-F', '--thetaF', help='Weight for fragmented BUSCOs in linear fxn. Default = 0.0', type=float, required=False)
parser.add_argument('-M', '--thetaM', help='Weight for missing BUSCOs in linear fxn. Default = 1.0', type=float, required=False)
# parser.add_argument('-T', '--thetaS', help='Weight for single BUSCOs in linear fxn. Default = 1.0', type=float, required=False)
parser.add_argument('-P', '--minPID', help='Restrict values of PID to be >= the value set here. Default = 0.2', type=float, required=False)
parser.add_argument('-Q', '--minQ', help='Restrict values of Q to be >= the value set here. Default = 0.2', type=float, required=False)
parser.add_argument('-R', '--minQR', help='Restrict values of QR to be >= the value set here. Cannot be 0. Default = 0.2', type=float, required=False)
parser.add_argument('--min', help='Minimum size of contigs for Primary assembly. Default = 1000', type=int, required=False)


args = parser.parse_args()

useprimaryformula = True

myasmFileName = args.input
pslalignmentfile = args.psl
pafalignmentfile = args.paf
buscofileloc = args.buscos
maxzeros = args.maxzeros
threads = args.threads
iterations = args.niterations
bestnscores = args.Bestn
thetaS = args.thetaS
thetaD = args.thetaD
thetaM = args.thetaM
thetaF = args.thetaF
# thetaT = args.thetaT
mode = args.mode
myMinPID = args.minPID
myMinQPctMin = args.minQ
myMinQRPctMin = args.minQR
myMinContigSize = args.min
aimode = 0

#if alignmentfile == None:
#    print('Please assign an alignment file. Either by using Minimap2 or Blat/PBlat')
#    print('This can be done by doing either --paf or --psl. Please only submit one file.')
#    quit(1)
if maxzeros is None:
    maxzeros = 10
if threads is None:
    threads = 1
if iterations is None:
    iterations = 1000
if bestnscores is None:
    bestnscores = 1
if thetaS is None:
    thetaS = 1.0
elif thetaS == 0.0:
    print('Warning: --thetaS cannot be 0 (division by zero in cost function). Using default 1.0')
    thetaS = 1.0
if thetaD is None:
    thetaD = 1.0
if thetaM is None:
    thetaM = 1.0
if thetaF is None:
    thetaF = 0.0
# if thetaT is None:
    # thetaT = 0.0
if mode is None:
    mode = 0
elif mode == 1:
    bestnscores = 1
    # In mode 1 (no optimization, fixed thresholds), the user-supplied
    # --minPID / --minQ / --minQR values ARE the fixed thresholds.
    # If not provided, default to 0.7 for backwards compatibility with the
    # original hapsolo behavior.
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

# maxASMSize = 600 * 1000000

dumpscores = True
stepsize = 0.0001
buscotypes = ['C', 'S', 'D', 'F', 'M']
resolution = 0.0001
mypddf = pd.DataFrame()
missingrefcontigset = set()
qrycontigset = set()
allcontigsset = set()
smallcontigset = set()
busco2contigdict = dict()
contigs2buscodict = dict()
pythonversion = sys.version_info[0]
# special_chars are !@#$%^&*()-=+,./\[{}]|;:'><?
special_chars = '!@#$%^&*-=+,/\\()[{]}|;:"\'><?' # removed . from special chars
myContigsDict = dict()
myscerrorlog = ''

if pythonversion != 2:
    print("Note: HapSolo was originally developed for Python 2.7. Running on Python " + str(pythonversion) + ".")


def open_gzip(filename):
    """Open a gzip file for text reading, compatible with Python 2 and 3."""
    if sys.version_info[0] >= 3:
        return gzip.open(filename, 'rt')
    return gzip.open(filename, 'r')

######################################
def CalculateContigSizes(asmFileName):
    global myscerrorlog
    # contigsDict[contigname] = [contiglen,headerpos,startseqpos,endseqpos]
    myContigSizeDict = dict()
    with open(asmFileName) as fin:
        lastPos = headerPos = fin.tell()
        totalLines = sum(1 for line in fin)
        fin.seek(lastPos)
        seqLen = 0
        seqName = ''
        lastPos = 0
        count = 0
        # print('begin for loop')
        while count < totalLines:
            # print('for loop executed')
            lastPos = headerPos = fin.tell()
            line = fin.readline().replace('\n', '')
            count = count + 1
            if line[0:1] == '>':
                header = line[1:]
                special_char = False
                for char in header:
                    if char in special_chars:
                        special_char = char
                        break
                # print('found seq_name ' + line)
                if len(header.split(" ")) > 1:
                    print('Spaces found in contig headers. Please remove spaces from contig names before proceeding with any analysis. Spaces, -"s, //"s and other special characters are not allowed in contig names.')
                    quit(1)
                if special_char:
                    my_log_str_sc = 'Warning! Special characters except _ cause isues in aligners and BUSCO analysis. HapSolo found: ' + special_char + ' in header: ' + header + '. This may cause HapSolo to fail.'
                    myscerrorlog = myscerrorlog + my_log_str_sc + '\n'
                    print(my_log_str_sc)
                    special_char = False
                    #quit(1)
                seqName = header.split(" ")[0].replace('/', '_')
                # seqName = line.split("_")[0]
                lastPos = startPos = fin.tell()
                line = fin.readline().replace('\n', '')
                count = count + 1
                # print('begin while loop on seq ' + line)
                while line[0:1] != '>' and line[0:1] != '':
                    seqLen = seqLen + len(line)
                    endPos = lastPos
                    lastPos = fin.tell()
                    line = fin.readline().replace('\n', '')
                    count = count + 1
                if line[0:1] == '>' or line[0:1] == '':
                    myContigSizeDict[seqName] = [seqLen, headerPos, startPos, endPos]
                    # print(len(seq_read.replace("\n", "")))
                    seqName = ''
                    seqLen = 0
                    count = count - 1
                    fin.seek(lastPos)
    return myContigSizeDict


def calculateasmstats(bestcontigset):
    mycontiglist = list()
    for contig in bestcontigset:
        if contig in myContigsDict.keys():
            mycontiglist.append(myContigsDict[contig][0])
    if len(mycontiglist) == 0:
        return 0, 0, 0, 0
    mycontiglist.sort(reverse=True)
    largestcontig = mycontiglist[0]
    asmsize = sum(mycontiglist)
    topn50contigs = 0
    n50 = 0
    l50 = 0
    for i in range(len(mycontiglist)):
        n50 = mycontiglist[i]
        topn50contigs = topn50contigs + mycontiglist[i]
        if topn50contigs > asmsize / 2.0:
            l50 = i + 1
            break
    return asmsize, n50, l50, largestcontig


def importBuscos(buscofileloc):
    contignames = set()
    buscoids = set()
    # Support both legacy busco_* and new odbaln_* subfolder naming
    mybuscofiles = glob.glob(buscofileloc + '/odbaln_*/*/full_table_*.tsv')
    if len(mybuscofiles) == 0:
        mybuscofiles = glob.glob(buscofileloc + '/busco*/*/full_table_*.tsv')
    if len(mybuscofiles) == 0:
        print('No result files found matching: ' + buscofileloc + '/{odbaln_*,busco*}/*/full_table_*.tsv')
        print('Please verify the output directory path and that ortholog search has completed successfully.')
        quit(1)
    global busco2contigdict
    global contigs2buscodict
    # propogate busco ids into a set
    with open(mybuscofiles[0]) as fin:
        for line in fin:
            if line[0] != '#':
                buscoids.add(line.strip().split()[0])
    # propogate contig names into a set
    for i in range(0, len(mybuscofiles)):
        mylinecounter = 0
        with open(mybuscofiles[i]) as fin:
            for line in fin:
                mylinecounter+=1
                if line[0] == '#' and mylinecounter < 4:
                    if mylinecounter == 3:
                        contignames.add(line.split()[8].split('/')[-1].replace('.fasta',''))
                elif mylinecounter > 3:
                    break
        #contignames.add(mybuscofiles[i].split('/')[-1].replace('full_table_', '').split('_new')[0])
    if len(contignames) != len(set(contignames)):
        print('duplicate contig names exist. Please fix contig names so that no duplicates exist and rerun HapSolo')
        quit(1)
    # propogate dictionaries
    for buscoid in buscoids:
        busco2contigdict[buscoid] = dict()
        for buscotype in buscotypes:
            busco2contigdict[buscoid][buscotype] = list()
    for contigname in contignames:
        contigs2buscodict[contigname] = dict()
        for buscotype in buscotypes:
            contigs2buscodict[contigname][buscotype] = list()
    # create a data structure for duplicate, single and fragmented busco id lookups
    # should start with contigs? or buscoids? maybe both
    for file in mybuscofiles:
        mylines = list()
        contigname = None
        mylinecounter = 0
        with open(file) as fin:
            for line in fin:
                if line[0] != '#':
                    mylines.append(line.strip().split())
                elif line[0] == '#' and mylinecounter < 4:
                    mylinecounter+=1
                    if mylinecounter == 3:
                        contigname = line.split()[8].split('/')[-1].replace('.fasta','')
        if contigname is None:
            print('Warning: could not extract contig name from BUSCO file: ' + file)
            continue
        for i in range(0, len(mylines)):
            buscoid = mylines[i][0]
            buscotype = mylines[i][1][0]
            if buscotype != 'M':
                busco2contigdict[buscoid][buscotype].append(contigname)
                contigs2buscodict[contigname][buscotype].append(buscoid)
    return busco2contigdict, contigs2buscodict


def calculateBuscos(mycontigslist, busco2contigdict, contigs2buscodict):
    # how should we deal with fragmented busco exists but exists as complete elsewhere?
    duplicatebuscos = 0
    singlebuscos = 0
    fragmentedbuscos = 0
    buscotypecounts = dict()
    buscoids = busco2contigdict.keys()
    # count the number of complete buscos
    completebuscoidcounts = dict()
    fragmentedbuscoidcounts = dict()
    for buscoid in buscoids:
        completebuscoidcounts[buscoid] = 0
        fragmentedbuscoidcounts[buscoid] = 0
    for buscotype in buscotypes:
        buscotypecounts[buscotype] = 0
    mycontigset = set(mycontigslist).union(missingrefcontigset) - smallcontigset
    for contig in mycontigset:
        if contig in contigs2buscodict.keys():
            for buscotype in contigs2buscodict[contig]:
                buscosize = len(contigs2buscodict[contig][buscotype])
                if buscotype == 'C' and buscosize > 0:
                    for buscoid in contigs2buscodict[contig][buscotype]:
                        # print('Found complete busco: ' + buscoid)
                        completebuscoidcounts[buscoid] += 1
        # else:
            # print('contig: ' + contig + ' not found in contigs2busco dictionary.')
    for contig in mycontigset:
        if contig in contigs2buscodict.keys():
            for buscotype in contigs2buscodict[contig]:
                buscosize = len(contigs2buscodict[contig][buscotype])
                if buscosize > 0 and buscotype == 'F':
                    for buscoid in contigs2buscodict[contig][buscotype]:
                        if completebuscoidcounts[buscoid] == 0 and fragmentedbuscoidcounts[buscoid] == 0:
                            fragmentedbuscos += 1
    for buscoid in completebuscoidcounts:
        mybuscocount = completebuscoidcounts[buscoid]
        if mybuscocount == 1:
            singlebuscos += 1
        elif mybuscocount > 1:
            duplicatebuscos += 1
    buscotypecounts['D'] = duplicatebuscos
    buscotypecounts['S'] = singlebuscos
    buscotypecounts['C'] = singlebuscos + duplicatebuscos
    buscotypecounts['F'] = fragmentedbuscos
    buscotypecounts['M'] = len(completebuscoidcounts) - buscotypecounts['D'] - buscotypecounts['S'] - buscotypecounts['F']
    return buscotypecounts


def ReduceASM(myPID, myQPctMin, myQRPctMin):
    # mypddf.columns = Index([u'matches', u'misMatches', u'repMatches', u'nCount', u'qNumInsert',
    # u'qBaseInsert', u'tNumInsert', u'tBaseInsert', u'strand', u'qName',
    # u'qSize', u'qStart', u'qEnd', u'tName', u'tSize', u'tStart', u'tEnd',
    # u'blockCount', u'blockSizes', u'qStarts', u'tStarts', u'qMin', u'qMax',
    # u'tMin', u'tMax', u'qAlignLen', u'rAlignLen', u'QRAlignLenPct', u'QPct',
    # u'QRPct'], dtype='object')
    myQRPctMax = CalculateInverseProportion(myQRPctMin)
    temppd0 = mypddf[mypddf['PID'] >= myPID]
    temppd1 = temppd0[temppd0['QPct'] >= myQPctMin]
    temppd0 = temppd1[temppd1['QRAlignLenPct'] >= myQRPctMin]
    temppd1 = temppd0[temppd0['QRAlignLenPct'] <= myQRPctMax]
    goodcontigset = allcontigsset - set(temppd1['qName'])
    return goodcontigset


def hillclimbing(job_args):
    mythread = job_args[0]
    numofiterations = job_args[1]
    res = job_args[2]
    myPID = job_args[3]
    myQPctMin = job_args[4]
    myQRPctMin = job_args[5]

    pbar = None
    if HAS_TQDM and mode != 1:
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

    # Baseline: score the full unfiltered assembly
    allmycontigs = qrycontigset.union(missingrefcontigset) - smallcontigset - {''}
    allcontigsbuscoscore = calculateBuscos(allmycontigs, busco2contigdict, contigs2buscodict)
    totalbuscos = allcontigsbuscoscore['C'] + allcontigsbuscoscore['M'] + allcontigsbuscoscore['F']
    if allcontigsbuscoscore['S'] == 0:
        oldasmscorefxn = 5000.0
    else:
        oldasmscorefxn = myLinearFxn(allcontigsbuscoscore['M'], allcontigsbuscoscore['S'],
                                      allcontigsbuscoscore['D'], allcontigsbuscoscore['F'], totalbuscos)

    upq = UniquePriorityQueue(bestnscores)
    bestcontigset = allmycontigs.copy()
    bestpurgedset = allcontigsset - bestcontigset - {''}
    if mode != 1:
        upq.add([oldasmscorefxn, bestcontigset, bestpurgedset,
                 allcontigsbuscoscore.copy(), [0.0, 0.0, 0.0]])

    # Iteration 0: evaluate starting thresholds
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

    if mode == 1:
        if pbar is not None:
            pbar.close()
        return [upq.items, costfxn, costfxndelta]

    if mode == 2:
        eval_fn = lambda p, q, r: evaluate_thresholds(p, q, r, totalbuscos)
        optimizer = SteepestDescentOptimizer(myMinPID, myMinQPctMin, myMinQRPctMin,
                                             stepsize, maxzeros, res, eval_fn)
    else:
        optimizer = RandomWalkOptimizer(myMinPID, myMinQPctMin, myMinQRPctMin,
                                        stepsize, maxzeros, res)

    for i in range(1, numofiterations):
        myPID, myQPctMin, myQRPctMin = optimizer.step(
            myPID, myQPctMin, myQRPctMin, i, costfxndelta)

        if hasattr(optimizer, 'last_result') and optimizer.last_result is not None:
            cost, contigs, purged, scores = optimizer.last_result
        else:
            cost, contigs, purged, scores = evaluate_thresholds(
                myPID, myQPctMin, myQRPctMin, totalbuscos)

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
            upq.add([cost, contigs, purged, scores,
                     [myPID, myQPctMin, myQRPctMin]])

    if pbar is not None:
        pbar.close()
    return [upq.items, costfxn, costfxndelta]


def CalculatePctAlign(myAlignLen, myTotalLen):
    if myTotalLen == 0:
        myFloat = 0.0000
    else:
        myFloat = float(myAlignLen) / float(myTotalLen)
    return myFloat


def CalculateInverseProportion(myPct):
    if myPct < 0.02:
        inversePct = myPct
    else:
        inversePct = exp(-1.0 * log(myPct, 2))
    return inversePct


class UniquePriorityQueue:
    """Bounded priority queue that deduplicates entries with identical contig sets.

    Entries are [score, contig_set, purged_set, busco_scores, thresholds].
    Lower score = better.  When two entries share the same contig set the one
    with the lower score is kept.
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
    upq = UniquePriorityQueue(bestnscores)
    for item in pqlist:
        upq.add(item)
    upq.add(myvalue)
    return upq.items


class BaseOptimizer:
    """Base class for threshold optimizers.

    Subclass and implement step() to create a new optimization strategy.
    """

    def __init__(self, min_pid, min_qpct, min_qrpct):
        self.min_pid = min_pid
        self.min_qpct = min_qpct
        self.min_qrpct = min_qrpct

    def step(self, pid, qpct, qrpct, iteration, cost_deltas):
        """Return new (pid, qpct, qrpct) thresholds."""
        raise NotImplementedError


class RandomWalkOptimizer(BaseOptimizer):
    """Random walk optimizer with plateau detection and boundary resets.

    This is the original HapSolo mode 0 algorithm: takes random steps in
    threshold space, resets to random when stuck on plateaus or when
    thresholds drift out of bounds [0, 1].
    """

    def __init__(self, min_pid, min_qpct, min_qrpct, step_size, maxzeros, resolution):
        super().__init__(min_pid, min_qpct, min_qrpct)
        self.step_size = step_size
        self.maxzeros = maxzeros
        self.resolution = resolution
        self._steps = [0, 0, 0]

    def step(self, pid, qpct, qrpct, iteration, cost_deltas):
        # After two consecutive cost increases, constrain the step vector
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
    """Steepest-descent 8-neighbor optimizer with plateau detection and boundary resets.

    Evaluates all 8 octant neighbors (+/- step on each of 3 dimensions) per
    iteration and moves to the lowest-cost neighbor. Uses the same restart
    triggers as RandomWalkOptimizer: plateau detection and out-of-bounds resets.

    Based on the steepest-descent hill climber designed by
    Juan Yin, Mansi Agrawal, and Prashansa for GPU HapSolo.
    """

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


def evaluate_thresholds(pid, qpct, qrpct, total_buscos):
    """Evaluate a set of filter thresholds against the global alignment data.

    Returns (cost, good_contigs, purged_contigs, busco_scores).
    """
    mygoodcontigs = ReduceASM(pid, qpct, qrpct)
    mygoodcontigs = mygoodcontigs.union(missingrefcontigset) - smallcontigset - {''}
    purgedcontigs = allcontigsset - mygoodcontigs - {''}
    scores = calculateBuscos(mygoodcontigs, busco2contigdict, contigs2buscodict)
    if scores['S'] == 0:
        cost = 50000000.0
    else:
        cost = myLinearFxn(scores['M'], scores['S'], scores['D'], scores['F'], total_buscos)
    return cost, mygoodcontigs, purgedcontigs, scores


def _compute_detailed_asm_stats(contig_set):
    """Compute assembly statistics at multiple size thresholds."""
    sizes = []
    for contig in contig_set:
        if contig in myContigsDict:
            sizes.append(myContigsDict[contig][0])
    if not sizes:
        return None
    sizes.sort(reverse=True)
    total = sum(sizes)
    thresholds = [0, 1000, 5000, 10000, 25000, 50000]
    counts = {}
    lengths = {}
    for t in thresholds:
        filtered = [s for s in sizes if s >= t]
        counts[t] = len(filtered)
        lengths[t] = sum(filtered)
    cumsum = 0
    n50 = n75 = 0
    l50 = l75 = 0
    for i, s in enumerate(sizes):
        cumsum += s
        if n50 == 0 and cumsum > total / 2.0:
            n50 = s
            l50 = i + 1
        if n75 == 0 and cumsum > total * 0.75:
            n75 = s
            l75 = i + 1
            break
    return {
        'counts': counts, 'lengths': lengths,
        'total_contigs': len(sizes), 'largest': sizes[0],
        'total_length': total,
        'n50': n50, 'l50': l50, 'n75': n75, 'l75': l75,
    }


def _compute_gc_and_ns(fasta_path):
    """Compute GC% and Ns per 100 kbp from a written FASTA file."""
    gc = 0
    at = 0
    n_count = 0
    total = 0
    with open(fasta_path) as f:
        for line in f:
            if line[0:1] == '>':
                continue
            seq = line.strip().upper()
            gc += seq.count('G') + seq.count('C')
            at += seq.count('A') + seq.count('T')
            n_count += seq.count('N')
            total += len(seq)
    gc_pct = 100.0 * gc / (gc + at) if (gc + at) > 0 else 0.0
    ns_per_100k = 100000.0 * n_count / total if total > 0 else 0.0
    return gc_pct, ns_per_100k


def write_report(primary_fasta_path, contig_set, ortho_scores):
    """Write assembly statistics and ortholog completeness report.

    primary_fasta_path: path to the written primary FASTA (in asms/).
    contig_set: set of contig names in the primary assembly.
    ortho_scores: dict with keys S, D, C, F, M.
    """
    report_path = primary_fasta_path.replace('_primary.fasta', '_report.txt')
    asm_name = os.path.basename(primary_fasta_path).replace('.fasta', '')

    stats = _compute_detailed_asm_stats(contig_set)
    if stats is None:
        return
    gc_pct, ns_per_100k = _compute_gc_and_ns(primary_fasta_path)

    total_ortho = ortho_scores['C'] + ortho_scores['F'] + ortho_scores['M']
    c_pct = 100.0 * ortho_scores['C'] / total_ortho if total_ortho > 0 else 0.0
    s_pct = 100.0 * ortho_scores['S'] / total_ortho if total_ortho > 0 else 0.0
    d_pct = 100.0 * ortho_scores['D'] / total_ortho if total_ortho > 0 else 0.0
    f_pct = 100.0 * ortho_scores['F'] / total_ortho if total_ortho > 0 else 0.0
    m_pct = 100.0 * ortho_scores['M'] / total_ortho if total_ortho > 0 else 0.0

    w = 28
    vw = 12
    with open(report_path, 'w') as f:
        f.write('All statistics are based on contigs of size >= 500 bp, unless otherwise noted '
                '(e.g., "# contigs (>= 0 bp)" and "Total length (>= 0 bp)" include all contigs).\n\n')
        f.write('{:<{w}}{}\n'.format('Assembly', asm_name, w=w))
        for t in [0, 1000, 5000, 10000, 25000, 50000]:
            f.write('{:<{w}}{:<{vw}}\n'.format(
                '# contigs (>= ' + str(t) + ' bp)', stats['counts'][t], w=w, vw=vw))
        for t in [0, 1000, 5000, 10000, 25000, 50000]:
            f.write('{:<{w}}{:<{vw}}\n'.format(
                'Total length (>= ' + str(t) + ' bp)', stats['lengths'][t], w=w, vw=vw))
        f.write('{:<{w}}{:<{vw}}\n'.format('# contigs', stats['total_contigs'], w=w, vw=vw))
        f.write('{:<{w}}{:<{vw}}\n'.format('Largest contig', stats['largest'], w=w, vw=vw))
        f.write('{:<{w}}{:<{vw}}\n'.format('Total length', stats['total_length'], w=w, vw=vw))
        f.write('{:<{w}}{:<{vw}.2f}\n'.format('GC (%)', gc_pct, w=w, vw=vw))
        f.write('{:<{w}}{:<{vw}}\n'.format('N50', stats['n50'], w=w, vw=vw))
        f.write('{:<{w}}{:<{vw}}\n'.format('N75', stats['n75'], w=w, vw=vw))
        f.write('{:<{w}}{:<{vw}}\n'.format('L50', stats['l50'], w=w, vw=vw))
        f.write('{:<{w}}{:<{vw}}\n'.format('L75', stats['l75'], w=w, vw=vw))
        f.write('{:<{w}}{:<{vw}.2f}\n'.format("# N's per 100 kbp", ns_per_100k, w=w, vw=vw))
        f.write('#\n')
        f.write('# Ortholog completeness (n=' + str(total_ortho) + ')\n')
        f.write('#\n')
        f.write('\tC:' + '{:.1f}'.format(c_pct) + '%'
                '[S:' + '{:.1f}'.format(s_pct) + '%,'
                'D:' + '{:.1f}'.format(d_pct) + '%],'
                'F:' + '{:.1f}'.format(f_pct) + '%,'
                'M:' + '{:.1f}'.format(m_pct) + '%,'
                'n:' + str(total_ortho) + '\n')
        f.write('\n')
        f.write('\t' + str(ortho_scores['C']) + '\tComplete orthologs (C)\n')
        f.write('\t' + str(ortho_scores['S']) + '\tComplete and single-copy orthologs (S)\n')
        f.write('\t' + str(ortho_scores['D']) + '\tComplete and duplicated orthologs (D)\n')
        f.write('\t' + str(ortho_scores['F']) + '\tFragmented orthologs (F)\n')
        f.write('\t' + str(ortho_scores['M']) + '\tMissing orthologs (M)\n')
        f.write('\t' + str(total_ortho) + '\tTotal ortholog groups searched\n')

    return report_path


def _print_purge_breakdown(fcounter, mcounter, purge_self, purge_size,
                            purge_pid, purge_qpct, purge_qrpct):
    """Print a breakdown of how many alignments were purged by each filter."""
    purged = fcounter - mcounter
    print(str(purged) + ' alignments Purged due to Search Space constraints')
    if purged > 0:
        print('  Breakdown (alignments may fail multiple filters; counted by first failure):')
        print('    Self-alignments (qName == tName):    ' + str(purge_self))
        print('    Query length < ' + str(myMinContigSize) + ' bp (--min):     ' + str(purge_size))
        print('    PID < ' + str(myMinPID) + ' (-P/--minPID):              ' + str(purge_pid))
        print('    QPct < ' + str(myMinQPctMin) + ' (-Q/--minQ):              ' + str(purge_qpct))
        print('    QRAlignLenPct < ' + str(myMinQRPctMin) + ' (-R/--minQR):     ' + str(purge_qrpct))
    print('  Alignments retained:                   ' + str(mcounter)
          + ' / ' + str(fcounter))


def _open_with_progress(filename, is_gz, desc='Reading'):
    """Open a file (or .gz) and return (file_handle, tqdm_bar, pos_callable).

    pos_callable() returns the current byte position for progress tracking.
    The bar is None if tqdm isn't available.

    Important: the returned file_handle MUST be read with readline() in a
    while loop, NOT with `for line in fin:`. The `for` loop uses Python's
    read-ahead buffer which disables tell() on text-mode files.

    For gzipped files, progress is tracked against the COMPRESSED file size
    (since uncompressed size isn't known cheaply).
    """
    total = os.path.getsize(filename)
    bar = None
    if HAS_TQDM:
        suffix = ' (gz)' if is_gz else ''
        bar = tqdm(total=total, unit='B', unit_scale=True, unit_divisor=1024,
                   desc=desc + suffix, dynamic_ncols=True)
    if is_gz:
        # Open underlying raw file in binary so tell() always works,
        # then wrap with gzip + TextIOWrapper for line reading.
        raw = open(filename, 'rb')
        gz = gzip.GzipFile(fileobj=raw, mode='rb')
        if sys.version_info[0] >= 3:
            import io
            text = io.TextIOWrapper(gz)
        else:
            text = gz
        # Position is tracked on the COMPRESSED stream
        pos_fn = raw.tell
        return text, bar, pos_fn
    else:
        # For plain text: open in binary and decode on the fly so tell() works
        f = open(filename, 'rb')
        if sys.version_info[0] >= 3:
            import io
            # We need readline() to return decoded strings, but we want to
            # tell() on the binary handle.  Wrap in TextIOWrapper but keep
            # the binary handle for tell().
            text = io.TextIOWrapper(f)
            pos_fn = f.tell
            return text, bar, pos_fn
        else:
            return f, bar, f.tell


def CreateMM2AlignmentDataStructure(alignmentfile):
    global mypddf
    fileext = alignmentfile.split('.')[-1]
    is_gz = (fileext == 'gz')
    if is_gz:
        newalignfile = alignmentfile.replace('.paf.gz','.hap')
    else:
        newalignfile = alignmentfile.replace('.paf','.hap')

    fcounter = 0
    mcounter = 0
    purge_self = purge_size = purge_pid = purge_qpct = purge_qrpct = 0

    fin, bar, pos_fn = _open_with_progress(alignmentfile, is_gz, 'Reading PAF')
    last_pos = 0
    try:
        with open(newalignfile, 'w') as fout:
            # IMPORTANT: use readline() in a while loop, NOT `for line in fin`,
            # so that tell() on the underlying binary handle remains valid.
            while True:
                line = fin.readline()
                if not line:
                    break
                # Update progress bar based on bytes consumed
                if bar is not None and fcounter % 1000 == 0:
                    cur = pos_fn()
                    bar.update(cur - last_pos)
                    last_pos = cur
                line = line.strip().split()
                if len(line) < 11:
                    print('Error in reading PAF file')
                    quit(1)
                fcounter+=1
                fqAlignLen = max(int(line[2]), int(line[3])) - min(int(line[2]), int(line[3]))
                frAlignLen = max(int(line[7]), int(line[8])) - min(int(line[7]), int(line[8]))
                fQRAlignLenPct = CalculatePctAlign(fqAlignLen, frAlignLen)
                fQPct = CalculatePctAlign(fqAlignLen, int(line[1]))
                fPID = CalculatePctAlign(int(line[9]), fqAlignLen)
                if line[0] == line[5]:
                    purge_self += 1
                elif int(line[1]) < myMinContigSize:
                    purge_size += 1
                elif fPID < myMinPID:
                    purge_pid += 1
                elif fQPct < myMinQPctMin:
                    purge_qpct += 1
                elif fQRAlignLenPct < myMinQRPctMin:
                    purge_qrpct += 1
                else:
                    mcounter+=1
                    fout.write('"' + line[0] + '"' + '\t' + '"' + line[5] + '"' + '\t' + line[1] + '\t' + str(fQPct) + '\t' + str(fPID) + '\t' + str(fQRAlignLenPct) + '\n')
    finally:
        if bar is not None:
            try:
                cur = pos_fn()
                bar.update(cur - last_pos)
            except (OSError, ValueError):
                pass
            bar.close()
        fin.close()

    _print_purge_breakdown(fcounter, mcounter, purge_self, purge_size,
                            purge_pid, purge_qpct, purge_qrpct)
    with open(newalignfile) as fin:
        myline = fin.readline()
        if len(myline) <= 3:
            print('Empty HAP file. Please fix and rerun')
            quit(1)
    print('Loading HAP file into pandas DataFrame...', flush=True)
    mypddf = pd.read_csv(newalignfile, sep='\t', header=None, names=['qName', 'tName', 'qSize', 'QPct', 'PID', 'QRAlignLenPct'], dtype={'qName': object, 'tName': object})
    print('  Loaded ' + str(len(mypddf)) + ' alignments', flush=True)
    #if len(myline == 18:
        #mypddf = pd.DataFrame(myLines[:],
        #                  columns=['qName', 'qSize', 'qStart', 'qEnd', 'strand', 'tName', 'tSize', 'tStart', 'tEnd',
        #                           'matches', 'gaps+matches', 'mappingqv', 'alignmenttype', 'numofminschain',
        #                           'chainingscore', 'secondchainingscore', 'approxdivergence', 'lqrhrepseeds'])
    #elif len(myline) == 17:
        #mypddf = pd.DataFrame(myLines[:],
        #                  columns=['qName', 'qSize', 'qStart', 'qEnd', 'strand', 'tName', 'tSize', 'tStart', 'tEnd',
        #                           'matches', 'gaps+matches', 'mappingqv', 'alignmenttype', 'numofminschain',
        #                           'chainingscore', 'secondchainingscore', 'approxdivergence'])
    #else:
        #print('Error in PAF file. expected 17 or 18 columns but received ' + str(len(myLines[lineNum])) + ' columns.')
    #myLines = list() #clear this var to release RAM
    #mypddf['qStart'] = pd.to_numeric(mypddf['qStart'])
    #mypddf['qEnd'] = pd.to_numeric(mypddf['qEnd'])
    #mypddf['tStart'] = pd.to_numeric(mypddf['tStart'])
    #mypddf['tEnd'] = pd.to_numeric(mypddf['tEnd'])
    mypddf['qSize'] = pd.to_numeric(mypddf['qSize'])
    mypddf['QPct'] = pd.to_numeric(mypddf['QPct'])
    mypddf['PID'] = pd.to_numeric(mypddf['PID'])
    mypddf['QRAlignLenPct'] = pd.to_numeric(mypddf['QRAlignLenPct'])
    #mypddf['tSize'] = pd.to_numeric(mypddf['tSize'])
    #mypddf['matches'] = pd.to_numeric(mypddf['matches'])
    #mypddf['gaps+matches'] = pd.to_numeric(mypddf['gaps+matches'])
    # mypddf['qName'] = mypddf['qName'].str.replace('|','_').str[0:13]
    # mypddf['tName'] = mypddf['tName'].str.replace('|','_').str[0:13]
    # mypddf['qName'] = mypddf['qName'].str.split('|').str[0]
    # mypddf['tName'] = mypddf['tName'].str.split('|').str[0]
    #mypddf['qMin'] = mypddf[['qStart', 'qEnd']].min(axis=1)
    #mypddf['qMax'] = mypddf[['qStart', 'qEnd']].max(axis=1)
    #mypddf['tMin'] = mypddf[['tStart', 'tEnd']].min(axis=1)
    #mypddf['tMax'] = mypddf[['tStart', 'tEnd']].max(axis=1)
    #mypddf['qAlignLen'] = mypddf['qMax'] - mypddf['qMin']
    #mypddf['rAlignLen'] = mypddf['tMax'] - mypddf['tMin']
    #mypddf['QRAlignLenPct'] = mypddf[['qAlignLen', 'rAlignLen']].apply(lambda x: CalculatePctAlign(*x), axis=1)
    #mypddf['QPct'] = mypddf[['qAlignLen', 'qSize']].apply(lambda x: CalculatePctAlign(*x), axis=1)
    #mypddf['PID'] = mypddf[['matches', 'qAlignLen']].apply(lambda x: CalculatePctAlign(*x), axis=1)
    #mypddf['RPct'] = mypddf[['rAlignLen', 'tSize']].apply(lambda x: CalculatePctAlign(*x), axis=1)
    # Here we create a mask where qryname != refname
    #lenbeforemask = len(mypddf)
    #mypddf = mypddf[mypddf['qName'] != mypddf['tName']]
    #lenaftermask = len(mypddf)
    #print(str(lenbeforemask - lenaftermask) + ' alignments Purged where query = reference')
    #lenbeforemask = len(mypddf)
    #mypddf = mypddf[['qName', 'qSize', 'QPct', 'PID', 'QRAlignLenPct']]
    #mypddf = mypddf[mypddf['qSize'] >= myMinContigSize]
    #mypddf = mypddf[mypddf['PID'] >= myMinPID]
    #mypddf = mypddf[mypddf['QPct'] >= myMinQPctMin]
    #mypddf = mypddf[mypddf['QRAlignLenPct'] >= myMinQRPctMin]
    #mypddf = mypddf[mypddf['QRAlignLenPct'] <= CalculateInverseProportion(myMinQRPctMin)]
    #lenaftermask = len(mypddf)
    #print(str(lenbeforemask - lenaftermask) + ' alignments Purged due to Search Space constraints')
    return mypddf


# Create a dictionary based on the alignment file
def CreateBlatAlignmentDataStructure(alignmentfile):
    global mypddf
    fileext = alignmentfile.split('.')[-1]
    is_gz = (fileext == 'gz')
    if is_gz:
        newalignfile = alignmentfile.replace('.psl.gz','.hap')
    else:
        newalignfile = alignmentfile.replace('.psl','.hap')

    fcounter = 0
    mcounter = 0
    mylinenum = 0
    purge_self = purge_size = purge_pid = purge_qpct = purge_qrpct = 0

    fin, bar, pos_fn = _open_with_progress(alignmentfile, is_gz, 'Reading PSL')
    last_pos = 0
    try:
        with open(newalignfile, 'w') as fout:
            # IMPORTANT: use readline() in a while loop, NOT `for line in fin`,
            # so that tell() on the underlying binary handle remains valid.
            while True:
                line = fin.readline()
                if not line:
                    break
                mylinenum += 1
                if bar is not None and mylinenum % 1000 == 0:
                    cur = pos_fn()
                    bar.update(cur - last_pos)
                    last_pos = cur
                line = line.strip().split()
                if len(line) < 21 and mylinenum > 5:
                    print('Error in reading PSL file. Length of line < 21')
                    print(line)
                    quit(1)
                elif mylinenum > 5:
                    fcounter+=1
                    fqAlignLen = max(int(line[11]), int(line[12])) - min(int(line[11]), int(line[12]))
                    frAlignLen = max(int(line[15]), int(line[16])) - min(int(line[15]), int(line[16]))
                    fQRAlignLenPct = CalculatePctAlign(fqAlignLen, frAlignLen)
                    fQPct = CalculatePctAlign(fqAlignLen, int(line[10]))
                    fPID = CalculatePctAlign(int(line[0]), fqAlignLen)
                    if line[9] == line[13]:
                        purge_self += 1
                    elif int(line[10]) < myMinContigSize:
                        purge_size += 1
                    elif fPID < myMinPID:
                        purge_pid += 1
                    elif fQPct < myMinQPctMin:
                        purge_qpct += 1
                    elif fQRAlignLenPct < myMinQRPctMin:
                        purge_qrpct += 1
                    else:
                        mcounter+=1
                        fout.write('"' + line[9] + '"' + '\t' + '"' + line[13] + '"' + '\t' + line[10] + '\t' + str(fQPct) + '\t' + str(fPID) + '\t' + str(fQRAlignLenPct) + '\n')
    finally:
        if bar is not None:
            try:
                cur = pos_fn()
                bar.update(cur - last_pos)
            except (OSError, ValueError):
                pass
            bar.close()
        fin.close()

    _print_purge_breakdown(fcounter, mcounter, purge_self, purge_size,
                            purge_pid, purge_qpct, purge_qrpct)
    with open(newalignfile) as fin:
        myline = fin.readline()
        if len(myline) <= 3:
            print('Empty HAP file. Please fix and rerun')
            quit(1)
    print('Loading HAP file into pandas DataFrame...', flush=True)
    mypddf = pd.read_csv(newalignfile, sep='\t', header=None, names=['qName', 'tName', 'qSize', 'QPct', 'PID', 'QRAlignLenPct'], dtype={'qName': object, 'tName': object})
    print('  Loaded ' + str(len(mypddf)) + ' alignments', flush=True)
    #for lineNum in range(0, len(myLines)):
    #    if myLines[lineNum][0] == '-':
    #        mystop = lineNum + 1
    #        break
    #if (mystop - 1) == len(myLines) or mystop == 0:
    #    print('Invalid PSL format. Missing ----\n. Please fix and rerun')
    #    quit(1)
    #mypddf = pd.DataFrame(myLines[mystop-1:],
    #                      columns=['matches', 'misMatches', 'repMatches', 'nCount', 'qNumInsert', 'qBaseInsert',
    #                               'tNumInsert', 'tBaseInsert', 'strand', 'qName', 'qSize', 'qStart', 'qEnd', 'tName',
    #                               'tSize', 'tStart', 'tEnd'])
    #myLines = list() #clear this var to release RAM
    mypddf['qSize'] = pd.to_numeric(mypddf['qSize'])
    mypddf['QPct'] = pd.to_numeric(mypddf['QPct'])
    mypddf['PID'] = pd.to_numeric(mypddf['PID'])
    mypddf['QRAlignLenPct'] = pd.to_numeric(mypddf['QRAlignLenPct'])
    # mypddf['misMatches'] = pd.to_numeric(mypddf['misMatches'])
    # mypddf['repMatches'] = pd.to_numeric(mypddf['repMatches'])
    # mypddf['nCount'] = pd.to_numeric(mypddf['nCount'])
    # mypddf['qBaseInsert'] = pd.to_numeric((mypddf['qBaseInsert']))
    # mypddf['qName'] = mypddf['qName'].str.replace('|','_').str[0:13]
    # mypddf['tName'] = mypddf['tName'].str.replace('|','_').str[0:13]
    # mypddf['qName'] = mypddf['qName'].str.split('|').str[0]
    # mypddf['tName'] = mypddf['tName'].str.split('|').str[0]
    #mypddf['qMin'] = mypddf[['qStart', 'qEnd']].min(axis=1)
    #mypddf['qMax'] = mypddf[['qStart', 'qEnd']].max(axis=1)
    #mypddf['tMin'] = mypddf[['tStart', 'tEnd']].min(axis=1)
    #mypddf['tMax'] = mypddf[['tStart', 'tEnd']].max(axis=1)
    #mypddf['qAlignLen'] = mypddf['qMax'] - mypddf['qMin']
    #mypddf['rAlignLen'] = mypddf['tMax'] - mypddf['tMin']
    #mypddf['QRAlignLenPct'] = mypddf[['qAlignLen', 'rAlignLen']].apply(lambda x: CalculatePctAlign(*x), axis=1)
    #mypddf['QPct'] = mypddf[['qAlignLen', 'qSize']].apply(lambda x: CalculatePctAlign(*x), axis=1)
    #mypddf['PID'] = mypddf[['matches', 'qAlignLen']].apply(lambda x: CalculatePctAlign(*x), axis=1)
    #mypddf['RPct'] = mypddf[['rAlignLen', 'tSize']].apply(lambda x: CalculatePctAlign(*x), axis=1)
    # Here we create a mask where qryname != refname
    #lenbeforemask = len(mypddf)
    #mypddf = mypddf[mypddf['qName'] != mypddf['tName']]
    #lenaftermask = len(mypddf)
    #print(str(lenbeforemask - lenaftermask) + ' alignments Purged where query = reference')
    #lenbeforemask = len(mypddf)
    #mypddf = mypddf[['qName', 'qSize', 'QPct', 'PID', 'QRAlignLenPct']]
    #mypddf = mypddf[mypddf['qSize'] >= myMinContigSize]
    #mypddf = mypddf[mypddf['PID'] >= myMinPID]
    #mypddf = mypddf[mypddf['QPct'] >= myMinQPctMin]
    #mypddf = mypddf[mypddf['QRAlignLenPct'] >= myMinQRPctMin]
    #mypddf = mypddf[mypddf['QRAlignLenPct'] <= CalculateInverseProportion(myMinQRPctMin)]
    #lenaftermask = len(mypddf)
    #print(str(lenbeforemask - lenaftermask) + ' alignments Purged due to Search Space constraints')
    return mypddf


def myLinearFxn(mbusco, sbusco, dbusco, fbusco, cbusco):
    myValue = float(thetaF * fbusco + thetaD * dbusco + thetaM * mbusco) / float(thetaS * sbusco)
    # todo: add custom linear function option
    return myValue


def sanitize_name(name):
    """Sanitize a contig name to match preprocessfasta.py logic.
    Replaces all non-alphanumeric characters (except .) with underscores."""
    return re.sub('[^a-zA-Z0-9.]', '_', name)


def build_conversion_dict(canonical_names, external_names):
    """Build a mapping from external names to canonical (FASTA) names.

    Tries exact match, then sanitized match, then prefix match
    (for truncated names from preprocessfasta.py).
    Returns (conversion_dict, unmatched_set).
    """
    conversion = dict()
    unmatched = set()
    canonical_set = set(canonical_names)

    # Build sanitized lookup: sanitized_name -> canonical_name
    sanitized_lookup = dict()
    for name in canonical_names:
        san = sanitize_name(name)
        if san in sanitized_lookup:
            sanitized_lookup[san] = None  # Ambiguous
        else:
            sanitized_lookup[san] = name

    for ext_name in external_names:
        if ext_name in canonical_set:
            continue  # Exact match, no conversion needed

        san_ext = sanitize_name(ext_name)

        # Try exact sanitized match
        if san_ext in sanitized_lookup and sanitized_lookup[san_ext] is not None:
            conversion[ext_name] = sanitized_lookup[san_ext]
            continue

        # Try prefix match (handles truncated names from preprocessfasta.py)
        prefix_matches = []
        for san_canon, canon in sanitized_lookup.items():
            if canon is None:
                continue
            if len(san_ext) > 0 and len(san_canon) > 0:
                if san_ext.startswith(san_canon) or san_canon.startswith(san_ext):
                    prefix_matches.append(canon)

        if len(prefix_matches) == 1:
            conversion[ext_name] = prefix_matches[0]
        elif len(prefix_matches) > 1:
            print('Warning: ambiguous prefix match for "' + ext_name + '", skipping: ' + str(prefix_matches))
            unmatched.add(ext_name)
        else:
            unmatched.add(ext_name)

    return conversion, unmatched


def WriteNewAssembly(myasmFileName, newASMFileName, myGoodContigsSet):
    mydirectory = 'asms'
    outfile = mydirectory + '/' + newASMFileName
    if not os.path.exists(mydirectory):
        os.makedirs(mydirectory)
    myGoodContigsSet = myGoodContigsSet - {''}
    # contigsDict[key] = [contiglen,headerpos,startseqpos,endseqpos]
    if len(myContigsDict) == 0:
        print('myContigsDict is empty! Please make sure your assembly fasta file is not empty. If not empty then post a question with output at https://github.com/esolares/HapSolo/issues Along with the following output:')
        print(myContigsDict)
        quit(2)
    mySetDiff = myGoodContigsSet - set(myContigsDict.keys())
    mySetDiffLen = len(mySetDiff)
    if mySetDiffLen != 0:
        print('Error: HapSolo has two seperate set of contigs! Please submit bug report and sent bugreport.log file at https://github.com/esolares/HapSolo/issues.')
        with open('bugreport.log', 'w') as foutlogfile:
            foutlogfile.write(myscerrorlog + '\n')
            foutlogfile.write('Begin ContigsDict keys with ' + str(len(myContigsDict.keys())) + ' # of keys:\n')
            for key in myContigsDict.keys():
                foutlogfile.write('"' + str(key) + '",')
            foutlogfile.write('\nEnd ContigsDict keys\n\n')
            foutlogfile.write('Begin good contig set with ' + str(len(myGoodContigsSet)) + ' # of elements:\n')
            for contig in myGoodContigsSet:
                foutlogfile.write('"' + str(contig) + '",')
            foutlogfile.write('\nEnd good contig set\n\n')
            foutlogfile.write('Begin non-matching contig set with ' + str(mySetDiffLen) + ' # of elements:\n')
            for contig in mySetDiff:
                foutlogfile.write('"' + str(contig) + '",')
            foutlogfile.write('\nEnd non-matching contig set\n\n')
        quit(1)
    with open(myasmFileName, 'r') as fin, open(outfile, 'w') as fout:
        for contig in myGoodContigsSet:
            myContigPositionsList = myContigsDict[contig]
            fin.seek(myContigPositionsList[1])  # extract headerpos
            fout.write(fin.readline())
            newPos = fin.tell()
            mySeq = fin.readline().replace('\n', '')
            while newPos != myContigPositionsList[3]:
                newPos = fin.tell()
                mySeq = mySeq + fin.readline().replace('\n', '')
            fout.write(mySeq + '\n')


if __name__ == '__main__':
    seed(1)
    try:
        myContigsDict = CalculateContigSizes(myasmFileName)
    except (IOError, OSError) as e:
        print('Error reading assembly file: ' + str(e))
        quit(1)
    except (IndexError, ValueError) as e:
        print('Error parsing assembly file (malformed FASTA?): ' + str(e))
        quit(1)
    for key in myContigsDict.keys():
        if myContigsDict[key][0] < myMinContigSize:
            smallcontigset.add(key)
    if pslalignmentfile is None:
        mypddf = CreateMM2AlignmentDataStructure(pafalignmentfile)
    elif pafalignmentfile is None:
        mypddf = CreateBlatAlignmentDataStructure(pslalignmentfile)
    # Check for contig name mismatches between FASTA and alignment file
    canonical_names = set(myContigsDict.keys())
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
    missingrefcontigset = set(myContigsDict.keys()) - qrycontigset
    allcontigsset = set(myContigsDict.keys())
    busco2contigdict, contigs2buscodict = importBuscos(buscofileloc)
    # Check for contig name mismatches between FASTA and BUSCO results
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
        for buscoid in busco2contigdict:
            for buscotype in buscotypes:
                busco2contigdict[buscoid][buscotype] = [busco_conversion.get(n, n) for n in busco2contigdict[buscoid][buscotype]]
    if busco_unmatched:
        print('Warning: ' + str(len(busco_unmatched)) + ' BUSCO contig name(s) could not be matched to assembly:')
        for name in sorted(busco_unmatched):
            print('  ' + name)
    # execute Hill Climbing here.
    job_args = list()
    if mode != 1:
        if threads == 1:
            job_args = [0, iterations, resolution, uniform(myMinPID, 1), uniform(myMinQPctMin, 1), uniform(myMinQRPctMin, 1)]
            mylist = hillclimbing(job_args)
            if HAS_TQDM:
                sys.stderr.write('\n')
                sys.stderr.flush()
            mybestnscoreslist = mylist[0]
            for i in range(0, min(bestnscores, len(mybestnscoreslist))):
                for j in range(0, len(mybestnscoreslist[0][4])):
                    mybestnscoreslist[i][4][j] = '%.4f' % mybestnscoreslist[i][4][j]
                newasmfilename = myasmFileName.replace('.fasta', '') + '_' + str(myMinContigSize) + '_' + str(
                    mybestnscoreslist[i][4][0]) + '_' + str(mybestnscoreslist[i][4][2]) + 'to' + str(
                    '%.4f' % CalculateInverseProportion(float(mybestnscoreslist[i][4][2]))) + '_' + str(
                    mybestnscoreslist[i][4][1]) + '_primary.fasta'
                print('Writing ' + newasmfilename + ' with score: ' + str(mybestnscoreslist[i][0]))
                WriteNewAssembly(myasmFileName, newasmfilename, mybestnscoreslist[i][1])
                WriteNewAssembly(myasmFileName, newasmfilename.replace('_primary.fasta', '_secondary.fasta'), mybestnscoreslist[i][2])
                write_report('asms/' + newasmfilename, mybestnscoreslist[i][1], mybestnscoreslist[i][3])
            if dumpscores:
                with open(myasmFileName.replace('.fasta', '_' + str(datetime.datetime.today()).replace(' ', '_').replace('-', '_').replace(':', '_').split('.')[0] + '.scores'), 'w') as fout:
                    fout.write(str(mylist[1][0]))
                    for i in range(1, iterations):
                        fout.write(',' + str(mylist[1][i]))
                with open(myasmFileName.replace('.fasta', '_' + str(datetime.datetime.today()).replace(' ', '_').replace('-', '_').replace(':', '_').split('.')[0] + '.deltascores'), 'w') as fout:
                    fout.write(str(mylist[2][0]))
                    for i in range(1, iterations):
                        fout.write(',' + str(mylist[2][i]))
        elif threads > 1:
            for i in range(threads):
                job_args.append([i, iterations, resolution, uniform(myMinPID, 1), uniform(myMinQPctMin, 1), uniform(myMinQRPctMin, 1)])
            # Initialize pool with shared tqdm lock so per-thread progress
            # bars don't garble each other on the terminal.
            if HAS_TQDM:
                pool = mp.Pool(processes=threads,
                               initializer=tqdm.set_lock,
                               initargs=(tqdm.get_lock(),))
            else:
                pool = mp.Pool(processes=threads)
            mylist = pool.map(hillclimbing, job_args)
            pool.close()
            pool.join()
            # Position cursor below all progress bars so subsequent output
            # doesn't overwrite them. tqdm with `position=N` reserves rows 0..N
            # below the cursor, so we need to advance the cursor past row
            # `threads - 1` (a single newline does this since we are already
            # at row 0 after the bars are done refreshing).
            if HAS_TQDM:
                sys.stderr.write('\n' * threads)
                sys.stderr.flush()
            mybestnscoreslist = list()
            mybestnscoreslist.append(mylist[0][0][0])
            for i in range(0, threads):
                for j in range(0, min(bestnscores, len(mylist[i][0]))):
                    mybestnscoreslist = uniquepriorityqueue(mybestnscoreslist, mylist[i][0][j])
            for i in range(0, min(bestnscores, len(mybestnscoreslist))):
                for j in range(0, len(mybestnscoreslist[0][4])):
                    mybestnscoreslist[i][4][j] = '%.4f' % mybestnscoreslist[i][4][j]
                newasmfilename = myasmFileName.replace('.fasta', '') + '_' + str(myMinContigSize) + '_' + str(
                    mybestnscoreslist[i][4][0]) + '_' + str(mybestnscoreslist[i][4][2]) + 'to' + str(
                    '%.4f' % CalculateInverseProportion(float(mybestnscoreslist[i][4][2]))) + '_' + str(
                    mybestnscoreslist[i][4][1]) + '_primary.fasta'
                print('Writing ' + newasmfilename + ' with score: ' + str(mybestnscoreslist[i][0]))
                WriteNewAssembly(myasmFileName, newasmfilename, mybestnscoreslist[i][1])
                WriteNewAssembly(myasmFileName, newasmfilename.replace('_primary.fasta', '_secondary.fasta'), mybestnscoreslist[i][2])
                write_report('asms/' + newasmfilename, mybestnscoreslist[i][1], mybestnscoreslist[i][3])
            if dumpscores:
                with open(myasmFileName.replace('.fasta', '_' + str(datetime.datetime.today()).replace(' ', '_').replace('-', '_').replace(':', '_').split('.')[0] + '.scores'), 'w') as fout:
                    for i in range(0, threads):
                        fout.write(str(mylist[i][1][0]))
                        for j in range(1, iterations):
                            fout.write(',' + str(mylist[i][1][j]))
                        fout.write('\n')
                with open(myasmFileName.replace('.fasta', '_' + str(datetime.datetime.today()).replace(' ', '_').replace('-', '_').replace(':', '_').split('.')[0] + '.deltascores'), 'w') as fout:
                    for i in range(0, threads):
                        fout.write(str(mylist[i][2][0]))
                        for j in range(1, iterations):
                            fout.write(',' + str(mylist[i][2][j]))
                        fout.write('\n')
        else:
            print('Invalid # of threads set. Please use a positive integer for threads')
            quit(1)
    elif mode == 1:
        job_args = [0, 1, resolution, customMinPID, customMinQPctMin, customMinQRPctMin]
        mylist = hillclimbing(job_args)
        mybestnscoreslist = mylist[0]
        for i in range(0, min(bestnscores, len(mybestnscoreslist))):
            for j in range(0, len(mybestnscoreslist[0][4])):
                mybestnscoreslist[i][4][j] = '%.4f' % mybestnscoreslist[i][4][j]
            newasmfilename = myasmFileName.replace('.fasta', '') + '_' + str(myMinContigSize) + '_' + str(
                mybestnscoreslist[i][4][0]) + '_' + str(mybestnscoreslist[i][4][2]) + 'to' + str(
                '%.4f' % CalculateInverseProportion(float(mybestnscoreslist[i][4][2]))) + '_' + str(
                mybestnscoreslist[i][4][1]) + '_primary.fasta'
            print('Writing ' + newasmfilename + ' with score: ' + str(mybestnscoreslist[i][0]))
            WriteNewAssembly(myasmFileName, newasmfilename, mybestnscoreslist[i][1])
            WriteNewAssembly(myasmFileName, newasmfilename.replace('_primary.fasta', '_secondary.fasta'), mybestnscoreslist[i][2])
            write_report('asms/' + newasmfilename, mybestnscoreslist[i][1], mybestnscoreslist[i][3])
