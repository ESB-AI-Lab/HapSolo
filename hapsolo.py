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
    mybuscofiles = glob.glob(buscofileloc + '/busco*/*/full_table_*.tsv')
    if len(mybuscofiles) == 0:
        print('No BUSCO result files found matching: ' + buscofileloc + '/busco*/*/full_table_*.tsv')
        print('Please verify the BUSCO output directory path and that BUSCO has completed successfully.')
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
    # todo: remove resolution as it is already a global var. adjust job args
    resolution = job_args[2]
    myPID = job_args[3]
    myQPctMin = job_args[4]
    myQRPctMin = job_args[5]
    # parameters
    bestnscoreslist = list()
    mysteps = list()
    mysteps.append(0)
    mysteps.append(0)
    mysteps.append(0)
    # Set up per-thread progress bar (one line per core, updated in place).
    # Falls back to silent mode if tqdm isn't installed.
    pbar = None
    if HAS_TQDM and mode != 1:
        pbar = tqdm(
            total=numofiterations,
            position=mythread,
            desc='JOBID: ' + str(mythread),
            bar_format='{desc} [{bar:30}] {n_fmt}/{total_fmt} {postfix}',
            leave=True,
            dynamic_ncols=True,
            # Update at most every 0.2s to avoid terminal thrashing with many threads
            mininterval=0.2,
            # Force a refresh on every update so progress doesn't get out of order
            miniters=1,
        )
        pbar.set_postfix_str(
            'PID: ' + ('%.4f' % myPID)
            + ' QPMin: ' + ('%.4f' % myQPctMin)
            + ' QRPMin: ' + ('%.4f' % myQRPctMin)
            + ' CostΔ ' + ('%+.4f' % 0.0)
            + ' Score: ' + ('%.4f' % 0.0))
    costfxn = [0.0 for ii in range(numofiterations)]
    costfxndelta = [0.0 for ii in range(numofiterations)]
    # 0. Calculate scores for original assembly
    allmycontigs = qrycontigset.union(missingrefcontigset) - smallcontigset - {''}
    allcontigsbuscoscore = calculateBuscos(allmycontigs, busco2contigdict, contigs2buscodict)
    allsinglebuscos = allcontigsbuscoscore['S']
    allmissingbuscos = allcontigsbuscoscore['M']
    alldupebuscos = allcontigsbuscoscore['D']
    allfragbuscos = allcontigsbuscoscore['F']
    totalbuscos = allcontigsbuscoscore['C'] + allcontigsbuscoscore['M'] + allcontigsbuscoscore['F']
    # allasmsize = 0
    if allsinglebuscos == 0:
        oldasmscorefxn = 5000.0
    else:
        oldasmscorefxn = myLinearFxn(allmissingbuscos, allsinglebuscos, alldupebuscos, allfragbuscos, totalbuscos)
    bestcontigset = allmycontigs.copy()
    bestpurgedset = allcontigsset - bestcontigset  - {''}
    # bestscore = oldasmscorefxn
    bestbuscos = allcontigsbuscoscore.copy()
    # use fxn uniquepriorityqueue(pqlist, myvalues) for returning a sorted unique priority list
    myvalues = [oldasmscorefxn, bestcontigset, bestpurgedset, bestbuscos, [0.0, 0.0, 0.0]]
    # uniquepriorityqueue(bestnscoreslist[bestnscoreidx], [score, setofgoodcontigs, setofmissingcontigs, listofparameters, buscos])
    if mode != 1:
        bestnscoreslist.append(myvalues)
    # process:
    # 1. Make a step
    # 2. Calculate new assembly
    mygoodcontigs = ReduceASM(myPID, myQPctMin, myQRPctMin)
    mygoodcontigs = mygoodcontigs.union(missingrefcontigset) - smallcontigset - {''}
    purgedcontigs = allcontigsset - mygoodcontigs - {''}
    # 3. Calculate new busco scores
    mygoodcontigsbuscoscore = calculateBuscos(mygoodcontigs, busco2contigdict, contigs2buscodict)
    newsinglebuscos = mygoodcontigsbuscoscore['S']
    newmissingbuscos = mygoodcontigsbuscoscore['M']
    newdupebuscos = mygoodcontigsbuscoscore['D']
    newfragbuscos = mygoodcontigsbuscoscore['F']
    if newsinglebuscos == 0:
        newasmscorefxn = 5000.0
    else:
        newasmscorefxn = myLinearFxn(newmissingbuscos, newsinglebuscos, newdupebuscos, newfragbuscos, totalbuscos)
    costfxn[0] = newasmscorefxn
    costfxndelta[0] = newasmscorefxn
    myvalues = [newasmscorefxn, mygoodcontigs, purgedcontigs, mygoodcontigsbuscoscore, [myPID, myQPctMin, myQRPctMin]]
    bestnscoreslist = uniquepriorityqueue(bestnscoreslist, myvalues)
    # Account for iteration 0 in the progress bar so the final count matches total.
    if pbar is not None:
        pbar.set_postfix_str(
            'PID: ' + ('%.4f' % myPID)
            + ' QPMin: ' + ('%.4f' % myQPctMin)
            + ' QRPMin: ' + ('%.4f' % myQRPctMin)
            + ' CostΔ ' + ('%+.4f' % costfxndelta[0])
            + ' Score: ' + ('%.4f' % newasmscorefxn))
        pbar.update(1)
    #if useprimaryformula:
    #    newasmscorefxn = myLinearFxn(newmissingbuscos, newsinglebuscos, newdupebuscos, newfragbuscos, totalbuscos)
    #else:
    #    newasmscorefxn = myLinearFxn(newmissingbuscos, newsinglebuscos, newdupebuscos, newfragbuscos, totalbuscos)
    # 4. use cost function of previous busco scores with new busco scores
    # 5. make new step trying to minimize cost function of busco scores.
    if mode == 1:
        return [bestnscoreslist, costfxn, costfxndelta]
    # Implement random forward walking, optimized decision based walking, other AI's here
    #if aimode == 0:
        # call rfw function
    #elif aimode == 1:
        # call odbw function
    #elif aimode == 2:
        # call another function
    #else:
        # throw an error
    for i in range(1, numofiterations):
        # forward stepping of GD
        if (myPID > 1.0 and myQPctMin > 1.0 and myQRPctMin > 1.0) or (i >= maxzeros and all(abs(costfxndelta[k]) <= resolution for k in range(i-maxzeros+1, i+1))):
            # reassign myQ's
            myPID = uniform(myMinPID, 1.0)
            myQPctMin = uniform(myMinQPctMin, 1)
            myQRPctMin = uniform(myMinQRPctMin, 1)
        elif myQPctMin > 1.0 and myQRPctMin > 1.0:
            myQPctMin = uniform(myMinQPctMin, 1)
            myQRPctMin = uniform(myMinQRPctMin, 1)
        elif myPID > 1.0 and myQPctMin > 1.0:
            myPID = uniform(myMinPID, 1.0)
            myQPctMin = uniform(myMinQPctMin, 1)
        elif myPID > 1.0 and myQRPctMin > 1.0:
            myPID = uniform(myMinPID, 1.0)
            myQRPctMin = uniform(myMinQRPctMin, 1)
        elif myPID > 1.0:
            # reassign %ID
            myPID = uniform(myMinPID, 1.0)
        elif myQPctMin > 1.0:
            # reassign myQPctMin
            myQPctMin = uniform(myMinQPctMin, 1)
        elif myQRPctMin > 1.0:
            # reassign myQRPctMin
            myQRPctMin = uniform(myMinQRPctMin, 1)
        else:
            mystepindex = randint(0, len(mysteps)-1)
            mysteps[mystepindex] = stepsize
            while True:
                # print(str(mysteps[0]) + ':' + str(mysteps[1]))
                for j in range(0, len(mysteps)):
                    myrand = randint(0, len(mysteps)-1)
                    mysteps[j] = stepsize * myrand
                if mysteps[0] != 0.0 or mysteps[1] != 0.0 or mysteps[2] != 0.0:
                    break
            myPID = myPID + mysteps[0]
            myQPctMin = myQPctMin + mysteps[1]
            myQRPctMin = myQRPctMin + mysteps[2]
        mygoodcontigs = ReduceASM(myPID, myQPctMin, myQRPctMin)
        # include missing contigs > 10Mb
        mygoodcontigs = mygoodcontigs.union(missingrefcontigset) - smallcontigset - {''}
        purgedcontigs = allcontigsset - mygoodcontigs - {''}
        mygoodcontigsbuscoscore = calculateBuscos(mygoodcontigs, busco2contigdict, contigs2buscodict)
        newsinglebuscos = mygoodcontigsbuscoscore['S']
        newmissingbuscos = mygoodcontigsbuscoscore['M']
        newdupebuscos = mygoodcontigsbuscoscore['D']
        newfragbuscos = mygoodcontigsbuscoscore['F']
        # cost function here
        if newsinglebuscos == 0:
            newasmscorefxn = 50000000.0
        else:
            newasmscorefxn = myLinearFxn(newmissingbuscos, newsinglebuscos, newdupebuscos, newfragbuscos, totalbuscos)
        costfxn[i] = newasmscorefxn
        costfxndelta[i] = costfxn[i-1] - costfxn[i]
        if pbar is not None:
            pbar.set_postfix_str(
                'PID: ' + ('%.4f' % myPID)
                + ' QPMin: ' + ('%.4f' % myQPctMin)
                + ' QRPMin: ' + ('%.4f' % myQRPctMin)
                + ' CostΔ ' + ('%+.4f' % costfxndelta[i])
                + ' Score: ' + ('%.4f' % newasmscorefxn))
            pbar.update(1)
        # NOTE: no convergence break here. The random-walk algorithm needs to
        # explore the full iteration budget. The maxzeros mechanism above (line
        # ~428) resets thresholds when stuck in a local minimum, which is the
        # actual convergence-escape strategy. A naive delta-based break exits
        # prematurely on small random steps.
        if costfxndelta[i] < 0.0 and costfxndelta[i-1] < 0.0:
            while mysteps[0] != 0 or mysteps[1] != 0:
                for j in range(0, len(mysteps)):
                    myrand = randint(0, len(mysteps)-1)
                    mysteps[j] = stepsize * myrand
        # store new data into data structure unique priority queue
        myvalues = [newasmscorefxn, mygoodcontigs, purgedcontigs, mygoodcontigsbuscoscore,
                    [myPID, myQPctMin, myQRPctMin]]
        if len(bestnscoreslist) < bestnscores:
            bestnscoreslist = uniquepriorityqueue(bestnscoreslist, myvalues)
        else:
            if bestnscoreslist[bestnscores-1][0] >= newasmscorefxn:
                bestnscoreslist = uniquepriorityqueue(bestnscoreslist, myvalues)
    if pbar is not None:
        pbar.close()
    return [bestnscoreslist, costfxn, costfxndelta]


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


def uniquepriorityqueue(pqlist, myvalue):
    pqlist = pqlist[:]
    pqlist.append(myvalue)
    pqlist = sorted(pqlist, key=lambda x: x[0])
    while True:
        #print(pqlist)
        mydupes = set()
        realdupes = list()
        mysamesizesets = list()
        mymasterdupelist = list()
        myfinaldupelist = list()
        for i in range(0, len(pqlist)):
            mysamesizesets.append([len(pqlist[i][1]), i])
        mysamesizesets = sorted(mysamesizesets)
        for i in range(1, len(mysamesizesets)):
            if mysamesizesets[i-1][0] == mysamesizesets[i][0]:
                mydupes.add(mysamesizesets[i-1][1])
                mydupes.add(mysamesizesets[i][1])
            else:
                if len(mydupes) > 0:
                    mymasterdupelist.append(mydupes)
                    mydupes = set()
        if len(mydupes) > 0:
            mymasterdupelist.append(mydupes)
        for i in range(0, len(mymasterdupelist)):
            templist = list(mymasterdupelist[i])
            for j in range(0, len(templist)):
                for k in range(j, len(templist)):
                    if j != k:
                        myfinaldupelist.append((templist[j], templist[k]))
        # here we remove the dupes. Keep lowest score
        for i in range(0, len(myfinaldupelist)):
            if pqlist[myfinaldupelist[i][0]][1] == pqlist[myfinaldupelist[i][1]][1]:
                realdupes.append(i)
        if len(realdupes) == 0:
            return pqlist[0:bestnscores]
        else:
            # Remove only ONE duplicate per pass to avoid stale indices,
            # then restart the while loop to recalculate fresh indices.
            i = realdupes[0]
            idx_a = myfinaldupelist[i][0]
            idx_b = myfinaldupelist[i][1]
            if pqlist[idx_a][0] < pqlist[idx_b][0]:
                pqlist = pqlist[0:idx_b] + pqlist[idx_b + 1:]
            else:
                pqlist = pqlist[0:idx_a] + pqlist[idx_a + 1:]


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
