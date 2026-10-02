import os


def CalculateContigSizes(asmFileName):
    """Parse FASTA file to extract contig names, sizes, and file positions.

    Returns dict[contigname] = [contiglen, headerpos, startseqpos, endseqpos].
    """
    error_log = ''
    myContigSizeDict = dict()
    with open(asmFileName) as fin:
        lastPos = headerPos = fin.tell()
        totalLines = sum(1 for line in fin)
        fin.seek(lastPos)
        seqLen = 0
        seqName = ''
        lastPos = 0
        count = 0
        while count < totalLines:
            lastPos = headerPos = fin.tell()
            line = fin.readline().replace('\n', '')
            count = count + 1
            if line[0:1] == '>':
                header = line[1:]
                if len(header.split(" ")) > 1:
                    raise ValueError(
                        'Spaces found in contig headers. Please remove spaces '
                        'from contig names before proceeding with any analysis. '
                        'Spaces, -"s, //"s and other special characters are not '
                        'allowed in contig names.'
                    )
                seqName = header.split(" ")[0].replace('/', '_')
                lastPos = startPos = fin.tell()
                line = fin.readline().replace('\n', '')
                count = count + 1
                while line[0:1] != '>' and line[0:1] != '':
                    seqLen = seqLen + len(line)
                    endPos = lastPos
                    lastPos = fin.tell()
                    line = fin.readline().replace('\n', '')
                    count = count + 1
                if line[0:1] == '>' or line[0:1] == '':
                    myContigSizeDict[seqName] = [seqLen, headerPos, startPos, endPos]
                    seqName = ''
                    seqLen = 0
                    count = count - 1
                    fin.seek(lastPos)
    return myContigSizeDict, error_log


def calculateasmstats(bestcontigset, contig_sizes):
    """Compute assembly stats (total size, N50, L50, largest contig)."""
    mycontiglist = list()
    for contig in bestcontigset:
        if contig in contig_sizes:
            mycontiglist.append(contig_sizes[contig][0])
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


def WriteNewAssembly(myasmFileName, newASMFileName, myGoodContigsSet, contig_sizes, error_log='', outdir='asms'):
    """Write filtered FASTA file containing only selected contigs."""
    outfile = outdir + '/' + newASMFileName
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    myGoodContigsSet = myGoodContigsSet - {''}
    if len(contig_sizes) == 0:
        raise ValueError(
            'contig_sizes is empty! Please make sure your assembly fasta '
            'file is not empty. If not empty then post a question with output '
            'at https://github.com/esolares/HapSolo/issues'
        )
    mySetDiff = myGoodContigsSet - set(contig_sizes.keys())
    mySetDiffLen = len(mySetDiff)
    if mySetDiffLen != 0:
        print('Error: HapSolo has two seperate set of contigs! Please submit bug report and sent bugreport.log file at https://github.com/esolares/HapSolo/issues.')
        with open('bugreport.log', 'w') as foutlogfile:
            foutlogfile.write(error_log + '\n')
            foutlogfile.write('Begin ContigsDict keys with ' + str(len(contig_sizes.keys())) + ' # of keys:\n')
            for key in contig_sizes.keys():
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
        raise RuntimeError(
            'Error: HapSolo has two separate set of contigs! Please submit bug '
            'report and send bugreport.log file at '
            'https://github.com/esolares/HapSolo/issues.'
        )
    # Write in input-assembly order: contig_sizes is filled while reading the FASTA, so
    # its key order is file order. Iterating the set instead gave a different contig
    # order on every run (per-process str hashing), so identical runs had different md5s;
    # file order also makes the seeks sequential (~19% faster on a 3.2 Gb assembly).
    with open(myasmFileName, 'r') as fin, open(outfile, 'w') as fout:
        for contig in (c for c in contig_sizes if c in myGoodContigsSet):
            myContigPositionsList = contig_sizes[contig]
            fin.seek(myContigPositionsList[1])
            fout.write(fin.readline())
            newPos = fin.tell()
            mySeq = fin.readline().replace('\n', '')
            while newPos != myContigPositionsList[3]:
                newPos = fin.tell()
                mySeq = mySeq + fin.readline().replace('\n', '')
            fout.write(mySeq + '\n')
