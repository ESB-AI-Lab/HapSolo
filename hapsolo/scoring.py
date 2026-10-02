import glob
import os

BUSCO_TYPES = ['C', 'S', 'D', 'F', 'M']


def import_orthologs_tsv(filepath):
    """Load ortholog classification from a single consolidated TSV file.

    Reads the format produced by search_orthologs.py (full_table_results.tsv):
      # header lines starting with #
      BuscoID  Status  Contig  Start  End  Score  Length
      BuscoID  Missing

    Returns (busco2contigdict, contigs2buscodict) — same format as import_orthologs().
    """
    buscoids = set()
    contignames = set()
    rows = []

    with open(filepath) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            buscoid = parts[0]
            status = parts[1]
            buscoids.add(buscoid)
            if status == 'Missing' or len(parts) < 3:
                rows.append((buscoid, 'M', None))
            else:
                contig = parts[2]
                contignames.add(contig)
                status_char = status[0]  # C, F, M
                rows.append((buscoid, status_char, contig))

    busco2contigdict = {}
    for buscoid in buscoids:
        busco2contigdict[buscoid] = {t: [] for t in BUSCO_TYPES}

    contigs2buscodict = {}
    for contig in contignames:
        contigs2buscodict[contig] = {t: [] for t in BUSCO_TYPES}

    for buscoid, status_char, contig in rows:
        if status_char != 'M' and contig is not None:
            busco2contigdict[buscoid][status_char].append(contig)
            contigs2buscodict[contig][status_char].append(buscoid)

    return busco2contigdict, contigs2buscodict


def load_orthologs(path):
    """Load ortholog classification from either a directory or a single TSV file.

    If path is a file, reads it as a consolidated TSV (full_table_results.tsv format).
    If path is a directory, scans for per-contig full_table_*.tsv files.
    """
    if os.path.isfile(path):
        return import_orthologs_tsv(path)
    return import_orthologs(path)


def import_orthologs(buscofileloc):
    """Load ortholog classification from full_table_*.tsv files.

    Returns (busco2contigdict, contigs2buscodict).
    """
    contignames = set()
    buscoids = set()
    mybuscofiles = glob.glob(buscofileloc + '/odbaln_*/*/full_table_*.tsv')
    if len(mybuscofiles) == 0:
        mybuscofiles = glob.glob(buscofileloc + '/busco*/*/full_table_*.tsv')
    if len(mybuscofiles) == 0:
        mybuscofiles = glob.glob(buscofileloc + '/full_table_*.tsv')
    if len(mybuscofiles) == 0:
        raise FileNotFoundError(
            'No ortholog result files found in: ' + buscofileloc + '\n'
            'Searched patterns:\n'
            '  ' + buscofileloc + '/odbaln_*/*/full_table_*.tsv\n'
            '  ' + buscofileloc + '/busco*/*/full_table_*.tsv\n'
            '  ' + buscofileloc + '/full_table_*.tsv\n'
            'Please verify the output directory path and that ortholog search '
            'has completed successfully.'
        )
    with open(mybuscofiles[0]) as fin:
        for line in fin:
            if line[0] != '#':
                buscoids.add(line.strip().split()[0])
    for i in range(0, len(mybuscofiles)):
        mylinecounter = 0
        with open(mybuscofiles[i]) as fin:
            for line in fin:
                mylinecounter += 1
                if line[0] == '#' and mylinecounter < 4:
                    if mylinecounter == 3:
                        contignames.add(line.split()[8].split('/')[-1].replace('.fasta', ''))
                elif mylinecounter > 3:
                    break
    if len(contignames) != len(set(contignames)):
        raise ValueError(
            'Duplicate contig names exist. Please fix contig names so that '
            'no duplicates exist and rerun HapSolo.'
        )

    busco2contigdict = dict()
    contigs2buscodict = dict()

    for buscoid in buscoids:
        busco2contigdict[buscoid] = dict()
        for buscotype in BUSCO_TYPES:
            busco2contigdict[buscoid][buscotype] = list()
    for contigname in contignames:
        contigs2buscodict[contigname] = dict()
        for buscotype in BUSCO_TYPES:
            contigs2buscodict[contigname][buscotype] = list()

    for file in mybuscofiles:
        mylines = list()
        contigname = None
        mylinecounter = 0
        with open(file) as fin:
            for line in fin:
                if line[0] != '#':
                    mylines.append(line.strip().split())
                elif line[0] == '#' and mylinecounter < 4:
                    mylinecounter += 1
                    if mylinecounter == 3:
                        contigname = line.split()[8].split('/')[-1].replace('.fasta', '')
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


def calculate_scores(contig_list, busco2contig, contigs2busco,
                     missing_ref_contigs, small_contigs):
    """Score ortholog completeness for a contig set.

    Preserves exact legacy calculateBuscos behavior.
    """
    duplicatebuscos = 0
    singlebuscos = 0
    fragmentedbuscos = 0
    buscotypecounts = dict()
    buscoids = busco2contig.keys()
    completebuscoidcounts = dict()
    fragmentedbuscoidcounts = dict()
    for buscoid in buscoids:
        completebuscoidcounts[buscoid] = 0
        fragmentedbuscoidcounts[buscoid] = 0
    for buscotype in BUSCO_TYPES:
        buscotypecounts[buscotype] = 0
    mycontigset = set(contig_list).union(missing_ref_contigs) - small_contigs
    for contig in mycontigset:
        if contig in contigs2busco.keys():
            for buscotype in contigs2busco[contig]:
                buscosize = len(contigs2busco[contig][buscotype])
                if buscotype == 'C' and buscosize > 0:
                    for buscoid in contigs2busco[contig][buscotype]:
                        completebuscoidcounts[buscoid] += 1
    for contig in mycontigset:
        if contig in contigs2busco.keys():
            for buscotype in contigs2busco[contig]:
                buscosize = len(contigs2busco[contig][buscotype])
                if buscosize > 0 and buscotype == 'F':
                    for buscoid in contigs2busco[contig][buscotype]:
                        if completebuscoidcounts[buscoid] == 0 and fragmentedbuscoidcounts[buscoid] == 0:
                            fragmentedbuscos += 1
                            fragmentedbuscoidcounts[buscoid] = 1
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


def cost_function(m, s, d, f, total, theta_s=1.0, theta_d=1.0, theta_f=0.0, theta_m=1.0,
                  formula_fn=None):
    """Cost function for ortholog completeness scoring. Lower is better.

    Default formula: (thetaF*F + thetaD*D + thetaM*M) / (thetaS*S).
    When formula_fn is provided, it is called instead.
    """
    if formula_fn is not None:
        return float(formula_fn(
            S=s, D=d, F=f, M=m, C=s + d, n=total,
            theta_s=theta_s, theta_d=theta_d, theta_f=theta_f, theta_m=theta_m))
    return float(theta_f * f + theta_d * d + theta_m * m) / float(theta_s * s)
