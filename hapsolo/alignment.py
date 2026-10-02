import contextlib
import os
import tempfile
import gzip

import pandas as pd

from hapsolo.utils import CalculatePctAlign, CalculateInverseProportion

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def reduce_asm(df, pid, qpct, qrpct, all_contigs):
    """Filter alignment DataFrame by thresholds, return good contig set.

    Contigs appearing in alignments that pass ALL thresholds are removed
    from the assembly (they are haplotigs). Returns the set of contigs
    to keep.
    """
    qrpct_max = CalculateInverseProportion(qrpct)
    t0 = df[df['PID'] >= pid]
    t1 = t0[t0['QPct'] >= qpct]
    t0 = t1[t1['QRAlignLenPct'] >= qrpct]
    t1 = t0[t0['QRAlignLenPct'] <= qrpct_max]
    qnames = t1['qName']
    try:
        removed = set(qnames.to_pandas())
    except AttributeError:
        removed = set(qnames)
    return all_contigs - removed


def _print_purge_breakdown(fcounter, mcounter, purge_self, purge_size,
                           purge_pid, purge_qpct, purge_qrpct,
                           min_contig, min_pid, min_qpct, min_qrpct):
    """Print a breakdown of how many alignments were purged by each filter."""
    purged = fcounter - mcounter
    print(str(purged) + ' alignments Purged due to Search Space constraints')
    if purged > 0:
        print('  Breakdown (alignments may fail multiple filters; counted by first failure):')
        print('    Self-alignments (qName == tName):    ' + str(purge_self))
        print('    Query length < ' + str(min_contig) + ' bp (--min):     ' + str(purge_size))
        print('    PID < ' + str(min_pid) + ' (-P/--minPID):              ' + str(purge_pid))
        print('    QPct < ' + str(min_qpct) + ' (-Q/--minQ):              ' + str(purge_qpct))
        print('    QRAlignLenPct < ' + str(min_qrpct) + ' (-R/--minQR):     ' + str(purge_qrpct))
    print('  Alignments retained:                   ' + str(mcounter)
          + ' / ' + str(fcounter))


def _open_with_progress(filename, is_gz, desc='Reading'):
    """Open a file (or .gz) and return (file_handle, tqdm_bar, pos_callable).

    The returned file_handle MUST be read with readline() in a while loop,
    NOT with `for line in fin:`.
    """
    total = os.path.getsize(filename)
    bar = None
    if HAS_TQDM:
        suffix = ' (gz)' if is_gz else ''
        bar = tqdm(total=total, unit='B', unit_scale=True, unit_divisor=1024,
                   desc=desc + suffix, dynamic_ncols=True)
    if is_gz:
        raw = open(filename, 'rb')
        gz = gzip.GzipFile(fileobj=raw, mode='rb')
        import io
        text = io.TextIOWrapper(gz)
        pos_fn = raw.tell
        return text, bar, pos_fn
    else:
        f = open(filename, 'rb')
        import io
        text = io.TextIOWrapper(f)
        pos_fn = f.tell
        return text, bar, pos_fn


def load_hap_file(hapfile):
    """Load a pre-existing .hap file directly into a DataFrame."""
    print('Loading pre-existing HAP file: ' + hapfile, flush=True)
    rows = []
    with open(hapfile, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 6:
                continue
            qname = parts[0].strip('"')
            tname = parts[1].strip('"')
            qsize = int(parts[2])
            qpct = float(parts[3])
            pid = float(parts[4])
            qrpct = float(parts[5])
            rows.append((qname, tname, qsize, qpct, pid, qrpct))
    print('Loaded ' + str(len(rows)) + ' alignments from HAP file.', flush=True)
    df = pd.DataFrame(rows, columns=['qName', 'tName', 'qSize', 'QPct', 'PID', 'QRAlignLenPct'])
    df['qName'] = df['qName'].astype(object)
    df['tName'] = df['tName'].astype(object)
    return df


@contextlib.contextmanager
def _atomic_hap_writer(path):
    """Write a .hap cache via a temp file that is renamed into place only on success.

    The cache is reused by every later run whenever it exists and is non-empty, so a partial
    file left by an interrupted run (as happened to the thorny skate cache on 2026-07-12:
    1,502 of 207,119 alignments) silently corrupts all later training.
    """
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + '.', suffix='.tmp',
                               dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, 'w') as fout:
            yield fout
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def create_paf_alignment(alignmentfile, min_contig, min_pid, min_qpct, min_qrpct):
    """Process PAF alignment file into a filtered DataFrame.

    Creates a .hap intermediate file, loads it into pandas.
    If the .hap file already exists, loads it directly (skips PAF parsing).
    Returns the alignment DataFrame.
    """
    fileext = alignmentfile.split('.')[-1]
    is_gz = (fileext == 'gz')
    if is_gz:
        newalignfile = alignmentfile.replace('.paf.gz', '.hap')
    else:
        newalignfile = alignmentfile.replace('.paf', '.hap')

    if os.path.exists(newalignfile) and os.path.getsize(newalignfile) > 0:
        return load_hap_file(newalignfile)

    fcounter = 0
    mcounter = 0
    purge_self = purge_size = purge_pid = purge_qpct = purge_qrpct = 0
    rows = []

    fin, bar, pos_fn = _open_with_progress(alignmentfile, is_gz, 'Reading PAF')
    last_pos = 0
    try:
        with _atomic_hap_writer(newalignfile) as fout:
            while True:
                line = fin.readline()
                if not line:
                    break
                if bar is not None and fcounter % 1000 == 0:
                    cur = pos_fn()
                    bar.update(cur - last_pos)
                    last_pos = cur
                line = line.strip().split()
                if len(line) < 11:
                    raise ValueError(
                        'Error in reading PAF file: line has fewer than 11 fields'
                    )
                fcounter += 1
                fqAlignLen = max(int(line[2]), int(line[3])) - min(int(line[2]), int(line[3]))
                frAlignLen = max(int(line[7]), int(line[8])) - min(int(line[7]), int(line[8]))
                fQRAlignLenPct = CalculatePctAlign(fqAlignLen, frAlignLen)
                fQPct = CalculatePctAlign(fqAlignLen, int(line[1]))
                fPID = CalculatePctAlign(int(line[9]), fqAlignLen)
                if line[0] == line[5]:
                    purge_self += 1
                elif int(line[1]) < min_contig:
                    purge_size += 1
                elif fPID < min_pid:
                    purge_pid += 1
                elif fQPct < min_qpct:
                    purge_qpct += 1
                elif fQRAlignLenPct < min_qrpct:
                    purge_qrpct += 1
                else:
                    mcounter += 1
                    rows.append((line[0], line[5], int(line[1]), fQPct, fPID, fQRAlignLenPct))
                    fout.write('"' + line[0] + '"' + '\t' + '"' + line[5] + '"' + '\t'
                               + line[1] + '\t' + str(fQPct) + '\t' + str(fPID) + '\t'
                               + str(fQRAlignLenPct) + '\n')
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
                           purge_pid, purge_qpct, purge_qrpct,
                           min_contig, min_pid, min_qpct, min_qrpct)
    if mcounter == 0:
        raise ValueError(
            'No alignments passed filtering. The resulting HAP file is empty. '
            'Please check your alignment file and filter thresholds.'
        )
    print('Building DataFrame from ' + str(mcounter) + ' alignments...', flush=True)
    df = pd.DataFrame(rows, columns=['qName', 'tName', 'qSize', 'QPct', 'PID', 'QRAlignLenPct'])
    df['qName'] = df['qName'].astype(object)
    df['tName'] = df['tName'].astype(object)
    return df


def create_psl_alignment(alignmentfile, min_contig, min_pid, min_qpct, min_qrpct):
    """Process PSL alignment file into a filtered DataFrame.

    Creates a .hap intermediate file, loads it into pandas.
    If the .hap file already exists, loads it directly (skips PSL parsing).
    Returns the alignment DataFrame.
    """
    fileext = alignmentfile.split('.')[-1]
    is_gz = (fileext == 'gz')
    if is_gz:
        newalignfile = alignmentfile.replace('.psl.gz', '.hap')
    else:
        newalignfile = alignmentfile.replace('.psl', '.hap')

    if os.path.exists(newalignfile) and os.path.getsize(newalignfile) > 0:
        return load_hap_file(newalignfile)

    fcounter = 0
    mcounter = 0
    mylinenum = 0
    purge_self = purge_size = purge_pid = purge_qpct = purge_qrpct = 0
    rows = []

    fin, bar, pos_fn = _open_with_progress(alignmentfile, is_gz, 'Reading PSL')
    last_pos = 0
    try:
        with _atomic_hap_writer(newalignfile) as fout:
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
                    raise ValueError(
                        'Error in reading PSL file: line ' + str(mylinenum)
                        + ' has fewer than 21 fields'
                    )
                elif mylinenum > 5:
                    fcounter += 1
                    fqAlignLen = max(int(line[11]), int(line[12])) - min(int(line[11]), int(line[12]))
                    frAlignLen = max(int(line[15]), int(line[16])) - min(int(line[15]), int(line[16]))
                    fQRAlignLenPct = CalculatePctAlign(fqAlignLen, frAlignLen)
                    fQPct = CalculatePctAlign(fqAlignLen, int(line[10]))
                    fPID = CalculatePctAlign(int(line[0]), fqAlignLen)
                    if line[9] == line[13]:
                        purge_self += 1
                    elif int(line[10]) < min_contig:
                        purge_size += 1
                    elif fPID < min_pid:
                        purge_pid += 1
                    elif fQPct < min_qpct:
                        purge_qpct += 1
                    elif fQRAlignLenPct < min_qrpct:
                        purge_qrpct += 1
                    else:
                        mcounter += 1
                        rows.append((line[9], line[13], int(line[10]), fQPct, fPID, fQRAlignLenPct))
                        fout.write('"' + line[9] + '"' + '\t' + '"' + line[13] + '"' + '\t'
                                   + line[10] + '\t' + str(fQPct) + '\t' + str(fPID) + '\t'
                                   + str(fQRAlignLenPct) + '\n')
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
                           purge_pid, purge_qpct, purge_qrpct,
                           min_contig, min_pid, min_qpct, min_qrpct)
    if mcounter == 0:
        raise ValueError(
            'No alignments passed filtering. The resulting HAP file is empty. '
            'Please check your alignment file and filter thresholds.'
        )
    print('Building DataFrame from ' + str(mcounter) + ' alignments...', flush=True)
    df = pd.DataFrame(rows, columns=['qName', 'tName', 'qSize', 'QPct', 'PID', 'QRAlignLenPct'])
    df['qName'] = df['qName'].astype(object)
    df['tName'] = df['tName'].astype(object)
    return df
