import gzip
from math import exp, log


def open_gzip(filename):
    """Open a gzip file for text reading."""
    return gzip.open(filename, 'rt')


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
