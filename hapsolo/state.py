import pandas as pd


class HapSoloState:
    """Central state object replacing module-level globals.

    Passed explicitly to functions instead of relying on global variables.
    Holds pandas DataFrames; GPU path extracts cupy arrays in optimizers._ensure_gpu().
    """

    def __init__(self):
        self.alignment_df = pd.DataFrame()
        self.contig_sizes = {}
        self.scoring_data = None
        self.all_contigs = set()
        self.query_contigs = set()
        self.missing_ref_contigs = set()
        self.small_contigs = set()
        self.error_log = ''
        # Legacy dicts — used until ScoringData migration is complete
        self.busco2contigdict = {}
        self.contigs2buscodict = {}
