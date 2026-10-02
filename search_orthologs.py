#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backward-compat stub — delegates to hapsolo.search.

Re-exports all public functions so existing imports keep working.
"""
from hapsolo.search import (
    find_protein_file,
    _open_protein_file,
    load_scores_cutoff,
    load_lengths_cutoff,
    build_protein_to_busco_map,
    run_miniprot,
    parse_miniprot_paf,
    classify_buscos,
    write_odb_output,
    detect_lineage_name,
    parse_classify_params,
    CLASSIFY_PARAM_KEYS,
    CLASSIFY_PARAM_DEFAULTS,
    main,
)

if __name__ == '__main__':
    main()
