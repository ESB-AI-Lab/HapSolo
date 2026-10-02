import re


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

    sanitized_lookup = dict()
    for name in canonical_names:
        san = sanitize_name(name)
        if san in sanitized_lookup:
            sanitized_lookup[san] = None
        else:
            sanitized_lookup[san] = name

    for ext_name in external_names:
        if ext_name in canonical_set:
            continue

        san_ext = sanitize_name(ext_name)

        if san_ext in sanitized_lookup and sanitized_lookup[san_ext] is not None:
            conversion[ext_name] = sanitized_lookup[san_ext]
            continue

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
