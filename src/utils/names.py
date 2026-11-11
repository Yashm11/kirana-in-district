"""District-name matching helpers (used by A3).

District names differ between NFHS-5, Census 2011 and others ("Bengaluru Urban" vs
"Bangalore Urban"). Workflow:
    1. normalise names (case, punctuation, "&" -> "and", a trailing "district")
    2. use hand-written overrides from config/district_name_map.csv first
    3. exact match on the normalised name (then the same with spaces removed)
    4. everything left is printed as UNMATCHED and written, with fuzzy suggestions, to a
       template CSV. You review it and copy confirmed rows into config/district_name_map.csv.
Fuzzy matches are suggestions only; they are never applied automatically.
"""

import difflib
import re

import pandas as pd


def normalise_name(name: object) -> str:
    """Lower-case, drop punctuation and a trailing 'district', collapse spaces."""
    text = "" if name is None or name != name else str(name).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+district$", "", text)


def match_names(
    source: list[str], reference: list[str], overrides: dict[str, str] | None = None
) -> tuple[dict[str, str], list[str]]:
    """Match source names to reference names.

    overrides: {normalised source name: exact reference name} from the hand-made map.
    Returns ({source name: reference name}, [unmatched source names]).
    """
    overrides = overrides or {}
    ref_by_key = {normalise_name(r): r for r in reference}
    # same names with all spaces removed, to survive splits like "Farrukhab ad"
    ref_by_compact = {k.replace(" ", ""): r for k, r in ref_by_key.items()}
    matched: dict[str, str] = {}
    unmatched: list[str] = []
    for name in source:
        key = normalise_name(name)
        if key in overrides:
            matched[name] = overrides[key]
        elif key in ref_by_key:
            matched[name] = ref_by_key[key]
        elif key.replace(" ", "") in ref_by_compact:
            matched[name] = ref_by_compact[key.replace(" ", "")]
        else:
            unmatched.append(name)
    return matched, unmatched


def suggest_matches(
    unmatched: list[str], reference: list[str], cutoff: float = 0.7
) -> pd.DataFrame:
    """Fuzzy suggestions for unmatched names, as a template for the hand-made map.

    Columns: source_name, suggested_match, score, standard_name (empty: you fill this in).
    """
    ref_by_key = {normalise_name(r): r for r in reference}
    rows = []
    for name in unmatched:
        best = difflib.get_close_matches(normalise_name(name), list(ref_by_key), n=1, cutoff=cutoff)
        suggestion = ref_by_key[best[0]] if best else ""
        score = (
            difflib.SequenceMatcher(None, normalise_name(name), best[0]).ratio() if best else 0.0
        )
        rows.append(
            {
                "source_name": name,
                "suggested_match": suggestion,
                "score": round(score, 2),
                "standard_name": "",
            }
        )
    return pd.DataFrame(rows, columns=["source_name", "suggested_match", "score", "standard_name"])
