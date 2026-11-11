"""Helpers for cleaning text, standardising brands, parsing pack sizes and unit prices.

Pack sizes are converted to base units:
    weight -> grams ("g"), volume -> millilitres ("ml"), counts -> pieces ("pc").
"""

import re
from dataclasses import dataclass

NUM = r"\d+(?:\.\d+)?"

# unit text -> (base unit, multiplier to base unit). Count words (tablets, pulls...) are pieces.
UNITS: dict[str, tuple[str, float]] = {
    "kg": ("g", 1000),
    "gms": ("g", 1),
    "gm": ("g", 1),
    "g": ("g", 1),
    "ltr": ("ml", 1000),
    "litre": ("ml", 1000),
    "liter": ("ml", 1000),
    "lt": ("ml", 1000),
    "l": ("ml", 1000),
    "ml": ("ml", 1),
    "pcs": ("pc", 1),
    "pc": ("pc", 1),
    "pieces": ("pc", 1),
    "tablets": ("pc", 1),
    "capsules": ("pc", 1),
    "softgels": ("pc", 1),
    "pulls": ("pc", 1),
    "wipes": ("pc", 1),
    "sachets": ("pc", 1),
    "sachet": ("pc", 1),
    "teabags": ("pc", 1),
}
# longest first so "kg" is tried before "g" and "pcs" before "pc"
UNIT_RE = "|".join(sorted(UNITS, key=len, reverse=True))

# "2 x 100 g", "500ml", "1 L each", "120 g 12 pcs" (anything after the unit is checked later)
_MAIN = re.compile(rf"(?:({NUM})\s*x\s*)?({NUM})\s*({UNIT_RE})(?![a-z])")
# "3x(3x125 g)"
_NESTED = re.compile(rf"({NUM})\s*x\s*\(\s*({NUM})\s*x\s*({NUM})\s*({UNIT_RE})\s*\)")
# "10's pack", "3x10's pack"
_APOSTROPHE_PACK = re.compile(rf"(?:({NUM})\s*x\s*)?({NUM})'s\s*pack")
# "set of 4", "pack of 2", "(Pack Of 50)"
_SET_OF = re.compile(rf"(?:set|pack) of ({NUM})")
# a weight/volume written inside a product name, e.g. "Aloo Bhujia 150 g"
_NAME_SIZE = re.compile(rf"({NUM})\s*(kg|gms|gm|g|ml|ltr|litre|l)(?![a-z])")


@dataclass(frozen=True)
class PackSize:
    """Result of parsing one Quantity string.

    basis:     "g", "ml", "pc", or None when it could not be parsed
    total:     total amount in base units across the whole pack (e.g. 2 x 100 g -> 200)
    unit_size: size of one unit inside a multipack (100 in the example above)
    count:     number of units in the pack (2 in the example above)
    status:    "ok", "combo" or "unparsed"
    note:      extra detail: "each", "trailing_text", "from_name", or the failure reason
    """

    basis: str | None
    total: float | None
    unit_size: float | None
    count: float | None
    status: str
    note: str = ""


def unparsed(reason: str) -> PackSize:
    """A PackSize for a quantity we could not understand."""
    return PackSize(None, None, None, None, "unparsed", reason)


def clean_text(value: object) -> str:
    """Trim and collapse repeated spaces. Missing values become an empty string."""
    if value is None or value != value:  # NaN check without importing pandas
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def standardise_brand(raw: str, aliases: dict[str, str] | None = None) -> str:
    """Return a tidy brand name.

    1. Trim and collapse spaces.
    2. If the brand is in the alias map (case-insensitive), use the mapped name.
    3. ALL-CAPS words of 5+ letters (e.g. "BIOTIQUE") become "Biotique". Short names
       (DP, MTR, VLCC) and names with symbols (H&B, SE7EN) are left alone because
       they are probably acronyms.
    4. Fully lowercase brands get a capital letter on each word ("kwality walls").
    """
    brand = clean_text(raw)
    if aliases and brand.casefold() in aliases:
        return aliases[brand.casefold()]
    if brand.isupper() and re.fullmatch(r"[A-Z ]{5,}", brand):
        return brand.title()
    if brand.islower():
        return " ".join(w[:1].upper() + w[1:] for w in brand.split(" "))
    return brand


def _parse_single(part: str) -> PackSize | None:
    """Parse one piece of a quantity (no '+' in it). Returns None if not understood."""
    nested = _NESTED.fullmatch(part)
    if nested:
        outer, inner, size, unit = nested.groups()
        basis, factor = UNITS[unit]
        count = float(outer) * float(inner)
        return PackSize(basis, count * float(size) * factor, float(size) * factor, count, "ok")

    apos = _APOSTROPHE_PACK.fullmatch(part)
    if apos:
        mult, size = apos.groups()
        count = float(mult or 1)
        return PackSize("pc", count * float(size), float(size), count, "ok")

    set_of = _SET_OF.search(part)
    if set_of and part.startswith(("set", "pack")):
        n = float(set_of.group(1))
        return PackSize("pc", n, 1.0, n, "ok")

    m = _MAIN.match(part)
    if not m:
        return None
    mult, size, unit = m.groups()
    rest = part[m.end() :].strip(" .")
    if "&" in rest:  # e.g. "1.3 g & 12 ml": two different quantities, do not guess
        return None
    basis, factor = UNITS[unit]
    count = float(mult or 1)
    unit_size = float(size) * factor
    note = ""
    if rest == "each":
        note = "each"
    elif rest:
        note = "trailing_text"
    return PackSize(basis, count * unit_size, unit_size, count, "ok", note)


def parse_quantity(text: object) -> PackSize:
    """Parse a BigBasket Quantity string such as "500 g", "2x200 ml" or "12 pcs".

    Combos ("Combo") are flagged, not parsed. Anything else we cannot read is returned
    with status "unparsed" and a reason, so nothing is silently lost.
    """
    t = clean_text(text).lower()
    if not t:
        return unparsed("empty")
    if "combo" in t:
        return PackSize(None, None, None, None, "combo", "combo")

    nested = _NESTED.fullmatch(t)
    if nested:
        return _parse_single(t)  # type: ignore[return-value]

    # Drop notes in brackets such as "(approx. 1 -3 kg)", but keep a whole "(Pack Of 50)"
    if t.startswith("(") and t.endswith(")"):
        t = t[1:-1]
    else:
        t = re.sub(r"\s*\([^)]*\)", "", t).strip()

    parts = [p.strip() for p in t.split("+")]
    results = [_parse_single(p) for p in parts]
    if any(r is None for r in results):
        return unparsed("unrecognised format")
    if len({r.basis for r in results}) > 1:
        return unparsed("mixed units")
    if len(results) == 1:
        return results[0]  # type: ignore[return-value]
    # "25 pcs + 12 pcs": add the pieces together
    total = sum(r.total for r in results)  # type: ignore[misc]
    return PackSize(results[0].basis, total, None, None, "ok", "sum_of_parts")  # type: ignore[union-attr]


def parse_size_from_name(name: object) -> PackSize:
    """Fallback: read a pack size out of the product name when Quantity failed.

    Only used when exactly one distinct size appears in the name, to avoid guessing.
    """
    found = {(float(v), u) for v, u in _NAME_SIZE.findall(clean_text(name).lower())}
    if len(found) != 1:
        return unparsed("no single size in name")
    value, unit = next(iter(found))
    basis, factor = UNITS[unit]
    size = value * factor
    return PackSize(basis, size, size, 1.0, "ok", "from_name")


def price_per_100(price: float, total: float | None, basis: str | None) -> float | None:
    """Price per 100 g or 100 ml. Returns None for counts, combos and missing sizes."""
    if basis not in ("g", "ml") or not total or total <= 0:
        return None
    return price / total * 100


def price_per_piece(price: float, total: float | None, basis: str | None) -> float | None:
    """Price per piece for count-based packs (e.g. Rs 120 for 12 pcs -> Rs 10)."""
    if basis != "pc" or not total or total <= 0:
        return None
    return price / total
