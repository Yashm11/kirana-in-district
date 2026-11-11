"""Tests for pack-size parsing, brand cleaning and the price-per-100 calculations."""

import pytest

from src.utils.parsing import (
    parse_quantity,
    parse_size_from_name,
    price_per_100,
    price_per_piece,
    standardise_brand,
)


@pytest.mark.parametrize(
    ("text", "basis", "total"),
    [
        ("500 g", "g", 500),
        ("2 kg", "g", 2000),
        ("1.5 kg", "g", 1500),
        ("1 L", "ml", 1000),
        ("13 Litre", "ml", 13000),
        ("500 ml", "ml", 500),
        ("100ml", "ml", 100),  # no space
        ("5.5gm", "g", 5.5),  # "gm" spelling
        ("60g", "g", 60),
        ("2x200 g", "g", 400),  # multipack without spaces
        ("2 x 100 g", "g", 200),  # multipack with spaces
        ("6x1 L", "ml", 6000),
        ("2x1 kg", "g", 2000),
        ("3x(3x125 g)", "g", 1125),  # nested multipack
        ("12 pcs", "pc", 12),
        ("1 pc", "pc", 1),
        ("5x6 pcs", "pc", 30),
        ("10 Tablets", "pc", 10),
        ("set of 4", "pc", 4),
        ("(Pack Of 50)", "pc", 50),
        ("3x10's pack", "pc", 30),
        ("25 pcs + 12 pcs", "pc", 37),
        ("5L (4.58 kg)", "ml", 5000),  # bracket note ignored
        ("500 ml each", "ml", 500),
    ],
)
def test_parse_quantity_ok(text: str, basis: str, total: float) -> None:
    result = parse_quantity(text)
    assert result.status == "ok"
    assert result.basis == basis
    assert result.total == pytest.approx(total)


def test_multipack_keeps_unit_size_and_count() -> None:
    result = parse_quantity("2 x 100 g")
    assert (result.unit_size, result.count) == (100, 2)


def test_each_is_noted() -> None:
    assert parse_quantity("500 ml each").note == "each"


def test_combo_is_flagged_not_parsed() -> None:
    for text in ("Combo", "bb Combo", "combo"):
        result = parse_quantity(text)
        assert result.status == "combo"
        assert result.total is None


@pytest.mark.parametrize(
    "text", ["13 x 13 cm", "350 mg", "1 kit", "4 Products", "1.3 g & 12 ml", ""]
)
def test_unparseable_is_reported(text: str) -> None:
    result = parse_quantity(text)
    assert result.status == "unparsed"
    assert result.note  # always says why


def test_size_from_name_needs_exactly_one_size() -> None:
    assert parse_size_from_name("Aloo Bhujia 150 g").total == 150
    assert parse_size_from_name("Soap 100 g + Shampoo 200 ml").status == "unparsed"
    assert parse_size_from_name("Plain Mug").status == "unparsed"


def test_price_per_100_g_and_ml() -> None:
    assert price_per_100(50, 250, "g") == pytest.approx(20)  # Rs 50 for 250 g
    assert price_per_100(30, 1000, "ml") == pytest.approx(3)  # Rs 30 for 1 L
    assert price_per_100(120, 12, "pc") is None  # counts have no per-100g
    assert price_per_100(100, None, "g") is None
    assert price_per_100(100, 0, "g") is None


def test_price_per_piece() -> None:
    assert price_per_piece(120, 12, "pc") == pytest.approx(10)  # Rs 120 for 12 pcs
    assert price_per_piece(120, 500, "g") is None


def test_standardise_brand() -> None:
    assert standardise_brand("  BIOTIQUE ") == "Biotique"
    assert standardise_brand("MTR") == "MTR"  # short: probably an acronym
    assert standardise_brand("H&B") == "H&B"
    assert standardise_brand("kwality walls") == "Kwality Walls"
    assert standardise_brand("BB  Royal") == "BB Royal"
    assert standardise_brand("DR.OETKER", {"dr.oetker": "Dr. Oetker"}) == "Dr. Oetker"
