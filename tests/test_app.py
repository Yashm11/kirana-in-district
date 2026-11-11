"""Smoke test: every Streamlit section renders without an exception."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "src" / "app" / "app.py")
SECTIONS = [
    "Overview",
    "Price-pack (A2)",
    "Districts and clusters (A3, A4)",
    "Recommendations (A5)",
    "Maps (A6)",
    "Assumptions",
]


@pytest.mark.parametrize("section", SECTIONS)
def test_section_renders(section: str) -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value(section).run()
    assert not at.exception
