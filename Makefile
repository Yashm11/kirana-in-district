.PHONY: setup a0 a1 a2 a3 a3b a3c a0-osm a4 a5 a5-district a6 app check test lint all

setup:
	uv sync

# A0: download BigBasket, then report which hand-downloaded files are present
a0:
	uv run python -m src.data.download_bigbasket
	uv run python -m src.data.download_boundaries
	uv run python -m src.data.check_manual_files

a1:
	uv run python -m src.analytics.a1_clean_products
	uv run python -m src.analytics.check_products_clean

# Open the results viewer in the browser (run `make all` first)
app:
	uv run streamlit run src/app/app.py

check:
	uv run python -m src.analytics.check_products_clean

a2:
	uv run python -m src.analytics.a2_price_pack

a3:
	uv run python -m src.data.extract_hces_lookups
	uv run python -m src.analytics.a3_district_features

# HCES quantities: unit prices and monthly quantity per household (state x rural/urban)
a3b:
	uv run python -m src.analytics.a3b_hces_unit_values

# OSM shops per district (needs `make a0-osm`, a 1.7 GB download, and `make a6` for populations)
a0-osm:
	uv run python -m src.data.download_osm

a3c:
	uv run python -m src.analytics.a3c_kirana_density

a4:
	uv run python -m src.analytics.a4_cluster_districts

a5:
	uv run python -m src.analytics.a5_whitespace

# EXPLORATORY: district-level spend (the HCES data note advises against it); separate output files
a5-district:
	uv run python -m src.analytics.a5_whitespace --spend-level district

# Interactive state and district maps (needs the boundary download in a0)
a6:
	uv run python -m src.analytics.a6_maps

test:
	uv run pytest

lint:
	uv run ruff check .

all: a0 a1 a2 a3 a3b a4 a5 a6
