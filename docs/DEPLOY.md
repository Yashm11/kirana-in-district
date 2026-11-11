# Deploying K.I.D to Streamlit Community Cloud

The hosted app is `src/app/kid.py`. It reads only files that are committed to the repo:

| File | What it is |
|---|---|
| `app_data/kid.duckdb` | slim warehouse: district and brand aggregates, no scraped product listings |
| `data/processed/map_districts.geojson` | district polygons (LGD, simplified) |
| `src/app/requirements.txt` | the six packages the app needs (no geopandas, no osmium) |

It uses the **rules** backend only. A local model cannot be hosted for free, so the
"Local model" switch appears only when you run the app on a machine with Ollama.

## Steps (you do these in the browser)
1. Push `main` to GitHub (private repo `Yashm11/kirana-in-district` is fine).
2. Go to https://share.streamlit.io and sign in with the personal GitHub account.
3. When asked, let Streamlit read **private** repositories, and pick the personal account only.
4. New app, then:
   - Repository: `Yashm11/kirana-in-district`, branch `main`
   - Main file path: `src/app/kid.py`
   - Advanced settings: Python 3.12
5. Deploy. The first build takes a few minutes. Choose a URL such as `kid-india`.
6. In the app settings, set sharing to "Only specific people" while testing, public when you post.

## Refreshing the data
`make a7` rebuilds both `data/warehouse/kid.duckdb` (full, ignored by git) and
`app_data/kid.duckdb` (slim, committed). Commit the slim file and Streamlit redeploys.

## Before making anything public
- HCES: the data rider forbids redistributing unit-level microdata. Only derived aggregates are
  published here (state, district and brand tables). Nothing from `data/raw/` is committed.
- `data/processed/products_clean.parquet` (BigBasket listings with names and URLs) is committed
  in this private repo. If the repo is ever made public, remove it from git first. The hosted app
  does not read it.
- The LGD boundary file has no declared licence and is not committed; only the simplified
  polygons in `map_districts.geojson` are.
- Check India's border depiction against Survey of India guidance (assumption 49).
