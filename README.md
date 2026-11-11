# kirana-in-district (K.I.D)

Open-data analysis of pack-price gaps and district whitespace in Indian FMCG, with a kirana shelf-share detector (planned).

> Independent portfolio project. Not affiliated with any brand.

## Question

Which pack size and price point should a packaged-goods brand push in which type of Indian district? Only public data and proxies are used. There is no company sales data.

## Headline result

_TODO: filled in after A5 runs._

## Data sources

| Source | Used for | Access |
|---|---|---|
| BigBasket products (Kaggle) | Pack sizes, prices, discounts | Automatic: `make a0` |
| NFHS-5 district indicators (IIPS) | District development profile | Manual download into `data/raw/nfhs5/` |
| HCES 2022-23 (MoSPI) | Household weights, population, spend by state x rural/urban | Manual download (free login) into `data/raw/hces_2022_23/` |
| Census 2011 | Cross-check of the population estimate | Manual download into `data/raw/census_2011/` |
| DataMeet district boundaries | Interactive maps | Automatic: `make a0` (credit DataMeet, CC BY 2.5 India) |
| Open Food Facts (optional) | Not used yet | `uv run python -m src.data.download_off --include-off` |

Licences and caveats for each source are in [data/README.md](data/README.md). Raw data is never committed.

## How to run

```bash
uv sync          # install dependencies from uv.lock
make a0          # download BigBasket, list which hand-downloaded files are present
make all         # run every stage (a0 ... a5)
make a1          # or run one stage (a0 ... a5)
make app         # open the results viewer (Streamlit) in your browser
make test        # pytest
make lint        # ruff
```

Each stage also runs alone, e.g. `uv run python -m src.analytics.a1_clean_products`.
Put the NFHS-5, HCES and Census files in `data/raw/<source>/` first (see
`uv run python -m src.data.check_manual_files`). A3 to A5 print "SKIPPED" if they are missing.
Kaggle needs a token in `KAGGLE_API_TOKEN` or `~/.kaggle/access_token`.

## Pipeline

| Stage | What it does |
|---|---|
| A0 | Download data |
| A1 | Clean products: parse pack sizes, price per 100 g / 100 ml, price per piece |
| A2 | Price-pack analysis: ladders by brand x subcategory, pack premium, entry prices, ladder gaps |
| A3 | District features: NFHS-5 indicators + HCES population and state-level spend |
| A4 | Cluster districts: StandardScaler, PCA, KMeans, silhouette, sensitivity check |
| A5 | Whitespace score (relative) and recommendations table |
| A6 | Interactive state and district maps (`reports/maps/*.html`, and the Maps page of `make app`) |

## Method

1. **Products (A1, A2).** BigBasket quantities are parsed into grams, millilitres or pieces; price per
   100 g/ml is the selling price divided by pack size. For each brand in each subcategory, the
   smallest pack's price per 100 g/ml divided by the largest pack's is the pack premium, and steps
   between neighbouring pack sizes above 2.5x are flagged as ladder gaps.
2. **Districts (A3).** 707 NFHS-5 districts, with HCES survey weights giving district population and
   urban share, and HCES spend per person for 11 categories by state and rural/urban.
3. **Clusters (A4).** 13 features, StandardScaler, PCA (85% variance), KMeans with k from 4 to 8 by
   silhouette score, then a check of how stable the groups are.
4. **Whitespace (A5).** Opportunity index = (benchmark cluster spend per person - cluster spend per
   person) x population. It is a relative score, not rupees of sales.

Every proxy and simplification is logged in [docs/assumptions.md](docs/assumptions.md).

## Limitations

- BigBasket is one online retailer at one point in time; it is not what a kirana shelf looks like.
- HCES spend is only reliable at state level (the survey's own guidance), so districts in a state
  share spend values and differ through urban share. Clusters therefore reflect NFHS-5 development
  indicators, not local spending.
- Clusters are a rough development gradient (silhouette 0.22); one cluster depends on a single
  indicator (drinking water).
- The opportunity index multiplies by population, so large clusters score high in most categories.
- Pack-size and price-point suggestions come from separate tables and are directions, not a price list.
- Maps use 2011 district boundaries: 635 of 706 NFHS-5 districts are drawn; newer districts are listed, not guessed.
- District populations come from survey weights and are relative sizes (HCES is about 13% below
  official projections); 22 districts have imputed populations.

> Independent portfolio project. Not affiliated with any brand.
