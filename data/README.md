# Data cards

One section per source. Raw data is never committed; the scripts in `src/data/` recreate what can
be downloaded automatically, the rest is downloaded by hand into `data/raw/<source>/`.

## BigBasket products (automatic)
- Origin: Kaggle dataset `chinmayshanbhag/big-basket-products`, via `src/data/download_bigbasket.py`
- File: `data/raw/bigbasket/BigBasket.csv`, 8,208 rows, one retailer snapshot (online prices)
- Licence: TODO (check the dataset page on Kaggle)
- Caveats: one retailer, one snapshot, no sales volumes; `Price` is treated as MRP, `DiscountPrice` as
  the selling price (assumption 2)

## Open Food Facts, India subset (optional, automatic)
- Origin: Hugging Face `openfoodfacts/product-database`, `food.parquet` (about 8 GB)
- Licence: the dataset card lists ODbL and AGPL-3.0; verify before reuse
- Not used so far

## NFHS-5 district indicators (manual)
- Origin: National Family Health Survey 5 (2019-21), IIPS / Ministry of Health and Family Welfare,
  district fact sheet data compiled into one workbook (`data/raw/nfhs5/nfhs5_districts.xls`,
  707 districts, 109 columns). Record the exact download page and date here.
- Licence: TODO (government open data; confirm the terms of the source you used)
- Caveats: "*" marks suppressed small-sample values; district boundaries are those of 2019-21;
  column headers carry footnote numbers

## HCES 2022-23 (manual, MoSPI)
- Origin: Household Consumption Expenditure Survey 2022-23, National Statistics Office, MoSPI
  (free microdata login). Unit-level files in `data/raw/hces_2022_23/unit_level/`, documents
  (Vol I, Vol II, layout, state codes, methodology, rider) beside them. 261,746 households.
- Licence / terms: see the "Rider for users of unit level data" PDF; do not redistribute the data
- Caveats: the survey is designed for State/UT estimates, and the data note says to refrain from
  estimates below that level, so spend is used at state x rural/urban level only (assumption 22)

## Census 2011 district population (manual)
- Origin: Primary Census Abstract, Registrar General of India
  (`data/raw/census_2011/census2011_district_pca.xlsx`, 640 districts)
- Licence: TODO (government open data; confirm terms)
- Used only to cross-check the HCES-weight population (district boundaries are those of 2011)

## India district boundaries (automatic)
- Origin: DataMeet "Community Created Maps of India", `Districts/Census_2011` (641 district polygons),
  via `src/data/download_boundaries.py` into `data/raw/boundaries/`
- Licence: Creative Commons Attribution 2.5 India. Credit DataMeet when showing a map
- Caveats: Census 2011 vintage (districts created later have no polygon); DataMeet warns of errors and
  older boundaries in some northeastern states, Jammu and Kashmir and Jharkhand; Telangana and Ladakh
  need a state fix (assumption 41)
