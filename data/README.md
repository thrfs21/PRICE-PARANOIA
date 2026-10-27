# PRICE PARANOIA — Simulated Dataset

## Important data provenance note
All rows in this dataset are synthetic/simulated and were generated for the PRICE PARANOIA academic mini-project. They are NOT actual prices collected from Amazon, Flipkart, or any retailer. Do not present them as real market history.

## Files
- `price_history.csv`: 5,400 daily observations for 30 products over 180 days (2026-04-30 to 2026-10-26).
- `product_catalog.csv`: one row per product with category, brand, platform label, MRP, and currency.
- `latest_price_snapshot.csv`: most recent observation per product plus illustrative advertised-price/discount fields.
- `scenario_test_labels.csv`: scenario labels for testing only. Do not use these labels as model input features.
- `raw_prices_for_cleaning_practical.csv`: deliberately dirty sample with duplicates, missing prices, and inconsistent product-name formatting for data-cleaning demonstrations.

## Suggested schema
`product_id`, `product_name`, `category`, `brand`, `platform`, `date`, `price_inr`, `mrp_inr`, `currency`, `data_source`

## Suggested project uses
- Pandas loading and exploration (`head`, `tail`, `info`, `describe`)
- Cleaning: missing values, duplicate rows, whitespace/case normalization, date parsing
- Statistics: mean, median, min, max, percentage changes
- Visualization: price trends, comparisons, anomaly markers
- Anomaly detection: IQR on price values per product
- Discount analysis: compare current price to MRP and historical median/low
- SQLite storage and queries (optional)

## Cautions
- `advertised_original_price_inr` is set from the synthetic MRP in the demo snapshot. This is an illustrative scenario, not evidence of an actual retailer's advertised claim.
- A historical price lower than MRP does not prove that a discount is deceptive.
- The dataset is deterministic and can be regenerated with the same seed.
