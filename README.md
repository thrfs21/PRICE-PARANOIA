# PRICE PARANOIA — upgraded DAV mini-project

A Streamlit-based price intelligence dashboard with historical analysis, heuristic deal scoring, discount reality checks, IQR anomaly review, exploratory forecasting, product comparison, a local target-price watchlist, and a data-quality report.

## Run locally
1. Extract the project folder.
2. Open a terminal in the project root.
3. Create/activate a virtual environment (optional but recommended).
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Start the dashboard:
   ```bash
   streamlit run app.py
   ```

## Project structure
- `app.py` — dashboard and user interface
- `src/data_cleaner.py` — data validation, cleaning and quality summary
- `src/price_analyzer.py` — summaries, deal score, discount assessment, IQR anomalies and baseline forecast
- `data/price_history.csv` — historical price observations
- `data/product_catalog.csv` — product metadata
- `watchlist.json` — created automatically when target prices are saved

## Important data note
The supplied dataset is simulated for educational use. The forecast is a simple linear trend baseline with an approximate residual-based band, not a guarantee. Deal scores are transparent heuristics and should not be interpreted as financial advice or proof of deceptive pricing.

## Git update
After replacing the old files with these files:
```bash
git add app.py src requirements.txt README.md data
git commit -m "Upgrade PRICE PARANOIA analytics dashboard"
git push
```
