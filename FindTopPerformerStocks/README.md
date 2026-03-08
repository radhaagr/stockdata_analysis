# 📈 Finding Top Performer Stocks using AlphaVantage

A end-to-end quantitative pipeline that fetches fundamental and price data from [AlphaVantage](https://www.alphavantage.co/), identifies the **top 20% performing stocks** each quarter, and trains an **LSTM model** to predict future top performers using 12-month forward returns.

---

## 🗂️ Repository Structure

| File | Description |
|------|-------------|
| `FetchFromAlphaVantage.ipynb` | Fetches balance sheet, income statement, and price data from the AlphaVantage API |
| `MissingDataHandler.ipynb` | Handles missing values using smart zero imputation with `_was_missing` indicator flags |
| `MergeWithStockPrice.ipynb` | Merges fundamental data with historical close prices on fiscal date |
| `FeatureEngineering.ipynb` | Creates advanced financial metrics and ratios from balance sheet and income statement data. |
| `GetTop20Performer_QuarterWise.ipynb` | Identifies top 20% stocks by return, ranked cross-sectionally per quarter |
| `Top20PerformerStocks.ipynb` | Summary analysis and visualization of top performer characteristics |
| `PredictUpwardMovementStocks.ipynb` | LSTM model to predict top 20% performers using 12-month forward return labels |
| `smart_zero_imputation.py` | Utility script for intelligent missing data imputation |
| `retry_failed.py` | Retry handler for failed AlphaVantage API calls |
| `fundamentals_with_prices.csv` | Merged dataset: fundamentals + close prices |
| `sorted_fundamentals_with_prices.csv` | Sorted version of the merged dataset by ticker and fiscal date |

---

## 🚀 Pipeline Overview

```
AlphaVantage API
      │
      ▼
FetchFromAlphaVantage          ← Pull balance sheet, income statement, prices
      │
      ▼
MissingDataHandler             ← Smart imputation + _was_missing flags
      │
      ▼
MergeWithStockPrice            ← Join fundamentals with close prices
      │
      ▼
GetTop20Performer_QuarterWise  ← Cross-sectional ranking, label top 20%
      │
      ▼
PredictUpwardMovementStocks    ← LSTM model, walk-forward validation
```

---

## 🏗️ Key Features

### Forward Return Label Construction
- **12-month forward return** computed as price appreciation over the next 4 quarters
- **Winsorized** at ±3 std dev per period to remove bankruptcy/acquisition noise
- **Excess return** vs. cross-sectional median to isolate stock-specific alpha
- **Volatility-adjusted return** for risk-adjusted performance ranking
- **Binary label**: top 20% per fiscal period = 1, rest = 0

### Feature Engineering
- Profitability ratios: gross margin, operating margin, ROA, ROE, EBITDA margin
- Leverage ratios: debt-to-equity, debt-to-assets, interest coverage
- Liquidity ratios: current ratio, quick ratio, cash ratio
- Growth signals: YoY revenue, net income, EBITDA, total assets growth
- Momentum signals: 12M momentum (skip-1), 6M momentum, 1Q reversal
- Earnings quality: accruals ratio

### Normalization
- **Cross-sectional z-score** per fiscal period (removes macro regime effects)
- `_was_missing` indicator columns retained as binary features

### Model
- **Stacked LSTM** with BatchNormalization and Dropout
- **Walk-forward (time-based) train/test split** — no temporal leakage
- Class weighting to handle the 80/20 label imbalance
- Early stopping on validation AUC

---

## ⚙️ Setup

### Requirements
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib requests
```

### AlphaVantage API Key
Get a free API key from [alphavantage.co](https://www.alphavantage.co/support/#api-key) and set it in `FetchFromAlphaVantage.ipynb`:
```python
API_KEY = "YOUR_API_KEY_HERE"
```

### Run Order
1. `FetchFromAlphaVantage.ipynb`
2. `MissingDataHandler.ipynb`
3. `MergeWithStockPrice.ipynb`
4. `GetTop20Performer_QuarterWiseV1.ipynb`
5. `PredictUpwardMovementStocks.ipynb`

---

## 📊 Dataset

The merged dataset (`fundamentals_with_prices.csv`) contains ~100 columns including:

- **Identifiers**: `ticker`, `fiscalDateEnding`, `price_date`, `close_price`
- **Balance Sheet**: assets, liabilities, equity, debt, inventory, receivables
- **Income Statement**: revenue, gross profit, EBITDA, net income, R&D, SG&A
- **Missing flags**: `*_was_missing` boolean columns for every financial field

---

## 📌 Notes

- AlphaVantage free tier is rate-limited to **25 requests/day**. `retry_failed.py` handles retries for failed calls automatically.
- The pipeline is designed to be **point-in-time (PIT) correct** — no future data leaks into features.
- Walk-forward split holds out the last 3 years as the test set to simulate live deployment.

---

## 📄 License

MIT
