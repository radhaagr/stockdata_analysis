# 📊 Stock Data Analysis & Growth Scoring
This repository contains Jupyter notebooks for fetching stock market data with robust error handling and calculating growth scores for selected tickers.
---

## 📁 Project Files

### 1️⃣ FetchAll_StockData_ErrorHandling.ipynb

**Purpose:**
Fetches stock market data for multiple tickers with built-in error handling and validation.

**Features:**
- Bulk ticker data fetching
- API error handling
- Missing data handling
- Logging and exception management
- Clean structured output

**Typical Use Case:**
Use this notebook to download and validate historical stock data before performing analysis.
Currently it is 

---

### 2️⃣ getTickersGrowthScore.ipynb

**Purpose:**
Calculates growth scores for stock tickers based on financial metrics.

**Features:**
- Revenue growth analysis
- EPS growth calculation
- Multi-year comparison
- Growth scoring model
- Ranking tickers by performance

**Typical Use Case:**
Use this notebook to rank companies based on growth performance metrics.
---

## Data Architecture

The current implementation relies on `yfinance` for financial and historical price data.

Due to Yahoo Finance API constraints:
- Financial statements are limited to recent quarters.
- Some tickers may have incomplete reporting fields.
- No guaranteed SLA or data integrity validation.

Future versions will modularize the data layer to allow pluggable data providers.

### Planned Improvements
- Integrate alternative APIs (e.g., Alpha Vantage, Financial Modeling Prep)
- Expand historical financial coverage
- Add data validation cross-checks

## 🛠 Requirements

Install dependencies before running:

```bash
pip install pandas numpy yfinance matplotlib
