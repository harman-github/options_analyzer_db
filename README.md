# 📈 Unusual Options Flow Analyzer

A Python-based tool designed to analyze daily "unusual options flow" data. While this repository contains the source code which you are free to fork and explore, the primary way to access this dashboard is through our live web application.

## 🌟 Access the Dashboard

The fully deployed tool is available here:
### [https://optionsanalyzer.streamlit.app/](https://optionsanalyzer.streamlit.app/)

> **🔒 Note:** This dashboard is password protected. Access is exclusive to **Trading Edge Club** members.

The web dashboard is updated daily immediately after `u/TearRepresentative56` adds the day's Unusual Options entries to the database.

## 🎯 The Goal

The primary aim of this project is to provide retail traders with a quantitative "significance score" for unusual options activity. While raw data tells us *that* a whale bought options, this dashboard tells us *how big* that bet is relative to the company's total market value.

**Data Source:** Daily Unusual Options Flow Database (courtesy of `u/TearRepresentative56`).

## 🚀 For Developers: Forking & Cloning

You are welcome to fork this repository to experiment with the code or build your own version.

**Quick Start:**

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/options-flow-analyzer.git](https://github.com/yourusername/options-flow-analyzer.git)
    cd options-flow-analyzer
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas gspread yfinance
    ```

3.  **Setup Credentials:**
    You will need your own Google Cloud Platform (GCP) Service Account `credentials.json` file placed in the root directory to access Google Sheets programmatically.

## 📊 How It Works (The Math)

The tool attempts to answer: *"Is this trade actually big, or is it just big for me?"*

It uses the following logic:
1.  **Ingest:** Pulls rows like `Ticker: AAPL, Strike: 200, Premium: $5.4M`.
2.  **Fetch:** Gets AAPL Market Cap (e.g., $3.0 Trillion).
3.  **Ratio:** Calculates `(Total Premium / Market Cap) * 100`.

*A $1M premium on a penny stock is significantly more "unusual" than a $1M premium on NVDA or AAPL.*

## 🛠️ Advanced Tools

This dashboard represents the foundational analysis of unusual options flow. Since its development, **Trading Edge Club** has introduced a suite of significantly more advanced tools.

For access to the full suite of advanced trading tools, visit:
### [http://tools.tradingedge.club/](http://tools.tradingedge.club/)

## ⚠️ Disclaimer

This tool is for educational and research purposes only. It does not constitute financial advice. Option flow data represents past trades and does not guarantee future price movement. Trading options involves significant risk.

---
*Data provided daily by TearRepresentative56.*
