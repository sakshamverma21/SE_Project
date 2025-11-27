import requests
import streamlit as st
import pandas as pd
import numpy as np

API_KEY = st.secrets["ALPHA_KEY"]


# ============================================================
# REAL-TIME PRICE
# ============================================================

def get_live_price(ticker):
    url = (
        f"https://www.alphavantage.co/query?"
        f"function=GLOBAL_QUOTE&symbol={ticker}&apikey={API_KEY}"
    )
    r = requests.get(url).json()

    if "Global Quote" not in r:
        return None

    try:
        return float(r["Global Quote"]["05. price"])
    except:
        return None


# ============================================================
# HISTORICAL (DAILY OHLC)
# ============================================================

def get_history(ticker):
    url = (
        f"https://www.alphavantage.co/query?"
        f"function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={API_KEY}"
    )

    r = requests.get(url).json()

    if "Time Series (Daily)" not in r:
        return pd.Series(dtype=float)

    df = (
        pd.DataFrame(r["Time Series (Daily)"])
        .T.rename(columns={"5. adjusted close": "Close"})
    )

    df.index = pd.to_datetime(df.index)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = df.sort_index()
    return df["Close"].dropna()


def get_history_multi(tickers):
    frames = []
    for t in tickers:
        h = get_history(t)
        if not h.empty:
            frames.append(h.rename(t))
    if frames:
        return pd.concat(frames, axis=1).dropna()
    return pd.DataFrame()


# ============================================================
# BASIC SECTOR LOOKUP (static fallback)
# ============================================================

STATIC_SECTORS = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Communication Services",
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "NVDA": "Technology",
}

def get_sector(ticker):
    return STATIC_SECTORS.get(ticker.upper(), "Unknown")


# ============================================================
# MAIN MARKET FETCH FUNCTION
# ============================================================

def fetch_market_data(holdings):
    tickers = list(holdings.keys())

    # 1) History
    history = get_history_multi(tickers)

    # 2) Benchmark (S&P500 ETF: SPY)
    benchmark = get_history("SPY")  # SPY is very close to S&P500

    # 3) Fundamentals
    fundamentals_list = []
    sectors_map = {}

    for ticker, data in holdings.items():
        qty = float(data["qty"])
        buy = float(data["buy_price"])

        price = get_live_price(ticker)
        if price is None:
            price = buy

        value = price * qty
        cost = buy * qty
        pnl = value - cost

        sector = get_sector(ticker)
        sectors_map[ticker] = sector

        fundamentals_list.append({
            "Ticker": ticker,
            "Quantity": qty,
            "Current Price": price,
            "Position Value": value,
            "Cost Basis": cost,
            "Unrealized P&L": pnl,
            "Return %": (price / buy - 1) * 100 if buy > 0 else 0,
            "Sector": sector
        })

    fundamentals = pd.DataFrame(fundamentals_list)

    return history, benchmark, fundamentals, sectors_map


# ============================================================
# PORTFOLIO METRICS
# ============================================================

def calculate_portfolio_metrics(history, benchmark):
    if history.empty:
        return pd.Series(dtype=float), pd.DataFrame(), 0, pd.DataFrame()

    returns = history.pct_change().dropna()

    # Volatility
    vol = returns.std() * np.sqrt(252)

    # Correlation
    if returns.shape[1] > 1:
        corr = returns.corr()
    else:
        corr = pd.DataFrame([[1.0]])

    # Diversification
    if returns.shape[1] > 1:
        avg_corr = corr.values[np.triu_indices_from(corr.values, 1)].mean()
    else:
        avg_corr = 1
    div_score = int((1 - avg_corr) * 100)

    # Benchmark comparison
    port_ret = returns.mean(axis=1)
    port_cum = (1 + port_ret).cumprod() * 100
    comp = {"Portfolio": port_cum}

    if benchmark is not None and not benchmark.empty:
        b_ret = benchmark.pct_change().dropna()
        idx = port_cum.index.intersection(b_ret.index)
        comp["Benchmark"] = (1 + b_ret.loc[idx]).cumprod() * 100

    comp_df = pd.DataFrame(comp)

    return vol, corr, div_score, comp_df


# ============================================================
# MONTE CARLO SIMULATION
# ============================================================

def run_monte_carlo(history, weights, current_val, days=90, sims=200):
    if history.empty:
        return pd.DataFrame()

    ret = history.pct_change().dropna()
    port = pd.Series(0, index=ret.index)

    total_w = sum(weights.values()) or 1

    for t, w in weights.items():
        if t in ret.columns:
            port += ret[t] * (w / total_w)

    mu = port.mean()
    sigma = port.std()

    sim = {}

    for i in range(sims):
        shocks = np.random.normal(0, 1, days)
        path = [current_val]
        for s in shocks:
            growth = np.exp((mu - 0.5 * sigma**2) + sigma * s)
            path.append(path[-1] * growth)
        sim[f"Sim {i}"] = path

    return pd.DataFrame(sim)
