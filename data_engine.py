import requests
import streamlit as st
import pandas as pd
import numpy as np

RAPID_KEY = st.secrets["RAPIDAPI_KEY"]

BASE_SUMMARY = "https://yh-finance.p.rapidapi.com/stock/v2/get-summary"
BASE_CHART = "https://yh-finance.p.rapidapi.com/stock/v3/get-chart"


# ============================================================
# 1. REAL-TIME PRICE + FUNDAMENTALS
# ============================================================

def get_realtime_data(ticker):
    """
    Gets REAL price + sector + summary using RapidAPI Yahoo Finance.
    Works 100% on Streamlit Cloud.
    """
    params = {"symbol": ticker, "region": "US"}
    headers = {
        "X-RapidAPI-Key": RAPID_KEY,
        "X-RapidAPI-Host": "yh-finance.p.rapidapi.com"
    }

    r = requests.get(BASE_SUMMARY, headers=headers, params=params)
    data = r.json()

    price = data["price"]["regularMarketPrice"]["raw"]
    sector = data["assetProfile"]["sector"]
    market_cap = data["price"].get("marketCap", {}).get("raw", None)
    beta = data.get("defaultKeyStatistics", {}).get("beta", {}).get("raw", None)

    return {
        "price": float(price),
        "sector": sector,
        "market_cap": market_cap,
        "beta": beta
    }


# ============================================================
# 2. HISTORICAL DATA (needed for charts, volatility, Monte Carlo)
# ============================================================

def get_history(ticker, period="1y", interval="1d"):
    """
    Robust chart endpoint from RapidAPI.
    Provides OHLC + close prices.
    """
    params = {
        "symbol": ticker,
        "interval": interval,
        "range": period,
        "region": "US"
    }
    headers = {
        "X-RapidAPI-Key": RAPID_KEY,
        "X-RapidAPI-Host": "yh-finance.p.rapidapi.com"
    }

    r = requests.get(BASE_CHART, headers=headers, params=params)
    data = r.json()

    if "chart" not in data or data["chart"]["result"] is None:
        return pd.Series(dtype=float)

    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]

    df = pd.DataFrame({"Close": closes}, index=pd.to_datetime(timestamps, unit="s"))
    df.index.name = "Date"
    df = df.dropna()

    return df["Close"]


def get_history_multi(tickers):
    frames = []
    for t in tickers:
        h = get_history(t)
        if h.empty:
            continue
        frames.append(h.rename(t))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).dropna()


# ============================================================
# 3. MAIN MARKET FETCH FUNCTION
# ============================================================

def fetch_market_data(holdings):
    """
    holdings = {
        "AAPL": {"qty": 10, "buy_price": 150},
        ...
    }
    """
    tickers = list(holdings.keys())

    # 1) HISTORY FOR PORTFOLIO
    history = get_history_multi(tickers)

    # 2) BENCHMARK
    try:
        benchmark = get_history("^GSPC")
    except:
        benchmark = None

    # 3) FUNDAMENTALS TABLE
    fundamentals_list = []
    sectors_map = {}

    for ticker, data in holdings.items():
        qty = float(data["qty"])
        buy_price = float(data["buy_price"])

        realtime = get_realtime_data(ticker)
        price = realtime["price"]
        sector = realtime["sector"]

        current_val = price * qty
        cost_basis = buy_price * qty
        pnl = current_val - cost_basis

        sectors_map[ticker] = sector

        fundamentals_list.append({
            "Ticker": ticker,
            "Quantity": qty,
            "Current Price": price,
            "Position Value": current_val,
            "Cost Basis": cost_basis,
            "Unrealized P&L": pnl,
            "Return %": (price / buy_price - 1) * 100 if buy_price > 0 else 0,
            "Sector": sector,
            "Market Cap": realtime["market_cap"],
            "Beta": realtime["beta"]
        })

    fundamentals = pd.DataFrame(fundamentals_list)

    return history, benchmark, fundamentals, sectors_map


# ============================================================
# 4. PORTFOLIO METRICS (volatility, correlation, diversification, benchmark)
# ============================================================

def calculate_portfolio_metrics(history, benchmark):
    if history.empty:
        return (
            pd.Series(dtype=float),
            pd.DataFrame(),
            50,
            pd.DataFrame()
        )

    if isinstance(history, pd.Series):
        history = history.to_frame()

    returns = history.pct_change().dropna()
    if returns.empty:
        return (
            pd.Series(dtype=float),
            pd.DataFrame(),
            50,
            pd.DataFrame()
        )

    # Volatility
    vol = returns.std() * np.sqrt(252)

    # Correlation
    corr = returns.corr() if returns.shape[1] > 1 else pd.DataFrame([[1.0]], columns=returns.columns, index=returns.columns)

    # Diversification Score
    if returns.shape[1] > 1:
        avg_corr = corr.values[np.triu_indices_from(corr.values, k=1)].mean()
    else:
        avg_corr = 1
    div_score = int((1 - avg_corr) * 100)

    # Benchmark Comparison
    port_ret = returns.mean(axis=1)
    port_cum = (1 + port_ret).cumprod() * 100

    comp = {"Portfolio": port_cum}

    if benchmark is not None and not benchmark.empty:
        bench_ret = benchmark.pct_change().dropna()
        idx = port_cum.index.intersection(bench_ret.index)
        if len(idx) > 0:
            comp["S&P 500"] = (1 + bench_ret.loc[idx]).cumprod() * 100

    comp_df = pd.DataFrame(comp)

    return vol, corr, div_score, comp_df


# ============================================================
# 5. MONTE CARLO SIMULATION
# ============================================================

def run_monte_carlo(history, weights, current_val, days=90, sims=200):
    if history.empty:
        return pd.DataFrame()

    returns = history.pct_change().dropna()
    if returns.empty:
        return pd.DataFrame()

    port_ret = pd.Series(0, index=returns.index)
    total_w = sum(weights.values()) or 1

    for t, w in weights.items():
        if t in returns.columns:
            port_ret += returns[t] * (w / total_w)

    mu = port_ret.mean()
    sigma = port_ret.std()

    sim = {}

    for i in range(sims):
        shocks = np.random.normal(0, 1, days)
        path = [current_val]
        for s in shocks:
            growth = np.exp((mu - 0.5 * sigma**2) + sigma * s)
            path.append(path[-1] * growth)
        sim[f"Sim {i}"] = path

    return pd.DataFrame(sim)
