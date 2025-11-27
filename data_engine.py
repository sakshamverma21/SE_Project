# data_engine.py
# Patched to ensure Monte Carlo never returns an empty DataFrame and to be robust to missing history
import requests
import streamlit as st
import pandas as pd
import numpy as np

API_KEY = None
try:
    API_KEY = st.secrets["ALPHA_KEY"]
except Exception:
    # If running outside Streamlit with .env etc, leave API_KEY None (fetch functions will return empty)
    API_KEY = None


# ============================================================
# REAL-TIME PRICE
# ============================================================

def get_live_price(ticker):
    if not API_KEY:
        return None
    url = (
        f"https://www.alphavantage.co/query?"
        f"function=GLOBAL_QUOTE&symbol={ticker}&apikey={API_KEY}"
    )
    try:
        r = requests.get(url, timeout=10).json()
    except Exception:
        return None

    if "Global Quote" not in r:
        return None

    try:
        return float(r["Global Quote"]["05. price"])
    except Exception:
        return None


# ============================================================
# HISTORICAL (DAILY OHLC)
# ============================================================

def get_history(ticker):
    if not API_KEY:
        return pd.Series(dtype=float)

    url = (
        f"https://www.alphavantage.co/query?"
        f"function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={API_KEY}"
    )

    try:
        r = requests.get(url, timeout=15).json()
    except Exception:
        return pd.Series(dtype=float)

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
        # join on index and drop rows with NaNs to keep consistent returns
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
        try:
            qty = float(data["qty"])
        except Exception:
            qty = 0.0
        try:
            buy = float(data["buy_price"])
        except Exception:
            buy = 0.0

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
        # return sensible empties: volatility Series, correlation DataFrame, diversification score, comp_df
        return pd.Series(dtype=float), pd.DataFrame(), 0, pd.DataFrame()

    returns = history.pct_change().dropna()

    # Volatility
    vol = returns.std() * np.sqrt(252)

    # Correlation
    if returns.shape[1] > 1:
        corr = returns.corr()
    else:
        # Single asset: correlation to itself
        corr = pd.DataFrame([[1.0]], index=returns.columns, columns=returns.columns)

    # Diversification
    if returns.shape[1] > 1:
        # average of upper triangle (exclude diagonal)
        tri_idx = np.triu_indices_from(corr.values, 1)
        try:
            avg_corr = corr.values[tri_idx].mean()
        except Exception:
            avg_corr = 1.0
    else:
        avg_corr = 1.0
    div_score = int((1 - avg_corr) * 100)

    # Benchmark comparison
    port_ret = returns.mean(axis=1)
    port_cum = (1 + port_ret).cumprod() * 100
    comp = {"Portfolio": port_cum}

    if benchmark is not None and not benchmark.empty:
        b_ret = benchmark.pct_change().dropna()
        idx = port_cum.index.intersection(b_ret.index)
        if not idx.empty:
            comp["Benchmark"] = (1 + b_ret.loc[idx]).cumprod() * 100

    comp_df = pd.DataFrame(comp)

    return vol, corr, div_score, comp_df


# ============================================================
# MONTE CARLO SIMULATION (robust)
# ============================================================

def run_monte_carlo(history, weights, current_val, days=90, sims=200):
    """
    Returns a DataFrame with index 0..days and sims columns 'Sim 0', 'Sim 1', ...
    Each column is a numeric path with length days+1 (including day 0 = current_val).
    This function will always return a non-empty DataFrame of numeric dtype.
    """
    # Defensive: ensure days >= 1 and sims >= 1
    days = max(1, int(days))
    sims = max(1, int(sims))

    # If history is empty or not useful, generate flat deterministic sims (repeat current_val)
    if history.empty or not isinstance(history, (pd.DataFrame, pd.Series)) or history.shape[1] < 1:
        # Build deterministic flat paths so UI computations won't fail
        base = [float(current_val)] * (days + 1)
        sim = {f"Sim {i}": list(base) for i in range(sims)}
        df = pd.DataFrame(sim)
        df.index = list(range(days + 1))
        return df.astype(float)

    ret = history.pct_change().dropna()
    port = pd.Series(0.0, index=ret.index)

    total_w = sum(weights.values()) or 1.0

    for t, w in weights.items():
        if t in ret.columns:
            try:
                port += ret[t] * (w / total_w)
            except Exception:
                # ignore asset if broadcast fails
                continue

    # If port is all zeros or degenerate, fallback
    if port.empty or port.std() == 0:
        mu = 0.0
        sigma = 0.0
    else:
        mu = port.mean()
        sigma = port.std()

    sim = {}
    for i in range(sims):
        # generate geometric brownian-like multiplicative shocks
        shocks = np.random.normal(0, 1, days)
        path = [float(current_val)]
        for s in shocks:
            # If sigma is zero, growth becomes 1 (no change)
            growth = np.exp((mu - 0.5 * sigma ** 2) + sigma * s) if sigma != 0 else 1.0
            path.append(path[-1] * growth)
        sim[f"Sim {i}"] = path

    df = pd.DataFrame(sim)
    df.index = list(range(days + 1))
    # Ensure numeric dtype
    df = df.apply(pd.to_numeric, errors="coerce").fillna(method="ffill").fillna(float(current_val))
    return df.astype(float)
