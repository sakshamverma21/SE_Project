import yfinance as yf
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# REAL PRICE FETCHER (bulletproof on Streamlit Cloud)
# ============================================================

def get_live_price(ticker):
    t = yf.Ticker(ticker)

    # --- 1) Real-time / 1-min price (MOST ACCURATE) ---
    try:
        h = t.history(period="1d", interval="1m")
        if not h.empty:
            price = float(h["Close"].dropna().iloc[-1])
            if price > 0:
                return price
    except:
        pass

    # --- 2) 1-day EOD fallback ---
    try:
        h = t.history(period="1d")
        if not h.empty:
            price = float(h["Close"].dropna().iloc[-1])
            if price > 0:
                return price
    except:
        pass

    # --- 3) 5-day fallback ---
    try:
        h = t.history(period="5d")
        prices = h["Close"].dropna()
        if not prices.empty:
            return float(prices.iloc[-1])
    except:
        pass

    logger.warning(f"{ticker}: Could NOT fetch price.")
    return None


# ============================================================
# RELIABLE HISTORY FETCHER (Cloud-safe)
# ============================================================

def download_history(tickers, period="1y"):
    frames = []

    for t in tickers:
        try:
            df = yf.Ticker(t).history(period=period)
            if df.empty:
                logger.warning(f"{t}: Empty history.")
                continue
            frames.append(df["Close"].rename(t))
        except Exception as e:
            logger.error(f"History error {t}: {e}")

    if not frames:
        return pd.DataFrame()

    hist = pd.concat(frames, axis=1).dropna(how="all")
    return hist


# ============================================================
# SECTOR FETCHER
# ============================================================

def get_sector(ticker):
    try:
        t = yf.Ticker(ticker)
        sector = t.info.get("sector", "Unknown")
        return sector
    except:
        return "Unknown"


# ============================================================
# MAIN MARKET DATA FUNCTION
# ============================================================

def fetch_market_data(holdings):
    """
    holdings = {
        "AAPL": {"qty": 10, "buy_price": 150},
        "MSFT": {"qty": 5, "buy_price": 280},
        ...
    }
    """
    if not holdings:
        return pd.DataFrame(), None, pd.DataFrame(), {}

    tickers = list(holdings.keys())
    logger.info(f"Fetching holdings: {tickers}")

    # ---------- 1) Full price history ----------
    history = download_history(tickers)

    # ---------- 2) S&P 500 Benchmark ----------
    try:
        bench = yf.Ticker("^GSPC").history(period="1y")
        benchmark = bench["Close"]
    except:
        benchmark = None

    # ---------- 3) Fundamentals Table ----------
    fundamentals_list = []
    sectors_map = {}

    for ticker in tickers:
        qty = float(holdings[ticker].get("qty", 0))
        buy_price = float(holdings[ticker].get("buy_price", 0))

        # --- Fetch REAL price ---
        price = get_live_price(ticker)

        # --- If still missing, fallback to buy price for now ---
        if price is None:
            price = buy_price

        # --- Build row ---
        current_val = price * qty
        cost = buy_price * qty
        pnl = current_val - cost

        sector = get_sector(ticker)
        sectors_map[ticker] = sector

        fundamentals_list.append({
            "Ticker": ticker,
            "Quantity": qty,
            "Current Price": price,
            "Position Value": current_val,
            "Cost Basis": cost,
            "Unrealized P&L": pnl,
            "Return %": ((price / buy_price - 1) * 100) if buy_price else 0,
            "Sector": sector
        })

    fundamentals = pd.DataFrame(fundamentals_list)

    return history, benchmark, fundamentals, sectors_map


# ============================================================
# PORTFOLIO METRICS
# ============================================================

def calculate_portfolio_metrics(history, benchmark):
    if history.empty:
        return (
            pd.Series(dtype=float),
            pd.DataFrame(),
            50,
            pd.DataFrame()
        )

    # --- Ensure correct format ---
    if isinstance(history, pd.Series):
        history = history.to_frame()

    # --- Calculate daily returns ---
    returns = history.pct_change().dropna()

    if returns.empty:
        return (
            pd.Series(dtype=float),
            pd.DataFrame([[1.0]]),
            50,
            pd.DataFrame()
        )

    # --- Volatility ---
    vol = returns.std() * np.sqrt(252)

    # --- Correlation ---
    if returns.shape[1] > 1:
        corr = returns.corr()
    else:
        corr = pd.DataFrame([[1.0]], columns=history.columns, index=history.columns)

    # --- Diversification Score ---
    if corr.shape[0] > 1:
        avg_corr = np.mean(corr.values[np.triu_indices_from(corr.values, k=1)])
    else:
        avg_corr = 1
    div_score = max(0, min(100, int((1 - avg_corr) * 100)))

    # --- Benchmark Comparison ---
    port_ret = returns.mean(axis=1)
    port_cum = (1 + port_ret).cumprod() * 100

    comp_data = {"Portfolio": port_cum}

    if benchmark is not None:
        bench_ret = benchmark.pct_change().dropna()
        idx = port_cum.index.intersection(bench_ret.index)
        if len(idx) > 0:
            bench_cum = (1 + bench_ret.loc[idx]).cumprod() * 100
            comp_data["S&P 500"] = bench_cum

    comp_df = pd.DataFrame(comp_data)

    return vol, corr, div_score, comp_df


# ============================================================
# MONTE CARLO SIMULATION
# ============================================================

def run_monte_carlo(history, weights, current_val, days=90, sims=200):
    if history.empty:
        return pd.DataFrame()

    if isinstance(history, pd.Series):
        history = history.to_frame()

    daily_ret = history.pct_change().dropna()

    port_ret = pd.Series(0.0, index=daily_ret.index)
    total_w = sum(weights.values()) or 1

    for t, w in weights.items():
        if t in daily_ret.columns:
            port_ret += daily_ret[t] * (w / total_w)

    mu = port_ret.mean()
    sigma = port_ret.std()

    sim_data = {}

    for i in range(sims):
        shocks = np.random.normal(0, 1, days)
        growth = np.exp((mu - 0.5 * sigma**2) + sigma * shocks)

        path = [current_val]
        for g in growth:
            path.append(path[-1] * g)

        sim_data[f"Sim {i}"] = path

    return pd.DataFrame(sim_data)
