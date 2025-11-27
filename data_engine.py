import yfinance as yf
import pandas as pd
import numpy as np

import yfinance as yf
import pandas as pd
import numpy as np

def download_history(tickers):
    frames = []
    for t in tickers:
        try:
            h = yf.download(t, period="1y")["Close"].rename(t)
            frames.append(h)
        except:
            pass

    if len(frames) == 0:
        return pd.DataFrame()

    return pd.concat(frames, axis=1)

def safe_price(ticker, history):
    t = yf.Ticker(ticker)

    # Try fast but often empty on cloud
    try:
        fi = t.fast_info
        p = fi.get("last_price") or fi.get("regular_market_price")
        if p and p > 0:
            return p
    except:
        pass

    # Try 1-day history (MOST reliable on cloud)
    try:
        h = t.history(period="1d")["Close"]
        if len(h.dropna()) > 0:
            return h.dropna().iloc[-1]
    except:
        pass

    # Try 5-day history
    try:
        h5 = t.history(period="5d")["Close"]
        if len(h5.dropna()) > 0:
            return h5.dropna().iloc[-1]
    except:
        pass

    # Fallback to main history if available
    try:
        if ticker in history.columns:
            return history[ticker].dropna().iloc[-1]
    except:
        pass

    return 0.0  # last fallback


def fetch_market_data(holdings):
    tickers = list(holdings.keys())

    # --- Download full history ---
    try:
        raw = yf.download(tickers, period="1y", group_by="ticker", auto_adjust=True)
        history = download_history(tickers)

    except:
        history = pd.DataFrame()

    # Benchmark
    try:
        bench_raw = yf.download("^GSPC", period="1y")
        benchmark = bench_raw["Close"] if "Close" in bench_raw else None
    except:
        benchmark = None

    fundamentals_list = []

    for ticker in tickers:
        data = holdings[ticker]

        qty = data.get("qty", 0)
        buy_price = data.get("buy_price", 0)

        # --- SAFE PRICE ---
        price = safe_price(ticker, history)

        val = price * qty
        cost_basis = buy_price * qty

        # --- Get sector with fallback ---
        try:
            sector = yf.Ticker(ticker).info.get("sector", "Unknown")
        except:
            sector = "Unknown"

        fundamentals_list.append({
            "Ticker": ticker,
            "Quantity": qty,
            "Current Price": price,
            "Position Value": val,
            "Cost Basis": cost_basis,
            "Unrealized P&L": val - cost_basis,
            "Sector": sector,
        })

    fundamentals = pd.DataFrame(fundamentals_list)

    return history, benchmark, fundamentals, {}


def calculate_portfolio_metrics(history, benchmark):
    if isinstance(history, pd.Series): history = history.to_frame()
    returns = history.pct_change(fill_method=None).dropna()
    
    vol = returns.std() * np.sqrt(252)
    corr = returns.corr() if returns.shape[1] > 1 else pd.DataFrame([[1.0]], columns=returns.columns, index=returns.columns)
    div_score = int(max(0, min(100, (1 - corr.values.mean()) * 100)))
    
    # Benchmark Comparison (Fixed for 1D error)
    port_cum = (1 + returns.mean(axis=1)).cumprod()
    
    bench_cum = None
    if benchmark is not None:
        # FORCE 1D Series
        if isinstance(benchmark, pd.DataFrame):
            benchmark = benchmark.iloc[:, 0]
            
        bench_returns = benchmark.pct_change(fill_method=None).dropna()
        common_index = port_cum.index.intersection(bench_returns.index)
        bench_cum = (1 + bench_returns.loc[common_index]).cumprod()
        port_cum = port_cum.loc[common_index]

    data_map = {"Portfolio": port_cum}
    if bench_cum is not None: data_map["S&P 500"] = bench_cum
        
    comp_df = pd.DataFrame(data_map) * 100
    return vol, corr, div_score, comp_df

def run_monte_carlo(history, weights, current_val, days=90, sims=200):
    if isinstance(history, pd.Series): history = history.to_frame()
    daily_ret = history.pct_change(fill_method=None).dropna()
    
    # Weighted Returns
    port_ret = pd.Series(0.0, index=daily_ret.index)
    total_w = sum(weights.values())
    if total_w == 0: total_w = 1
    
    for t, w in weights.items():
        if t in daily_ret.columns: 
            port_ret += daily_ret[t] * (w / total_w)
            
    mu, sigma = port_ret.mean(), port_ret.std()
    
    # Fast Vectorized Simulation
    sim_data = {}
    for i in range(sims):
        shocks = np.random.normal(0, 1, days)
        daily_growth = np.exp((mu - 0.5 * sigma**2) + sigma * shocks)
        path = [current_val]
        for g in daily_growth: path.append(path[-1] * g)
        sim_data[f"Sim {i}"] = path
        
    return pd.DataFrame(sim_data)
