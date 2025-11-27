import yfinance as yf
import pandas as pd
import numpy as np

def fetch_market_data(holdings):
    tickers = list(holdings.keys())

    # --- Download history safely ---
    try:
        raw = yf.download(tickers, period="1y")
        if "Close" in raw:
            history = raw["Close"]
        else:
            history = pd.DataFrame()
    except Exception:
        history = pd.DataFrame()

    # Benchmark
    try:
        bench_raw = yf.download("^GSPC", period="1y")
        benchmark = bench_raw["Close"] if "Close" in bench_raw else None
    except:
        benchmark = None

    fundamentals_list = []

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)

            # ---- SAFE PRICE FETCH ----
            info = t.fast_info  # safer alternative to .info

            price = info.get("last_price") \
                 or info.get("regular_market_price") \
                 or info.get("previous_close")

            if not price and ticker in history.columns:
                price = history[ticker].iloc[-1]

            if not price:
                price = 0.0

            # holdings data
            data = holdings.get(ticker, {})
            qty = data.get("qty", 0)
            buy_price = data.get("buy_price", 0)

            val = price * qty

            fundamentals_list.append({
                "Ticker": ticker,
                "Quantity": qty,
                "Current Price": price,
                "Position Value": val,
                "Cost Basis": buy_price * qty,
                "Unrealized P&L": val - buy_price * qty,
                "Sector": info.get("sector", "Unknown")
            })

        except Exception as e:
            # Instead of skip silently, append placeholder row
            fundamentals_list.append({
                "Ticker": ticker,
                "Quantity": holdings[ticker]["qty"],
                "Current Price": 0,
                "Position Value": 0,
                "Cost Basis": holdings[ticker]["qty"] * holdings[ticker]["buy_price"],
                "Unrealized P&L": 0,
                "Sector": "Unknown",
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
