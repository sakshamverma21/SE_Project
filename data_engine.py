import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_history(tickers, period="1y"):
    """
    Robust historical data download with fallback mechanisms.
    """
    frames = []
    for t in tickers:
        try:
            logger.info(f"Downloading {t}...")
            # Use auto_adjust=False to get raw OHLCV
            h = yf.download(t, period=period, progress=False)
            if h.empty or h["Close"].isna().all():
                logger.warning(f"No data for {t}, skipping")
                continue
            frames.append(h["Close"].rename(t))
        except Exception as e:
            logger.error(f"Error downloading {t}: {e}")
            pass

    if len(frames) == 0:
        logger.warning("No valid data frames found")
        return pd.DataFrame()

    result = pd.concat(frames, axis=1)
    return result.dropna(how='all')  # Remove completely empty columns


import requests

def safe_price(ticker, history):
    """
    100% cloud-safe price fetcher.
    Uses Yahoo JSON endpoint before yfinance fallbacks.
    """
    # --- 1) Try ultra-stable Yahoo finance JSON ---
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        r = requests.get(url, timeout=5)
        data = r.json()

        close = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
        close = [x for x in close if x is not None]

        if len(close) > 0:
            last_price = close[-1]
            if last_price > 0:
                return float(last_price)
    except:
        pass

    # --- 2) Try 1-day history ---
    try:
        h = yf.download(ticker, period="1d", progress=False)
        if not h.empty:
            p = h["Close"].iloc[-1]
            if p > 0:
                return float(p)
    except:
        pass

    # --- 3) Try 5-day history ---
    try:
        h = yf.download(ticker, period="5d", progress=False)
        if not h.empty:
            prices = h["Close"].dropna()
            if len(prices) > 0:
                return float(prices.iloc[-1])
    except:
        pass

    # --- 4) Try cached history ---
    try:
        if ticker in history.columns:
            p = history[ticker].dropna().iloc[-1]
            if p > 0:
                return float(p)
    except:
        pass

    return 0.0



def get_sector(ticker):
    """
    Get sector information for a ticker.
    """
    try:
        t = yf.Ticker(ticker)
        sector = t.info.get("sector", "Unknown")
        if sector:
            return sector
    except:
        pass
    return "Unknown"


def fetch_market_data(holdings):
    """
    Fetch market data, fundamentals, and benchmark data.
    
    Args:
        holdings: dict like {"AAPL": {"qty": 10, "buy_price": 150.0}, ...}
    
    Returns:
        tuple: (history_df, benchmark_series, fundamentals_df, sectors_dict)
    """
    if not holdings:
        return pd.DataFrame(), None, pd.DataFrame(), {}

    tickers = list(holdings.keys())
    logger.info(f"Fetching data for {len(tickers)} tickers: {tickers}")

    # --- Download full history ---
    history = download_history(tickers, period="1y")

    # --- Fetch benchmark (S&P 500) ---
    benchmark = None
    try:
        logger.info("Fetching S&P 500 benchmark...")
        bench_data = yf.download("^GSPC", period="1y", progress=False)
        if not bench_data.empty:
            benchmark = bench_data["Close"] if isinstance(bench_data, pd.DataFrame) else bench_data
            logger.info(f"Benchmark data: {len(benchmark)} days")
    except Exception as e:
        logger.warning(f"Could not fetch benchmark: {e}")

    # --- Build fundamentals table ---
    fundamentals_list = []
    sectors_map = {}

    for ticker in tickers:
        try:
            data = holdings[ticker]
            qty = float(data.get("qty", 0))
            buy_price = float(data.get("buy_price", 0))

            # Get current price
            price = safe_price(ticker, history)

            if price == 0:
                logger.warning(f"{ticker}: price fetch failed, using buy_price as fallback")
                price = buy_price  # fallback instead of dropping the ticker
            # Calculate metrics
            position_value = price * qty
            cost_basis = buy_price * qty if buy_price > 0 else 0
            unrealized_pnl = position_value - cost_basis

            # Get sector
            sector = get_sector(ticker)
            sectors_map[ticker] = sector

            fundamentals_list.append({
                "Ticker": ticker,
                "Quantity": qty,
                "Current Price": price,
                "Position Value": position_value,
                "Cost Basis": cost_basis,
                "Unrealized P&L": unrealized_pnl,
                "Return %": ((price / buy_price - 1) * 100) if buy_price > 0 else 0,
                "Sector": sector,
            })

        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            continue

    if not fundamentals_list:
        logger.error("No valid holdings after processing")
        return history, benchmark, pd.DataFrame(), {}

    fundamentals = pd.DataFrame(fundamentals_list)
    logger.info(f"Fundamentals table: {len(fundamentals)} holdings")

    return history, benchmark, fundamentals, sectors_map


def calculate_portfolio_metrics(history, benchmark):
    """
    Calculate portfolio-level metrics: volatility, correlation, diversification.
    """
    if history.empty:
        logger.error("Empty history provided")
        return pd.Series(), pd.DataFrame(), 50, pd.DataFrame()

    # Ensure Series->DataFrame conversion
    if isinstance(history, pd.Series):
        history = history.to_frame()

    # Calculate daily returns
    returns = history.pct_change(fill_method=None).dropna()

    if returns.empty:
        logger.warning("No valid returns calculated")
        return pd.Series([1.0]*len(history.columns)), pd.DataFrame([[1.0]]*len(history.columns)), 50, pd.DataFrame()

    # Annualized volatility
    vol = returns.std() * np.sqrt(252)

    # Correlation matrix
    if returns.shape[1] > 1:
        corr = returns.corr()
    else:
        corr = pd.DataFrame([[1.0]], columns=returns.columns, index=returns.columns)

    # Diversification score (0-100): based on average correlation
    avg_corr = corr.values[np.triu_indices_from(corr.values, k=1)].mean() if corr.shape[0] > 1 else 1.0
    div_score = int(max(0, min(100, (1 - avg_corr) * 100)))

    # Benchmark comparison
    port_returns = returns.mean(axis=1)
    port_cum = (1 + port_returns).cumprod() * 100

    comp_data = {"Portfolio": port_cum}

    if benchmark is not None:
        try:
            # Ensure benchmark is 1D Series
            if isinstance(benchmark, pd.DataFrame):
                benchmark = benchmark.iloc[:, 0]

            bench_returns = benchmark.pct_change(fill_method=None).dropna()
            common_idx = port_cum.index.intersection(bench_returns.index)

            if len(common_idx) > 0:
                bench_cum = (1 + bench_returns.loc[common_idx]).cumprod() * 100
                port_cum_aligned = port_cum.loc[common_idx]
                comp_data = {"Portfolio": port_cum_aligned, "S&P 500": bench_cum}
                logger.info(f"Benchmark comparison: {len(common_idx)} common dates")
        except Exception as e:
            logger.warning(f"Could not align benchmark: {e}")

    comp_df = pd.DataFrame(comp_data)
    logger.info(f"Metrics: vol={vol.mean():.2%}, div_score={div_score}, corr_avg={avg_corr:.2f}")

    return vol, corr, div_score, comp_df


def run_monte_carlo(history, weights, current_val, days=90, sims=200):
    """
    Run Monte Carlo simulation on weighted portfolio.
    """
    if history.empty:
        logger.warning("Empty history for Monte Carlo")
        return pd.DataFrame()

    if isinstance(history, pd.Series):
        history = history.to_frame()

    daily_ret = history.pct_change(fill_method=None).dropna()

    # Calculate weighted portfolio returns
    port_ret = pd.Series(0.0, index=daily_ret.index)
    total_w = sum(weights.values())
    if total_w <= 0:
        total_w = 1

    for ticker, w in weights.items():
        if ticker in daily_ret.columns:
            port_ret += daily_ret[ticker] * (w / total_w)

    mu, sigma = port_ret.mean(), port_ret.std()

    # Monte Carlo simulation
    sim_data = {}
    np.random.seed(42)  # For reproducibility

    for i in range(sims):
        shocks = np.random.normal(0, 1, days)
        daily_growth = np.exp((mu - 0.5 * sigma**2) + sigma * shocks)
        path = [current_val]
        for g in daily_growth:
            path.append(path[-1] * g)
        sim_data[f"Sim {i}"] = path

    result = pd.DataFrame(sim_data)
    logger.info(f"Monte Carlo: {sims} simulations over {days} days")
    return result
