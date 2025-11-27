# app.py
# Patched Streamlit app with additional guards around Monte Carlo and percentiles
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging
from datetime import datetime
from data_engine import (
    fetch_market_data,
    calculate_portfolio_metrics,
    run_monte_carlo
)
from agents import NewsAgent, RiskAgent, AdvisorAgent, RecommenderAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="IntelliQuant | AI Investment Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    h1, h2, h3 {
        color: #f0f2f6;
        font-weight: 600;
    }

    .metric-card {
        background-color: #1e2127;
        border: 1px solid #2e333d;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    section[data-testid="stSidebar"] {
        background-color: #161920;
        border-right: 1px solid #2e333d;
    }

    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid #2e333d;
    }

    .stButton button {
        border-radius: 4px;
        font-weight: 600;
        background-color: #0969da;
    }

    .stButton button:hover {
        background-color: #0860ca;
    }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if 'team' not in st.session_state:
    st.session_state.team = {
        'news': NewsAgent(),
        'risk': RiskAgent(),
        'advisor': AdvisorAgent(),
        'rec': RecommenderAgent()
    }

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# --- SIDEBAR: PORTFOLIO INPUT ---
with st.sidebar:
    st.title("📈 IntelliQuant")
    st.caption("AI-Powered Portfolio Intelligence")
    st.markdown("---")

    st.subheader("Portfolio Composition")

    # Default portfolio
    default_data = pd.DataFrame([
        {"Ticker": "AAPL", "Quantity": 10, "Avg Buy Price": 150.0},
        {"Ticker": "MSFT", "Quantity": 5, "Avg Buy Price": 300.0},
        {"Ticker": "GOOGL", "Quantity": 8, "Avg Buy Price": 120.0},
        {"Ticker": "NVDA", "Quantity": 4, "Avg Buy Price": 400.0},
        {"Ticker": "TSLA", "Quantity": 3, "Avg Buy Price": 200.0},
    ])

    # Data editor for portfolio
    edited_df = st.data_editor(
        default_data,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", help="Stock Symbol"),
            "Quantity": st.column_config.NumberColumn("Qty", min_value=0.01, format="%.2f"),
            "Avg Buy Price": st.column_config.NumberColumn("Buy Price ($)", min_value=0.0, format="%.2f")
        }
    )

    # Convert to holdings dict
    holdings = {}
    for _, row in edited_df.iterrows():
        ticker = row["Ticker"].strip().upper() if pd.notna(row["Ticker"]) else None
        qty = float(row["Quantity"]) if pd.notna(row["Quantity"]) and row["Quantity"] > 0 else 0
        price = float(row["Avg Buy Price"]) if pd.notna(row["Avg Buy Price"]) else 0

        if ticker and qty > 0:
            holdings[ticker] = {"qty": qty, "buy_price": price}

    st.markdown("### Simulation Settings")
    sim_days = st.slider("Forecast Horizon (Days)", 30, 365, 90, step=10)

    st.markdown("---")
    run_analysis = st.button(
        "🚀 Initialize Analysis",
        type="primary",
        use_container_width=True
    )

    st.markdown("---")
    st.caption("v2.0.0 | Production Build")

# --- MAIN DASHBOARD ---
st.title("Portfolio Command Center")
st.markdown(f"📅 {datetime.now().strftime('%B %d, %Y')} | 🟢 Market Active")

# --- RUN ANALYSIS ---
if run_analysis and len(holdings) > 0:
    with st.spinner("🤖 Orchestrating AI Agents..."):
        try:
            # Step 1: Fetch market data
            st.info("📊 Fetching market data...")
            history, benchmark, fundamentals, sectors_map = fetch_market_data(holdings)

            if fundamentals.empty:
                st.error("❌ Could not fetch market data. Check tickers and try again.")
                st.stop()

            # Step 2: Calculate metrics
            st.info("📈 Calculating portfolio metrics...")
            volatility, correlation, div_score, comparison_df = calculate_portfolio_metrics(
                history, benchmark
            )

            # Step 3: Run Monte Carlo
            st.info("🎲 Running Monte Carlo simulations...")
            # Use Position Value as weight (unchanged)
            weights = dict(zip(fundamentals['Ticker'], fundamentals['Position Value']))
            total_val = fundamentals['Position Value'].sum()

            if total_val > 0:
                # Ensure sims is reasonable (200 default)
                sim_df = run_monte_carlo(history, weights, total_val, days=sim_days, sims=200)
            else:
                st.error("Portfolio has no valid positions.")
                st.stop()

            # After sim_df produced, verify it's usable
            if sim_df is None or sim_df.empty:
                st.error("Simulation failed: no simulation paths produced.")
                st.stop()

            # Ensure numeric columns exist
            numeric_cols = sim_df.select_dtypes(include='number').columns
            if len(numeric_cols) == 0:
                st.error("Simulation produced no numeric results.")
                st.stop()

            # Step 4: Calculate KPIs
            total_cost = fundamentals['Cost Basis'].sum()
            total_pnl = total_val - total_cost
            pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

            # Calculate Sharpe (simplified)
            avg_vol = volatility.mean() if hasattr(volatility, 'mean') else volatility
            try:
                avg_vol_val = float(avg_vol) if pd.api.types.is_scalar(avg_vol) else avg_vol.mean()
            except Exception:
                avg_vol_val = 0.0
            sharpe = (pnl_pct / 100) / max(avg_vol_val, 0.001) if avg_vol_val > 0 else 0

            # Step 5: Generate AI reports
            st.info("🤖 Generating strategic reports...")

            # Advisor context
            advisor_context = f"""
Portfolio Analysis Summary:
- Total Value: ${total_val:,.2f}
- Total P&L: ${total_pnl:,.2f} ({pnl_pct:.1f}%)
- Diversification Score: {div_score}/100
- Portfolio Volatility: {avg_vol_val:.2%}
- Holdings: {list(holdings.keys())}

Holdings Breakdown:
{fundamentals.to_string(index=False)}
"""

            advisor_prompt = f"""
Analyze this portfolio and output strictly in this format:

### Strategic Score: {div_score}/100

### Strengths
* [Key strength 1]
* [Key strength 2]

### Weaknesses
* [Risk or concern 1]
* [Risk or concern 2]

### Executive Decision
**[BUY / SELL / HOLD]** - [Professional justification]
"""

            st.info("📰 Generating news summary...")
            ticker_list = ", ".join(list(holdings.keys()))
            advice_report = st.session_state.team['advisor'].run(advisor_context, advisor_prompt)

            news_summary = st.session_state.team['news'].run(
                ticker_list,
                "Summarize latest market sentiment for these tickers"
            )

            rec_summary = st.session_state.team['rec'].run(advisor_context)

            risk_context = f"""
Portfolio Volatility: {avg_vol_val:.2%}
Diversification: {div_score}/100
Holdings: {len(holdings)}
Avg Correlation: {correlation.values[0, 1] if correlation.shape[0] > 1 else 0:.2f}
"""

            risk_report = st.session_state.team['risk'].run(
                risk_context,
                "Identify key portfolio risks and concentration concerns"
            )

            # Save results
            st.session_state.analysis_results = {
                "total_val": total_val,
                "total_pnl": total_pnl,
                "pnl_pct": pnl_pct,
                "div_score": div_score,
                "sharpe": sharpe,
                "fundamentals": fundamentals,
                "correlation": correlation,
                "sim_df": sim_df,
                "volatility": volatility,
                "comparison_df": comparison_df,
                "advice_report": advice_report,
                "news_summary": news_summary,
                "rec_summary": rec_summary,
                "risk_report": risk_report,
                "sim_days": sim_days,
                "holdings": holdings
            }

            st.session_state.analysis_complete = True
            st.success("✅ Analysis Complete!")

        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            logger.error(f"Analysis error: {e}", exc_info=True)

# --- DISPLAY RESULTS ---
if st.session_state.analysis_complete and st.session_state.analysis_results:
    res = st.session_state.analysis_results

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Portfolio Value",
            f"${res['total_val']:,.0f}",
            help="Current total portfolio value"
        )

    with col2:
        st.metric(
            "Total P&L",
            f"${res['total_pnl']:,.0f}",
            delta=f"{res['pnl_pct']:.1f}%",
            delta_color="off"
        )

    with col3:
        st.metric(
            "Diversification",
            f"{res['div_score']}/100",
            delta="Target: >70" if res['div_score'] < 70 else "Excellent",
            delta_color="off"
        )

    with col4:
        st.metric(
            "Sharpe Ratio",
            f"{res['sharpe']:.2f}",
            help="Risk-adjusted returns"
        )

    st.markdown("---")

    # Tabs
    tab_strat, tab_intel, tab_risk, tab_data = st.tabs([
        "📋 Strategy",
        "📰 Intelligence",
        "⚠️ Risk & Forecast",
        "📊 Data"
    ])

    # TAB 1: Strategy
    with tab_strat:
        col_advice, col_alloc = st.columns([2, 1])

        with col_advice:
            st.subheader("Chief Investment Officer Verdict")
            st.info(res['advice_report'])

        with col_alloc:
            st.subheader("Asset Allocation")
            fig_pie = px.pie(
                res['fundamentals'],
                values='Position Value',
                names='Ticker',
                hole=0.4
            )
            fig_pie.update_layout(height=350, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)

            st.subheader("Sector Breakdown")
            fig_sector = px.bar(
                res['fundamentals'],
                x='Sector',
                y='Position Value',
                color='Sector'
            )
            fig_sector.update_layout(
                showlegend=False,
                height=250,
                margin=dict(t=0, b=0, l=0, r=0)
            )
            st.plotly_chart(fig_sector, use_container_width=True)

    # TAB 2: Intelligence
    with tab_intel:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Market Sentiment")
            st.markdown(res['news_summary'])

        with col2:
            st.subheader("Diversification Opportunities")
            st.markdown(res['rec_summary'])

    # TAB 3: Risk
    with tab_risk:
        st.subheader(f"Monte Carlo Simulation ({res['sim_days']}-Day Forecast)")

        sim_df = res.get('sim_df', pd.DataFrame())
        sim_days = res.get('sim_days', 90)

        # Defensive checks before plotting
        if sim_df is None or sim_df.empty:
            st.warning("No simulation data to display.")
        else:
            # ensure numeric columns
            sim_df = sim_df.select_dtypes(include='number')
            if sim_df.empty:
                st.warning("Simulation produced no numeric paths to plot.")
            else:
                fig_mc = go.Figure()

                # Plot up to first 50 simulations
                for col in sim_df.columns[:50]:
                    fig_mc.add_trace(go.Scatter(
                        y=sim_df[col].values,
                        mode='lines',
                        line=dict(width=1),
                        opacity=0.15,
                        showlegend=False
                    ))

                # Median
                median_line = sim_df.median(axis=1)
                fig_mc.add_trace(go.Scatter(
                    y=median_line,
                    name='Median',
                    mode='lines',
                    line=dict(width=3)
                ))

                # Percentiles - guard these calculations
                try:
                    p95 = sim_df.quantile(0.95, axis=1)
                    p05 = sim_df.quantile(0.05, axis=1)
                except Exception as e:
                    p95 = None
                    p05 = None
                    logger.warning(f"Percentile calculation failed: {e}")

                if p95 is not None:
                    fig_mc.add_trace(go.Scatter(
                        y=p95,
                        name='95th Percentile',
                        mode='lines',
                        line=dict(width=2, dash='dash')
                    ))

                if p05 is not None:
                    fig_mc.add_trace(go.Scatter(
                        y=p05,
                        name='5th Percentile',
                        mode='lines',
                        line=dict(width=2, dash='dash')
                    ))

                fig_mc.update_layout(
                    title=f"Portfolio Value Forecast ({sim_days} Days)",
                    xaxis_title="Days",
                    yaxis_title="Portfolio Value ($)",
                    template="plotly_dark",
                    height=500,
                    hovermode="x unified"
                )

                st.plotly_chart(fig_mc, use_container_width=True)

                # Risk analysis
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Risk Assessment")
                    st.warning(res.get('risk_report', 'No risk report available.'))

                with col2:
                    st.subheader("Projected Outcomes")
                    try:
                        final_vals = sim_df.iloc[-1]
                        tv = res['total_val'] or 1.0

                        opt = final_vals.quantile(0.95)
                        med = final_vals.median()
                        pess = final_vals.quantile(0.05)

                        st.write("| Scenario | Value | Change |")
                        st.write("|:---|---:|---:|")
                        st.write(f"| Optimistic (95%) | ${opt:,.0f} | +{(opt / tv - 1) * 100:.1f}% |")
                        st.write(f"| Base (Median) | ${med:,.0f} | +{(med / tv - 1) * 100:.1f}% |")
                        st.write(f"| Pessimistic (5%) | ${pess:,.0f} | {(pess / tv - 1) * 100:.1f}% |")
                    except Exception as e:
                        st.info("Unable to compute projected outcomes: " + str(e))

    # TAB 4: Data
    with tab_data:
        st.subheader("Holdings Fundamentals")
        st.dataframe(res['fundamentals'], use_container_width=True, hide_index=True)

        st.subheader("Correlation Matrix")
        try:
            st.dataframe(
                res['correlation'].style.background_gradient(cmap='RdBu', vmin=-1, vmax=1),
                use_container_width=True
            )
        except Exception:
            st.write("Correlation data unavailable")

        if not res['comparison_df'].empty:
            st.subheader("Performance vs Benchmark")
            fig_comp = go.Figure()

            for col in res['comparison_df'].columns:
                fig_comp.add_trace(go.Scatter(
                    y=res['comparison_df'][col],
                    name=col,
                    mode='lines'
                ))

            fig_comp.update_layout(
                title="Portfolio vs S&P 500 (1-Year)",
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                template="plotly_dark",
                height=400
            )

            st.plotly_chart(fig_comp, use_container_width=True)

else:
    st.info("👈 Add stocks in the sidebar and click 'Initialize Analysis' to begin.")
