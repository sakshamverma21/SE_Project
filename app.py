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

# --- ULTRA-MODERN CSS (React-like) ---
st.markdown("""
    <style>
    /* Import Modern Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Root Variables */
    :root {
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --secondary: #8b5cf6;
        --success: #10b981;
        --danger: #ef4444;
        --warning: #f59e0b;
        --bg-primary: #0f0f23;
        --bg-secondary: #1a1a2e;
        --bg-card: #16213e;
        --text-primary: #ffffff;
        --text-secondary: #94a3b8;
        --border: #2d3748;
        --shadow: rgba(99, 102, 241, 0.15);
    }
    
    /* Base App */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Remove Default Padding */
    .block-container {
        padding: 2rem 3rem !important;
        max-width: 100% !important;
    }
    
    /* Hide Sidebar Default Styling */
    section[data-testid="stSidebar"] {
        background: rgba(22, 33, 62, 0.95) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(148, 163, 184, 0.1);
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.3);
    }
    
    section[data-testid="stSidebar"] > div {
        padding: 2rem 1.5rem;
    }
    
    /* Gradient Text for Logo */
    .gradient-text {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.5rem;
    }
    
    /* Typography */
    h1 {
        color: var(--text-primary) !important;
        font-weight: 800 !important;
        font-size: 2.5rem !important;
        letter-spacing: -0.03em !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 0 40px rgba(99, 102, 241, 0.3);
    }
    
    h2 {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
        margin-top: 2rem !important;
    }
    
    h3 {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 1.125rem !important;
    }
    
    /* Glass Card Effect */
    .glass-card {
        background: rgba(22, 33, 62, 0.6);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 48px rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.3);
    }
    
    /* Metric Cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(22, 33, 62, 0.8) 0%, rgba(26, 26, 46, 0.8) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 16px;
        padding: 1.75rem !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 16px 48px rgba(99, 102, 241, 0.25);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-size: 0.813rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 2.25rem !important;
        font-weight: 800 !important;
        text-shadow: 0 0 20px rgba(99, 102, 241, 0.3);
    }
    
    [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }
    
    /* Modern Tabs */
    .stTabs {
        background: transparent;
        padding: 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(22, 33, 62, 0.4);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 0.5rem;
        border: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: var(--text-secondary);
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border: none;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(99, 102, 241, 0.1);
        color: var(--text-primary);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    
    /* Button Styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.02em;
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: none;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(99, 102, 241, 0.5);
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Data Editor */
    [data-testid="stDataFrame"], .stDataFrame {
        background: rgba(22, 33, 62, 0.6) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(148, 163, 184, 0.1) !important;
        border-radius: 12px !important;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: rgba(22, 33, 62, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 8px;
        color: var(--text-primary);
        padding: 0.75rem;
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: var(--primary) !important;
    }
    
    /* Info/Warning/Success Boxes */
    .stAlert {
        background: rgba(22, 33, 62, 0.6);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 12px;
        border-left: 4px solid var(--primary);
    }
    
    /* Divider */
    hr {
        border-color: rgba(148, 163, 184, 0.1) !important;
        margin: 2rem 0 !important;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 9999px;
        color: var(--success);
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background: var(--success);
        border-radius: 50%;
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Plotly Chart Containers */
    .js-plotly-plot {
        border-radius: 16px !important;
        overflow: hidden;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-dark);
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-top-color: var(--primary) !important;
    }
    
    /* Caption Styling */
    .caption {
        color: var(--text-secondary);
        font-size: 0.875rem;
        font-weight: 500;
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
    st.markdown('<div class="gradient-text">IntelliQuant</div>', unsafe_allow_html=True)
    st.markdown('<p class="caption">AI-Powered Portfolio Intelligence</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### Portfolio Composition")

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
    run_analysis = st.button("🚀 Initialize Analysis", type="primary")

    st.markdown("---")
    st.markdown('<p class="caption">v2.0.0 | Production Build</p>', unsafe_allow_html=True)

# --- MAIN DASHBOARD ---
st.markdown(f'<h1>Portfolio Command Center</h1>', unsafe_allow_html=True)
st.markdown(f'''
    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 2rem;">
        <span class="caption">📅 {datetime.now().strftime('%B %d, %Y')}</span>
        <span class="status-badge">
            <span class="status-dot"></span>
            Market Active
        </span>
    </div>
''', unsafe_allow_html=True)

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
            weights = dict(zip(fundamentals['Ticker'], fundamentals['Position Value']))
            total_val = fundamentals['Position Value'].sum()

            if total_val > 0:
                sim_df = run_monte_carlo(history, weights, total_val, days=sim_days, sims=200)
            else:
                st.error("Portfolio has no valid positions.")
                st.stop()

            # Step 4: Calculate KPIs
            total_cost = fundamentals['Cost Basis'].sum()
            total_pnl = total_val - total_cost
            pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

            # Calculate Sharpe (simplified)
            avg_vol = volatility.mean() if hasattr(volatility, 'mean') else volatility
            sharpe = (pnl_pct / 100) / max(avg_vol, 0.001) if avg_vol > 0 else 0

            # Step 5: Generate AI reports
            st.info("🤖 Generating strategic reports...")

            # Advisor context
            advisor_context = f"""
Portfolio Analysis Summary:
- Total Value: ${total_val:,.2f}
- Total P&L: ${total_pnl:,.2f} ({pnl_pct:.1f}%)
- Diversification Score: {div_score}/100
- Portfolio Volatility: {avg_vol:.2%}
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
Portfolio Volatility: {avg_vol:.2%}
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

    # KPI Row with modern cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Portfolio Value",
            f"${res['total_val']:,.0f}",
            help="Current total portfolio value"
        )

    with col2:
        delta_color = "normal" if res['pnl_pct'] >= 0 else "inverse"
        st.metric(
            "Total P&L",
            f"${res['total_pnl']:,.0f}",
            delta=f"{res['pnl_pct']:.1f}%"
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

    st.markdown("<br>", unsafe_allow_html=True)

    # Modern Tabs
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
            st.markdown("### Chief Investment Officer Verdict")
            st.info(res['advice_report'])

        with col_alloc:
            st.markdown("### Asset Allocation")
            fig_pie = px.pie(
                res['fundamentals'],
                values='Position Value',
                names='Ticker',
                hole=0.5,
                color_discrete_sequence=px.colors.sequential.Plasma
            )
            fig_pie.update_layout(
                height=350,
                margin=dict(t=20, b=0, l=0, r=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f0f2f6')
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("### Sector Breakdown")
            fig_sector = px.bar(
                res['fundamentals'],
                x='Sector',
                y='Position Value',
                color='Sector',
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            fig_sector.update_layout(
                showlegend=False,
                height=250,
                margin=dict(t=0, b=0, l=0, r=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f0f2f6'),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(148, 163, 184, 0.1)')
            )
            st.plotly_chart(fig_sector, use_container_width=True)

    # TAB 2: Intelligence
    with tab_intel:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Market Sentiment")
            st.markdown(res['news_summary'])

        with col2:
            st.markdown("### Diversification Opportunities")
            st.markdown(res['rec_summary'])

    # TAB 3: Risk
    with tab_risk:
        st.markdown("### Monte Carlo Simulation")

        sim_df = res['sim_df']
        sim_days = res['sim_days']

        fig_mc = go.Figure()

        # Plot simulations with gradient
        for i, col in enumerate(sim_df.columns[:50]):
            opacity = 0.05 + (i / 50) * 0.05
            fig_mc.add_trace(go.Scatter(
                y=sim_df[col],
                mode='lines',
                line=dict(width=1, color=f'rgba(99, 102, 241, {opacity})'),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Median line
        median_line = sim_df.median(axis=1)
        fig_mc.add_trace(go.Scatter(
            y=median_line,
            name='Median',
            mode='lines',
            line=dict(color='#6366f1', width=4)
        ))

        # Percentiles
        p95 = sim_df.quantile(0.95, axis=1)
        p05 = sim_df.quantile(0.05, axis=1)

        fig_mc.add_trace(go.Scatter(
            y=p95,
            name='95th Percentile',
            line=dict(color='#10b981', width=3, dash='dot')
        ))

        fig_mc.add_trace(go.Scatter(
            y=p05,
            name='5th Percentile',
            line=dict(color='#ef4444', width=3, dash='dot')
        ))

        fig_mc.update_layout(
            title=dict(
                text=f"Portfolio Value Forecast ({sim_days} Days)",
                font=dict(size=20, color='#f0f2f6')
            ),
            xaxis_title="Days",
            yaxis_title="Portfolio Value ($)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(22, 33, 62, 0.3)',
            font=dict(color='#f0f2f6'),
            height=500,
            hovermode="x unified",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(148, 163, 184, 0.1)'),
            legend=dict(
                bgcolor='rgba(22, 33, 62, 0.8)',
                bordercolor='rgba(148, 163, 184, 0.2)',
                borderwidth=1
            )
        )

        st.plotly_chart(fig_mc, use_container_width=True)

        # Risk analysis
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Risk Assessment")
            st.warning(res['risk_report'])

        with col2:
            st.markdown("### Projected Outcomes")
            final_vals = sim_df.iloc[-1]
            tv = res['total_val']

            outcomes_data = pd.DataFrame({
                'Scenario': ['Optimistic (95%)', 'Base (Median)', 'Pessimistic (5%)'],
                'Value': [
                    f"${final_vals.quantile(0.95):,.0f}",
                    f"${final_vals.median():,.0f}",
                    f"${final_vals.quantile(0.05):,.0f}"
                ],
                'Change': [
                    f"+{(final_vals.quantile(0.95)/tv-1)*100:.1f}%",
                    f"+{(final_vals.median()/tv-1)*100:.1f}%",
                    f"{(final_vals.quantile(0.05)/tv-1)*100:.1f}%"
                ]
            })
            
            st.dataframe(outcomes_data, use_container_width=True, hide_index=True)

    # TAB 4: Data
    with tab_data:
        st.markdown("### Holdings Fundamentals")
        st.dataframe(res['fundamentals'], use_container_width=True, hide_index=True)

        st.markdown("### Correlation Matrix")
        st.dataframe(
            res['correlation'].style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1),
            use_container_width=True
        )
