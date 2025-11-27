import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from data_engine import fetch_market_data, calculate_portfolio_metrics, run_monte_carlo
from agents import NewsAgent, RiskAgent, AdvisorAgent, RecommenderAgent
from ai_brain import get_chat_response

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="IntelliQuant | AI Investment Suite",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PROFESSIONAL UI ---
st.markdown("""
    <style>
    /* Main Background & Font */
    .stApp {
        background-color: #0e1117;
        font-family: 'Inter', sans-serif;
    }

    /* Card Styling */
    .metric-card {
        background-color: #1e2127;
        border: 1px solid #2e333d;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Headers */
    h1, h2, h3 {
        color: #f0f2f6;
        font-weight: 600;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }

    /* Custom Success/Warning/Error Messages */
    .stAlert {
        border-radius: 4px;
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161920;
        border-right: 1px solid #2e333d;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid #2e333d;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 500;
    }

    /* Buttons */
    .stButton button {
        border-radius: 4px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALIZE AGENTS ---
if 'team' not in st.session_state:
    st.session_state.team = {
        'news': NewsAgent(),
        'risk': RiskAgent(),
        'advisor': AdvisorAgent(),
        'rec': RecommenderAgent()
    }

# --- INITIALIZE CHAT HISTORY ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.title("IntelliQuant")
    st.caption("AI-Powered Portfolio Intelligence")
    st.markdown("---")

    st.subheader("Portfolio Composition")

    # Default Portfolio
    default_data = pd.DataFrame([
        {"Ticker": "AAPL", "Quantity": 10, "Avg Buy Price": 150.0},
        {"Ticker": "MSFT", "Quantity": 5, "Avg Buy Price": 300.0},
        {"Ticker": "NVDA", "Quantity": 8, "Avg Buy Price": 400.0},
        {"Ticker": "GOOGL", "Quantity": 12, "Avg Buy Price": 120.0}
    ])

    edited_df = st.data_editor(
        default_data,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", help="Stock Symbol (e.g. AAPL)", validate="^[A-Za-z]+$"),
            "Quantity": st.column_config.NumberColumn("Qty", min_value=0.01, format="%.2f"),
            "Avg Buy Price": st.column_config.NumberColumn("Buy Price", min_value=0.0, format="$%.2f")
        }
    )

    holdings = {}
    for index, row in edited_df.iterrows():
        if row["Ticker"] and row["Quantity"] > 0:
            holdings[row["Ticker"].strip().upper()] = {
                "qty": row["Quantity"],
                "buy_price": row.get("Avg Buy Price", 0)
            }

    st.markdown("### Simulation Settings")
    sim_days = st.slider("Forecast Horizon (Days)", 30, 365, 90)

    st.markdown("---")
    run_btn = st.button("Initialize Analysis", type="primary", use_container_width=True)
    st.markdown("v2.0.0 | Production Build")

# --- MAIN DASHBOARD ---
st.title("Portfolio Command Center")
st.markdown(f"**Date:** {datetime.now().strftime('%B %d, %Y')} | **Market Status:** Active")

# --- LOGIC: RUN ANALYSIS ---
if run_btn and len(holdings) > 0:
    with st.status("Orchestrating AI Agents...", expanded=True) as status:
        st.write("Connecting to Market Data Engine...")
        history, benchmark, fundamentals, news = fetch_market_data(holdings)

        st.write("Calculating Quantitative Metrics...")
        volatility, correlation, div_score, comparison_df = calculate_portfolio_metrics(history, benchmark)

        st.write("Running Monte Carlo Simulations...")
        # DEBUG: show dataframe content
        st.write("Fundamentals DataFrame:")
        st.write(fundamentals)

        # DEBUG: show all column names
        st.write("Columns in fundamentals:")
        st.write(fundamentals.columns)

        weights = dict(zip(fundamentals['Ticker'], fundamentals['Position Value']))
        total_val = fundamentals['Position Value'].sum()
        sim_df = run_monte_carlo(history, weights, total_val, days=sim_days)

        # Calculate KPIs
        total_cost = fundamentals['Cost Basis'].sum()
        total_pnl = total_val - total_cost
        pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
        sharpe = (pnl_pct / 100) / (volatility.mean() if not isinstance(volatility, float) else volatility)

        # Generate AI Reports
        advisor_context = f"""
        Portfolio Value: ${total_val:,.2f}
        Total P&L: ${total_pnl:,.2f} ({pnl_pct:.1f}%)
        Diversification Score: {div_score}/100
        Holdings: {list(holdings.keys())}
        """

        st.write("Generating Strategic Reports...")
        prompt = f"""
        Analyze this portfolio and output your answer strictly in the following format. Do not use emojis.

        ### Strategic Score: {div_score}/100

        ### Strengths
        * [Point 1]
        * [Point 2]

        ### Weaknesses
        * [Point 1]
        * [Point 2]

        ### Executive Decision
        **[BUY / SELL / HOLD]** - [Professional justification]
        """
        advice_report = st.session_state.team['advisor'].run(advisor_context, prompt)

        ticker_list = ", ".join(list(holdings.keys()))
        news_summary = st.session_state.team['news'].run(ticker_list, "Summarize latest market sentiment and specific news for these tickers. Do not use emojis.")

        rec_summary = st.session_state.team['rec'].run(advisor_context)

        risk_context = f"Vol: {volatility.to_dict() if hasattr(volatility, 'to_dict') else volatility}, Corr: {correlation.to_dict() if hasattr(correlation, 'to_dict') else correlation}"
        risk_report = st.session_state.team['risk'].run(risk_context, "Identify key portfolio risks. Do not use emojis.")

        # SAVE EVERYTHING TO SESSION STATE
        st.session_state.results = {
            "total_val": total_val,
            "total_pnl": total_pnl,
            "pnl_pct": pnl_pct,
            "div_score": div_score,
            "sharpe": sharpe,
            "fundamentals": fundamentals,
            "correlation": correlation,
            "sim_df": sim_df,
            "volatility": volatility,
            "advice_report": advice_report,
            "news_summary": news_summary,
            "rec_summary": rec_summary,
            "risk_report": risk_report,
            "sim_days": sim_days
        }

        st.session_state.analysis_context = f"""
        PORTFOLIO ANALYSIS REPORT
        -------------------------
        Total Value: ${total_val:,.2f}
        Total P&L: ${total_pnl:,.2f} ({pnl_pct:.1f}%)
        Diversification Score: {div_score}/100
        Sharpe Ratio: {sharpe:.2f}

        HOLDINGS:
        {fundamentals.to_string()}

        ADVISOR VERDICT:
        {advice_report}

        RISK REPORT:
        {risk_report}

        NEWS SUMMARY:
        {news_summary}

        OPPORTUNITIES:
        {rec_summary}
        """

        st.session_state.analysis_complete = True
        status.update(label="Analysis Complete", state="complete", expanded=False)

# --- DISPLAY DASHBOARD (IF ANALYSIS COMPLETE) ---
if st.session_state.analysis_complete:
    res = st.session_state.results

    # --- KPI ROW ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Portfolio Value", f"${res['total_val']:,.2f}", delta=None)
    with kpi2:
        st.metric("Total P&L", f"${res['total_pnl']:,.2f}", delta=f"{res['pnl_pct']:.2f}%")
    with kpi3:
        st.metric("Diversification Score", f"{res['div_score']}/100", delta="Target: >70", delta_color="off")
    with kpi4:
        st.metric("Est. Sharpe Ratio", f"{res['sharpe']:.2f}", help="Risk-adjusted return metric")

    # --- TABS LAYOUT ---
    tab_strat, tab_intel, tab_risk, tab_data, tab_chat = st.tabs([
        "AI Strategy", "Market Intelligence", "Risk & Simulation", "Data Explorer", "💬 AI Assistant"
    ])

    # --- TAB 1: STRATEGY ---
    with tab_strat:
        col_advice, col_alloc = st.columns([2, 1])
        with col_advice:
            st.subheader("Chief Investment Officer Verdict")
            st.info(res['advice_report'])
        with col_alloc:
            st.subheader("Asset Allocation")
            fig_pie = px.pie(res['fundamentals'], values='Position Value', names='Ticker', hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300, showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)

            st.subheader("Sector Breakdown")
            fig_sector = px.bar(res['fundamentals'], x='Sector', y='Position Value', color='Sector')
            fig_sector.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=200)
            st.plotly_chart(fig_sector, use_container_width=True)

    # --- TAB 2: INTELLIGENCE ---
    with tab_intel:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Global Market Sentiment")
            st.markdown(f"""
            <div style="background-color: #1e2127; padding: 20px; border-radius: 8px; border-left: 4px solid #3498db;">
                {res['news_summary']}
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.subheader("AI Opportunities")
            st.markdown(f"""
            <div style="background-color: #1e2127; padding: 20px; border-radius: 8px; border-left: 4px solid #2ecc71;">
                {res['rec_summary']}
            </div>
            """, unsafe_allow_html=True)

    # --- TAB 3: RISK & SIMULATION ---
    with tab_risk:
        st.subheader("Monte Carlo Simulation (95% Confidence)")
        sim_df = res['sim_df']
        sim_days = res['sim_days']

        fig_mc = go.Figure()
        for col in sim_df.columns[:50]:
            fig_mc.add_trace(go.Scatter(y=sim_df[col], mode='lines', line=dict(width=1, color='rgba(100, 100, 100, 0.1)'), showlegend=False))

        median_line = sim_df.median(axis=1)
        fig_mc.add_trace(go.Scatter(y=median_line, mode='lines', name='Median Outcome', line=dict(color='#3498db', width=3)))

        p95 = sim_df.quantile(0.95, axis=1)
        p05 = sim_df.quantile(0.05, axis=1)

        fig_mc.add_trace(go.Scatter(y=p95, mode='lines', name='95th Percentile (Upside)', line=dict(color='#2ecc71', width=2, dash='dash')))
        fig_mc.add_trace(go.Scatter(y=p05, mode='lines', name='5th Percentile (Downside)', line=dict(color='#e74c3c', width=2, dash='dash')))

        fig_mc.update_layout(
            title=f"Projected Portfolio Value ({sim_days} Days)",
            xaxis_title="Days",
            yaxis_title="Portfolio Value ($)",
            template="plotly_dark",
            height=500,
            hovermode="x unified"
        )
        st.plotly_chart(fig_mc, use_container_width=True)

        r1, r2 = st.columns(2)
        with r1:
            st.subheader("Risk Analysis")
            st.warning(res['risk_report'])
        with r2:
            st.subheader("Projected Outcomes")
            final_vals = sim_df.iloc[-1]
            total_val = res['total_val']
            st.markdown(f"""
            | Scenario | Projected Value | Change |
            | :--- | :--- | :--- |
            | **Optimistic (95%)** | **${final_vals.quantile(0.95):,.2f}** | <span style='color:#2ecc71'>+{(final_vals.quantile(0.95)/total_val - 1)*100:.1f}%</span> |
            | **Base Case (Median)** | **${final_vals.median():,.2f}** | <span style='color:#3498db'>+{(final_vals.median()/total_val - 1)*100:.1f}%</span> |
            | **Pessimistic (5%)** | **${final_vals.quantile(0.05):,.2f}** | <span style='color:#e74c3c'>{(final_vals.quantile(0.05)/total_val - 1)*100:.1f}%</span> |
            """, unsafe_allow_html=True)

    # --- TAB 4: DATA ---
    with tab_data:
        st.subheader("Holdings Fundamentals")
        st.dataframe(res['fundamentals'], use_container_width=True)
        st.subheader("Correlation Matrix")
        st.dataframe(res['correlation'].style.background_gradient(cmap='RdBu', vmin=-1, vmax=1), use_container_width=True)

    # --- TAB 5: AI ASSISTANT ---
    with tab_chat:
        st.subheader("AI Financial Assistant")
        st.caption("Ask questions about your portfolio, risk metrics, or market trends.")

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about your portfolio..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_chat_response(st.session_state.analysis_context, prompt)
                    st.markdown(response)

            st.session_state.chat_history.append({"role": "assistant", "content": response})

elif not st.session_state.analysis_complete:
    st.info("Get Started: Add your stocks in the sidebar and click 'Initialize Analysis'.")
