import os
import json
import logging
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API Keys
def load_api_keys():
    """Load API keys from Streamlit secrets or environment."""
    try:
        google_key = st.secrets["GOOGLE_API_KEY"]
    except:
        load_dotenv(override=True)
        google_key = os.getenv("GOOGLE_API_KEY")

    try:
        brave_key = st.secrets.get("BRAVE_API_KEY")
    except:
        brave_key = os.getenv("BRAVE_API_KEY")

    return google_key, brave_key


# Configure Generative AI
google_key, brave_key = load_api_keys()

if google_key:
    genai.configure(api_key=google_key)
else:
    logger.error("GOOGLE_API_KEY not configured")


class IntelliAgent:
    """Base agent class for portfolio analysis."""

    def __init__(self, name, role):
        self.name = name
        self.role = role
        try:
            self.model = genai.GenerativeModel('gemini-2.0-flash')
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            self.model = None

    def run(self, context, task):
        """
        Execute agent task using Gemini API.
        """
        if not self.model:
            return "⚠️ AI model not configured. Check GOOGLE_API_KEY."

        prompt = f"""IDENTITY: {self.name}
ROLE: {self.role}

CONTEXT DATA:
{context}

TASK:
{task}

RESPONSE (No emojis, professional tone):"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    top_p=0.9,
                    max_output_tokens=1000
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"{self.name} error: {e}")
            return f"Error: {str(e)}"


class NewsAgent(IntelliAgent):
    """
    News Agent: Summarizes market sentiment and relevant news.
    Simplified version without MCP for stability.
    """

    def __init__(self):
        super().__init__(
            "News Agent",
            "Financial Analyst. Summarize market news and sentiment for given stocks."
        )

    def run(self, tickers, task):
        """
        Simplified news analysis without web scraping.
        For production, integrate with NewsAPI or Brave Search separately.
        """
        # Create a synthetic context based on ticker analysis
        context = f"""
You are analyzing market sentiment for these tickers: {tickers}

For this analysis, consider:
- General market trends (tech, finance, energy sectors)
- Typical catalysts for these stock types
- Recent market volatility patterns

Provide a realistic market sentiment summary that an investor would find valuable.
"""

        prompt = f"""IDENTITY: {self.name}
ROLE: {self.role}

TICKERS: {tickers}

TASK: {task}

Provide a professional market sentiment analysis without using emojis."""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=800
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"NewsAgent error: {e}")
            return f"Market sentiment analysis unavailable: {str(e)}"


class RiskAgent(IntelliAgent):
    """
    Risk Agent: Analyzes portfolio risk, concentration, and exposure.
    """

    def __init__(self):
        super().__init__(
            "Risk Agent",
            """Quantitative Risk Manager. Analyze portfolio volatility, correlation, 
            sector concentration, and systemic risks. Be strict about over-exposure warnings."""
        )

    def run(self, context, task):
        """
        Analyze risk metrics from portfolio context.
        """
        enhanced_task = f"""{task}

Please provide:
1. Key Risk Findings (volatility, concentration issues)
2. Sector Exposure Warnings (if over 40% in one sector)
3. Correlation Analysis (diversification level)
4. Recommended Actions (rebalance suggestions)

Be direct and specific about risks. No emojis."""

        return super().run(context, enhanced_task)


class AdvisorAgent(IntelliAgent):
    """
    Advisor Agent: Chief Investment Officer - provides overall recommendations.
    """

    def __init__(self):
        super().__init__(
            "Advisor Agent",
            """Chief Investment Officer (CIO). You synthesize quantitative metrics, 
            risk analysis, and market sentiment. Provide clear, actionable Buy/Sell/Hold 
            recommendations with justification. Be conservative and data-driven."""
        )

    def run(self, context, task):
        """
        Generate CIO-level recommendation.
        """
        enhanced_task = f"""{task}

Format your response as:

### Strategic Score: [X]/100

### Portfolio Strengths
- [Key strength]
- [Key strength]

### Key Concerns
- [Risk or issue]
- [Risk or issue]

### Recommendation
**[BUY / SELL / HOLD]** - [Specific reasoning based on data]

### Action Items
1. [Specific action]
2. [Specific action]

No emojis. Professional tone."""

        return super().run(context, enhanced_task)


class RecommenderAgent(IntelliAgent):
    """
    Recommender Agent: Suggests stocks to improve diversification.
    """

    def __init__(self):
        super().__init__(
            "Recommender Agent",
            """Portfolio Manager. Identify missing sectors and recommend specific stocks 
            to improve diversification. Focus on underrepresented sectors in the current portfolio."""
        )

    def run(self, portfolio_context):
        """
        Generate stock recommendations based on portfolio gaps.
        """
        task = """Analyze the portfolio and recommend 3 stocks to improve diversification.

For each recommendation, provide:
1. **TICKER** (Sector) 
   - Why it fills a gap in the current portfolio
   - Expected risk/return profile

Format clearly. No emojis."""

        return super().run(portfolio_context, task)


class RiskEngineAgent(IntelliAgent):
    """
    Risk Engine Agent: Calculates and explains risk metrics.
    (Can be extended to call local risk calculations)
    """

    def __init__(self):
        super().__init__(
            "Risk Engine",
            "Quantitative analyst specializing in portfolio risk modeling."
        )

    def explain_volatility(self, vol_dict, context):
        """Explain volatility metrics."""
        task = f"""
Explain these volatility metrics in investor-friendly language:
{vol_dict}

Portfolio Context: {context}

What do these numbers mean for the investor's risk level?
"""
        return self.run(f"Volatility data: {vol_dict}", task)

    def explain_correlation(self, correlation_matrix, tickers):
        """Explain correlation insights."""
        task = f"""
The portfolio contains {len(tickers)} holdings: {tickers}

Their correlation matrix indicates:
- How related their price movements are
- Diversification effectiveness

Assess: Is this portfolio well-diversified?
"""
        return self.run(f"Holdings: {tickers}", task)


# Utility function for multi-agent collaboration
def collaborate_agents(portfolio_data, agents_to_use):
    """
    Coordinate multiple agents for comprehensive analysis.
    
    Args:
        portfolio_data: dict with analysis results
        agents_to_use: list of agent names to activate
    
    Returns:
        dict with results from each agent
    """
    results = {}

    if "news" in agents_to_use:
        news_agent = NewsAgent()
        tickers = ", ".join(portfolio_data.get("tickers", []))
        results["news"] = news_agent.run(
            tickers,
            "Summarize latest market sentiment and news for these tickers."
        )

    if "risk" in agents_to_use:
        risk_agent = RiskAgent()
        results["risk"] = risk_agent.run(
            portfolio_data.get("risk_context", ""),
            "Identify key portfolio risks and exposures."
        )

    if "advisor" in agents_to_use:
        advisor_agent = AdvisorAgent()
        results["advisor"] = advisor_agent.run(
            portfolio_data.get("advisor_context", ""),
            "Provide CIO-level portfolio recommendation."
        )

    if "recommender" in agents_to_use:
        rec_agent = RecommenderAgent()
        results["recommendations"] = rec_agent.run(
            portfolio_data.get("recommender_context", "")
        )

    return results
