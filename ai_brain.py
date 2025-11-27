import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load API Key

import streamlit as st


# Load API Key from Streamlit secrets OR environment
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("⚠️ GOOGLE_API_KEY is missing! Add it in Streamlit Cloud secrets.")
    st.stop()

genai.configure(api_key=api_key)

# Use the fast Flash model
model = genai.GenerativeModel('gemini-2.0-flash')

def get_agent_response(role, context_data, user_query):
    """
    Standard agent response for News, Risk, and Advisor.
    """
    try:
        system_prompts = {
            "News Agent": "You are a Financial News Analyst. Analyze the provided news headlines and summarize market sentiment.",
            "Risk Agent": "You are a Quantitative Risk Manager. Analyze the volatility and correlation data. Be strict about over-exposure.",
            "Advisor Agent": "You are a Chief Investment Officer. Synthesize the News and Risk insights. Provide a clear Buy/Sell/Hold recommendation."
        }

        prompt = f"""
        ROLE: {system_prompts.get(role, 'Assistant')}
        CONTEXT: {context_data}
        QUERY: {user_query}
        RESPONSE:
        """
        return model.generate_content(prompt).text
    except Exception as e:
        return f"⚠️ AI Error: {str(e)}"

def get_stock_recommendations(portfolio_context):
    """
    Generates 3 stock picks based on missing sectors.
    """
    try:
        prompt = f"""
        You are a Portfolio Manager.
        User's Current Portfolio: {portfolio_context}

        Task: Identify 3 SECTORS or STOCKS that are missing or under-represented in this portfolio to improve diversification.
        Recommend 3 specific tickers.

        Format output exactly like this:
        1. **TICKER** (Sector): Why it fits.
        2. **TICKER** (Sector): Why it fits.
        3. **TICKER** (Sector): Why it fits.
        """
        return model.generate_content(prompt).text
    except Exception as e:
        return f"⚠️ AI Error: {str(e)}"

def get_chat_response(context, user_input):
    """
    Chatbot logic for the AI Assistant tab.
    """
    try:
        prompt = f"""
        You are an intelligent Financial Assistant embedded in the IntelliQuant dashboard.

        CONTEXT OF CURRENT ANALYSIS (Data, Metrics, Reports):
        {context}

        USER QUESTION: {user_input}

        YOUR GOAL:
        Help the user understand their portfolio analysis, explain financial terms, or provide deeper insights based on the data above.
        Keep answers professional, concise, and data-driven. Do not use emojis.
        """
        return model.generate_content(prompt).text
    except Exception as e:
        return f"AI Error: {str(e)}"
