import os
import json
import asyncio
import google.generativeai as genai
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import streamlit as st

# Load API Keys
try:
    google_key = st.secrets["GOOGLE_API_KEY"]
    brave_key = st.secrets.get("BRAVE_API_KEY", None)
except:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    google_key = os.getenv("GOOGLE_API_KEY")
    brave_key = os.getenv("BRAVE_API_KEY")

genai.configure(api_key=google_key)

class IntelliAgent:
    def __init__(self, name, role):
        self.name = name
        self.role = role
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def run(self, context, task):
        prompt = f"IDENTITY: {self.name}\nROLE: {self.role}\nINPUT: {context}\nTASK: {task}\nOUTPUT:"
        try: return self.model.generate_content(prompt).text
        except Exception as e: return f"⚠️ Error: {e}"

# --- 1. NEWS AGENT (Improved: Clean Formatting + Smart Scrape) ---
# Update NewsAgent brave_key retrieval:
class NewsAgent(IntelliAgent):
    def __init__(self):
        super().__init__("News Agent", "Financial Analyst. Summarize web findings.")

    async def run_tools(self, tickers):
        try:
            brave_key = st.secrets["BRAVE_API_KEY"]
        except:
            brave_key = os.getenv("BRAVE_API_KEY")
            
        if not brave_key: 
            return "⚠️ BRAVE_API_KEY missing. Add it in Streamlit secrets."

        # Search Tool
        search_params = StdioServerParameters(
            command="npx", args=["-y", "@modelcontextprotocol/server-brave-search"],
            env={"BRAVE_API_KEY": brave_key}
        )
        # Scraper Tool
        fetch_params = StdioServerParameters(command="uvx", args=["mcp-server-fetch"])

        headlines = "No news found."
        deep_context = "No deep dive available."
        first_url = None

        try:
            # 1. SEARCH (Brave)
            async with stdio_client(search_params) as (r, w):
                async with ClientSession(r, w) as s:
                    await s.initialize()
                    # Increased count to 10 for more results
                    res = await s.call_tool("brave_web_search", arguments={"query": f"{tickers} latest financial news", "count": 10})
                    
                    if res.content:
                        raw_json = res.content[0].text
                        try:
                            # Parse JSON to fix "bad formatting"
                            search_data = json.loads(raw_json)
                            formatted_list = []
                            for item in search_data:
                                title = item.get('title', 'No Title')
                                desc = item.get('description', '')
                                link = item.get('url', '')
                                if not first_url: first_url = link # Grab first link for scraping
                                formatted_list.append(f"• {title}: {desc} ({link})")
                            
                            headlines = "\n".join(formatted_list)
                        except:
                            # Fallback if raw text
                            headlines = raw_json

            # 2. SCRAPE (Fetch) - Now dynamically scrapes the TOP result
            if first_url:
                async with stdio_client(fetch_params) as (r, w):
                    async with ClientSession(r, w) as s:
                        await s.initialize()
                        res = await s.call_tool("fetch", arguments={"url": first_url})
                        if res.content:
                            # Clean up the scraped text (limit to 1000 chars to save tokens)
                            raw_text = res.content[0].text
                            deep_context = " ".join(raw_text.split())[:1000] + "..."

            return f"HEADLINES (Top 10):\n{headlines}\n\nDEEP DIVE ({first_url}):\n{deep_context}"

        except Exception as e: return f"Tool Error: {e}"

    def run(self, tickers, query):
        try: tool_data = asyncio.run(self.run_tools(tickers))
        except Exception as e: tool_data = f"Web search failed: {e}"
        return super().run(tool_data, query)

# --- 2. ADVISOR AGENT (FileSystem MCP) ---
class AdvisorAgent(IntelliAgent):
    def __init__(self):
        super().__init__("Advisor Agent", "CIO. Write detailed memos.")

    async def save_memo(self, content):
        cwd = os.getcwd()
        params = StdioServerParameters(
            command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", cwd]
        )
        try:
            async with stdio_client(params) as (r, w):
                async with ClientSession(r, w) as s:
                    await s.initialize()
                    await s.call_tool("write_file", arguments={"path": f"{cwd}/Investment_Memo.md", "content": content})
            return "✅ Saved 'Investment_Memo.md' to project folder!"
        except Exception as e: return f"❌ Save failed: {e}"

# --- 3. STANDARD AGENTS ---
class RiskAgent(IntelliAgent):
    def __init__(self): super().__init__("Risk Agent", "Risk Manager.")

class RecommenderAgent(IntelliAgent):
    def __init__(self): super().__init__("Recommender Agent", "Portfolio Manager.")
    def run(self, context): return super().run(context, "Suggest 3 stocks for diversification.")
