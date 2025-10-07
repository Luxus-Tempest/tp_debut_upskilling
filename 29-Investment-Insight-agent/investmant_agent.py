from agno.agent import Agent
from agno.team.team import Team
from agno.models.openai import OpenAIChat
from agno.models.google import Gemini
from agno.models.mistral import MistralChat
from agno.tools.scrapegraph import ScrapeGraphTools
from agno.tools.file import FileTools
from agno.tools.exa import ExaTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.calculator import CalculatorTools
from dotenv import load_dotenv
import os
from polygon import RESTClient
import numpy as np
import pandas as pd
from scipy.stats import norm

# from huggingface_hub import login

load_dotenv()
id_gemini=os.getenv("id")
api_key_gemini=os.getenv("api_key")
id_openai = os.getenv("id_openai")
api_key_openai = os.getenv("api_key_openai")


MarketResearchBot = Agent(
    name="Market Research Bot",
    model=MistralChat(id="magistral-medium-2507", api_key=os.getenv("MISTRAL_API")),
    tools=[
        ExaTools(),
        YFinanceTools(),
    ],
    description="""
An AI agent that autonomously gathers, monitors, and analyzes market data, financial statements,
news sentiment, and analyst reports to provide actionable investment insights. 
It evaluates trends, sector performance, and macroeconomic indicators to generate buy/sell/hold signals,
prioritize research topics, and deliver structured reports for investment decision-making.
""",
    instructions="""
You are MarketResearchBot, an AI-powered investment analysis agent operating under the Investment Analysis Module. 

## Your Responsibilities:
1. Collect market data, stock prices, company fundamentals, analyst reports, and relevant news using the tools provided (YFinanceTools, ExaTools).
2. Scrape and process data autonomously; do not rely on pre-provided datasets.
3. Analyze trends across sectors, assets, and macroeconomic indicators using configured frameworks (technical, fundamental, sentiment, and quantitative analysis).
4. Identify actionable insights and generate buy/sell/hold signals with confidence levels and supporting reasoning.
5. Prioritize research based on market-moving events, volatility, and sector importance.
6. Summarize findings clearly in structured reports (JSON and/or Markdown) including:
   - Asset-level signals with reasoning
   - Sector summaries
   - Macro trends
   - Alerts for high-risk or unusual activity
   - Sources and citations for all data
7. Use only the tools provided to retrieve data. Do not fabricate or assume information outside of scraped or retrieved data.
8. Include visualizations or charts when relevant to support trends or signals.

## Tool Usage Guidelines:
- Stock prices, financial ratios, fundamentals, company news, or analyst reports → use YFinanceTools and ExaTools.
- Market trends, sentiment analysis, or miscellaneous financial data → use ExaTools.
- Always document the source of the data in your output.

Your goal is to produce a **comprehensive, evidence-backed market research report** that investment analysts or automated systems can directly use for decision-making.
""",
    markdown=True,
    stream=True,
    # reasoning=True,
)


# MarketResearchBot.print_response(
#     """
# You are tasked with producing a **market research report** for the following scenario:

# 1. Focus on the **Technology sectors** in the US market.
# 2. Identify the **top 2 performing stocks** in each sector over the past 3 months.
# 3. Analyze **trend patterns, recent company news, analyst ratings, and key financial ratios**.
# 4. Generate **buy/sell/hold signals** for each stock with reasoning and confidence levels.
# 5. Summarize sector performance and highlight any **macro or market-moving events** affecting these sectors.
# 6. Include **sources for all data**, and provide structured output suitable for downstream processing (JSON and Markdown).

# """, stream=True, stream_intermediate_steps=True
# )


# MarketResearchBot.print_response("""
# Retrieve and analyze the latest market trends for the energy sectors and Electric Vehicles.


# """, stream=True)


import numpy as np
import pandas as pd
def analyze_market_data(data: pd.DataFrame) -> dict:
    if data.empty:
        return {"error": "No data provided"}

    data.columns = [col.capitalize() for col in data.columns]
    close = data["Close"]
    volume = data["Volume"]

    ma20 = close.rolling(window=20).mean().iloc[-1]
    ma50 = close.rolling(window=50).mean().iloc[-1]
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi_value = (100 - (100 / (1 + rs))).iloc[-1]
    volatility = close.pct_change().rolling(window=20).std().iloc[-1]
    avg_volume = volume.rolling(window=20).mean().iloc[-1]

    trend = "Bullish" if ma20 > ma50 else "Bearish" if ma20 < ma50 else "Neutral"

    return {
        "latest_close": round(close.iloc[-1], 2),
        "ma20": round(ma20, 2),
        "ma50": round(ma50, 2),
        "rsi14": round(rsi_value, 2),
        "volatility_20d": round(volatility, 4),
        "avg_volume_20d": int(avg_volume),
        "trend_direction": trend
    }
import pandas as pd
import numpy as np

def PortfolioOptimizationTool(file_path: str, risk_free_rate: float = 0.02) -> dict:
    try:
        df = pd.read_csv(file_path)
        required_cols = {'Ticker', 'ExpectedReturn', 'Volatility', 'Weight'}
        if not required_cols.issubset(df.columns):
            return {"error": f"Missing required columns: {required_cols - set(df.columns)}"}

        if df['ExpectedReturn'].max() > 1:
            df['ExpectedReturn'] /= 100
        if df['Volatility'].max() > 1:
            df['Volatility'] /= 100

        returns = df['ExpectedReturn'].values
        vol = df['Volatility'].values
        sharpe_ratios = (returns - risk_free_rate) / vol
        weights = sharpe_ratios / np.sum(sharpe_ratios)

        df['OptimizedWeight'] = weights
        portfolio_return = np.sum(weights * returns)
        portfolio_vol = np.sqrt(np.sum((weights * vol) ** 2))
        portfolio_sharpe = (portfolio_return - risk_free_rate) / portfolio_vol

        return {
            "optimized_portfolio": df.to_dict(orient='records'),
            "portfolio_return": portfolio_return,
            "portfolio_volatility": portfolio_vol,
            "portfolio_sharpe_ratio": portfolio_sharpe
        }
    except Exception as e:
        return {"error": str(e)}


def equal_weight_portfolio(returns: pd.DataFrame) -> dict:
    """
    Equal-Weight Portfolio Optimizer.
    Args:
        returns (pd.DataFrame): Daily returns of each asset.
    Returns:
        dict: Equal weights and portfolio returns.
    """
    if returns.empty:
        return {"error": "No return data provided"}

    n_assets = returns.shape[1]
    weights = np.repeat(1/n_assets, n_assets)
    portfolio_returns = returns @ weights

    return {
        "method": "Equal Weight",
        "weights": dict(zip(returns.columns, weights)),
        "portfolio_returns": portfolio_returns.tolist()
    }

def risk_parity_portfolio(returns: pd.DataFrame) -> dict:
    """
    Risk Parity Portfolio Optimizer (inverse volatility).
    Args:
        returns (pd.DataFrame): Daily returns of each asset.
    Returns:
        dict: Risk parity weights and portfolio returns.
    """
    if returns.empty:
        return {"error": "No return data provided"}

    volatility = returns.std() * np.sqrt(252)
    inv_vol = 1 / volatility
    weights_rp = inv_vol / inv_vol.sum()
    portfolio_returns_rp = returns @ weights_rp

    return {
        "method": "Risk Parity",
        "weights": dict(zip(returns.columns, weights_rp)),
        "portfolio_returns": portfolio_returns_rp.tolist()
    }


PortfolioOptimizerAgent = Agent(
    name="Portfolio Optimizer Agent",
    # model=Gemini(id=id_gemini, api_key=api_key_gemini),
    model=MistralChat(id="magistral-medium-2507", api_key=os.getenv("MISTRAL_API")),
    tools=[
        analyze_market_data,  # market metrics from OHLCV
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                      historical_prices=True, company_info=True, income_statements=True,
                      key_financial_ratios=True, company_news=True),
        FileTools(),
        ExaTools(),
        PortfolioOptimizationTool,  # mean-variance optimizer
        equal_weight_portfolio,     # new tool
        risk_parity_portfolio,      # new tool
        CalculatorTools(),
    ],
    reasoning=True,
    stream=True,

    description="""
An AI agent that analyzes and optimizes investment portfolios based on risk tolerance,
return objectives, constraints, and market forecasts.
It applies multiple portfolio optimization strategies to propose target allocations,
calculate expected portfolio metrics, and suggest trade actions.
    """,

    instructions="""
You are the Portfolio Optimizer Agent.

Follow these steps in sequence:

1. Use **YFinanceTools** to retrieve:
   - Current portfolio holdings' prices and historical OHLCV data
   - Company fundamentals and analyst recommendations

2. Pass the OHLCV historical data of each asset to **analyze_market_data**
   to compute market metrics (MA20, MA50, RSI14, volatility, volume, trend).

3. Compute daily returns for all assets.

4. Run **equal_weight_portfolio** on the daily returns to get equal allocation results.

5. Run **risk_parity_portfolio** on the daily returns to get risk-balanced allocation results.

6. If the client specifies advanced optimization, prepare a CSV with columns:
   ['Ticker', 'ExpectedReturn', 'Volatility', 'Weight']
   and pass it to **PortfolioOptimizationTool** for mean-variance optimization.

7. Compare:
   - Current allocation
   - Equal-weight allocation
   - Risk parity allocation
   - Mean-variance optimized allocation

8. Identify rebalancing opportunities between the current allocation and the best method.

9. Factor in transaction costs, taxes, and liquidity before final recommendation.

10. Present all results in a structured markdown report:
    - Current portfolio metrics
    - Each optimization method's weights & performance
    - Recommended trades
    - Explanation of risk-return trade-offs
    """,
    markdown=True,
)


# PortfolioOptimizerAgent.print_response("""
# Analyze and optimize the following portfolio using the latest market data you can retrieve:

# Portfolio:
# - AAPL: 30 shares
# - MSFT: 20 shares
# - TSLA: 10 shares
# - SPY: 15 shares
# - Cash: $5,000

# Investor Profile:
# - Risk Tolerance: Balanced
# - Return Objective: Moderate capital growth over the next 5 years
# - Constraints: 
#     * No more than 25% in a single stock
#     * Maintain at least 5% in cash
#     * Avoid fossil fuel companies

# Present your findings in a clear, investor-friendly report.
# """, stream=True, stream_intermediate_steps=True)


RecommendationEngineBot = Agent(
    name="Recommendation Engine Bot",
    # model=OpenAIChat(id=OPENAI_ID, api_key=OPENAI_KEY),
    # model=Gemini(id=id_gemini,api_key=api_key_gemini),
    model=MistralChat(id="magistral-medium-2507", api_key=os.getenv("MISTRAL_API")),
    tools=[
        FileTools(),         # read/write client and product files
        YFinanceTools(),
        # ReasoningTools(),    # multi-step reasoning and rule evaluation
        ExaTools(),          # research lookup for product info and disclosures
        ScrapeGraphTools()   # optional: fetch live product docs or regulatory pages
    ],
    reasoning=True,
    show_tool_calls=True,
    stream=True,

    description="""
    An AI agent that generates personalized investment recommendations for clients by
    combining client profiles, formal risk assessments, current market conditions,
    and a curated product universe. The agent enforces suitability rules, prepares
    required disclosures, and outputs recommendations with rationale and risk/return
    characterization.
    """,
    instructions="""
    You are the Recommendation Engine agent.

    Role:
      - Produce tailored investment recommendations that are suitable for the client
        given their profile, risk assessment, and constraints.

    Inputs expected (either provided directly or fetched via tools):
      - client_profile: {age, income, net_worth, investment_horizon, liquidity_needs, risk_tolerance}
      - risk_assessment: output from Risk Assessment Module (credit, market, compliance summaries)
      - market_conditions: latest market signals and forecasts
      - product_universe: list of investable products with attributes (ticker/id, type, fees, historical_perf, risk_metrics)
      - constraints: regulatory, ESG, concentration limits, liquidity requirements

    Process:
      1. Validate the inputs and check for missing critical fields; if missing, request them.
      2. Apply suitability rules (age, risk_tolerance vs product_risk, liquidity needs).
      3. Score each candidate product on suitability, expected_return (based on forecasts), and risk.
      4. Construct a shortlist of recommended products (primary + alternatives) and a suggested allocation.
      5. Prepare required disclosures (fees, risks, conflicts) and an explanation for each recommendation.
      6. Output a structured JSON response and a human-friendly summary.

    Output format (JSON):
    {
      "client_id": "",
      "recommendations": [
        {
          "product_id": "",
          "product_name": "",
          "product_type": "",
          "recommended_allocation": 0.0,
          "suitability_score": "Low|Medium|High",
          "reasons": ["..."],
          "risks": ["..."],
          "fees": "..."
        }
      ],
      "portfolio_level": {
        "expected_return": "",
        "expected_volatility": "",
        "overall_suitability": "Compliant|Not Compliant",
        "disclosures": ["..."]
      },
      "audit_log_reference": ""
    }

    Rules & Guidelines:
      - Never recommend products that violate explicit client constraints (e.g., ESG exclusion).
      - Enforce suitability: do not recommend high-risk leveraged ETFs to a conservative client.
      - Provide clear disclosures for fees, risks, and any potential conflicts of interest.
      - If the input risk assessment indicates compliance flags, escalate and avoid automated recommendations.
    """,
    markdown=True
)
# RecommendationEngineBot.print_response("""
# Generate personalized investment recommendations for the following client profile:

# Client Profile:
# - Age: 42
# - Investment Horizon: 15 years
# - Risk Tolerance: Moderate to High
# - Current Portfolio: 
#     * 40% US Equities
#     * 20% International Equities
#     * 25% Bonds
#     * 10% Real Estate
#     * 5% Cash
# - Preferences: Interested in technology, healthcare, and renewable energy sectors
# - Restrictions: Avoid tobacco and firearms companies
# """)

investment_router_team = Team(
    name="Investment Router Team",
    mode="route",
    model=Gemini(id=id_gemini, api_key=api_key_gemini),
    members=[MarketResearchBot, PortfolioOptimizerAgent, RecommendationEngineBot],
    show_tool_calls=True,
    markdown=True,
    description="You are an investment query router that directs questions to the appropriate specialized agent.",
    instructions=[
        "Identify the main topic of the user's query and direct it to the relevant agent.",
        "If the query is about market research, trends, stock analysis, sector performance, or related investment insights, route to MarketResearchBot.",
        "If the query is about portfolio optimization, allocation strategies, rebalancing, or risk-return analysis, route to PortfolioOptimizerAgent.",
        "If the query is about personalized investment recommendations, suitability assessments, or product suggestions based on client profiles, route to RecommendationEngineBot.",
        "If the query does not match any of the above categories, respond in English with: 'I can only handle queries related to market research, portfolio optimization, or investment recommendations. Please rephrase your question accordingly.'",
        "Always analyze the query's content before routing to an agent.",
        "For ambiguous queries, ask for clarification before routing.",
    ],
)