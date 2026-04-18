import os
import sys
sys.path.insert(0, os.path.abspath("."))

from dotenv import load_dotenv
from langchain.tools import tool
import yfinance as yf

load_dotenv()

@tool
def get_company_news(ticker: str) -> str:
    """Get recent news headlines for a company using its ticker symbol.
    Example: ticker=INFY.NS for Infosys, TCS.NS for TCS"""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news

        if not news:
            return f"No recent news found for {ticker}"

        headlines = []
        for article in news[:8]:
            title = article.get('title', 'No title')
            publisher = article.get('publisher', 'Unknown')
            headlines.append(f"• {title} [{publisher}]")

        return f"=== Recent News: {ticker} ===\n" + "\n".join(headlines)
    except Exception as e:
        return f"Error fetching news for {ticker}: {str(e)}"

@tool
def get_financial_calendar(ticker: str) -> str:
    """Get upcoming earnings dates and financial events for a company.
    Example: ticker=INFY.NS"""
    try:
        stock = yf.Ticker(ticker)
        calendar = stock.calendar

        if calendar is None or calendar.empty:
            return f"No upcoming events found for {ticker}"

        return f"=== Financial Calendar: {ticker} ===\n{calendar.to_string()}"
    except Exception as e:
        return f"Error fetching calendar for {ticker}: {str(e)}"

@tool
def get_analyst_recommendations(ticker: str) -> str:
    """Get analyst buy/sell/hold recommendations for a stock.
    Example: ticker=INFY.NS"""
    try:
        stock = yf.Ticker(ticker)
        recs = stock.recommendations

        if recs is None or recs.empty:
            return f"No analyst recommendations found for {ticker}"

        recent = recs.tail(5)
        return f"=== Analyst Recommendations: {ticker} ===\n{recent.to_string()}"
    except Exception as e:
        return f"Error fetching recommendations: {str(e)}"


if __name__ == "__main__":
    print("Testing news tools...\n")
    print(get_company_news.invoke("INFY.NS"))
    print(get_analyst_recommendations.invoke("INFY.NS"))