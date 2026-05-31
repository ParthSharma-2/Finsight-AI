import os
import sys
sys.path.insert(0, os.path.abspath("."))

import yfinance as yf
from langchain.tools import tool
from langchain_ollama import OllamaLLM
from src.rag.retriever import retrieve_documents, format_context

@tool
def search_financial_document(query: str) -> str:
    """Search the ingested financial document (Infosys Annual Report) 
    for information. Use this for questions about company financials, 
    strategy, business segments, revenue, and annual report data."""
    docs = retrieve_documents(query, k=5)
    if not docs:
        return "No relevant information found in the document."
    return format_context(docs)

@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price and basic info for a given ticker symbol.
    For Indian stocks use .NS suffix (e.g., INFY.NS for Infosys).
    For US stocks use plain ticker (e.g., INFY)."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="5d")
        
        if hist.empty:
            return f"No price data found for ticker: {ticker}"
        
        current_price = hist['Close'].iloc[-1]
        prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change = ((current_price - prev_price) / prev_price) * 100
        
        return f"""
Stock: {ticker}
Current Price: {current_price:.2f}
Change: {change:.2f}%
52W High: {info.get('fiftyTwoWeekHigh', 'N/A')}
52W Low: {info.get('fiftyTwoWeekLow', 'N/A')}
Market Cap: {info.get('marketCap', 'N/A')}
P/E Ratio: {info.get('trailingPE', 'N/A')}
"""
    except Exception as e:
        return f"Error fetching stock data for {ticker}: {str(e)}"

@tool
def calculate_financial_metric(expression: str) -> str:
    """Calculate basic financial metrics and math expressions.
    Input should be a valid math expression like '153670 / 146767 * 100 - 100'
    for percentage growth calculation."""
    try:
        allowed = set('0123456789+-*/()., ')
        if not all(c in allowed for c in expression):
            return "Invalid expression — only numbers and basic operators allowed."
        result = eval(expression)
        return f"Result: {result:.4f}"
    except Exception as e:
        return f"Calculation error: {str(e)}"

@tool  
def get_company_summary(company_name: str) -> str:
    """Get a brief summary and key metrics for a company using its name.
    Tries to find the ticker automatically."""
    try:
        ticker_map = {
            "infosys": "INFY.NS",
            "tcs": "TCS.NS",
            "wipro": "WIPRO.NS",
            "hcl": "HCLTECH.NS",
            "reliance": "RELIANCE.NS",
        }
        ticker = ticker_map.get(company_name.lower(), company_name.upper())
        stock = yf.Ticker(ticker)
        info = stock.info
        return f"""
Company: {info.get('longName', company_name)}
Sector: {info.get('sector', 'N/A')}
Industry: {info.get('industry', 'N/A')}
Employees: {info.get('fullTimeEmployees', 'N/A')}
Description: {info.get('longBusinessSummary', 'N/A')[:300]}...
"""
    except Exception as e:
        return f"Error fetching company info: {str(e)}"

ALL_TOOLS = [
    search_financial_document,
    get_stock_price,
    calculate_financial_metric,
    get_company_summary
]