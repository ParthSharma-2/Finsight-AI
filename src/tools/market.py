import os
import sys
sys.path.insert(0, os.path.abspath("."))

import yfinance as yf
from langchain.tools import tool

@tool
def get_stock_price(ticker: str) -> str:
    """Get current stock price and key metrics for a ticker symbol.
    For Indian stocks add .NS suffix — example: INFY.NS, TCS.NS, WIPRO.NS
    For US stocks use plain ticker — example: INFY, MSFT, GOOGL"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        info = stock.info

        if hist.empty:
            return f"No data found for {ticker}. Check ticker symbol."

        current = hist['Close'].iloc[-1]
        prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
        change_pct = ((current - prev) / prev) * 100
        volume = hist['Volume'].iloc[-1]

        return f"""
=== {ticker} Stock Data ===
Current Price   : {current:.2f}
Daily Change    : {change_pct:+.2f}%
Volume          : {volume:,}
52W High        : {info.get('fiftyTwoWeekHigh', 'N/A')}
52W Low         : {info.get('fiftyTwoWeekLow', 'N/A')}
Market Cap      : {info.get('marketCap', 'N/A')}
P/E Ratio       : {info.get('trailingPE', 'N/A')}
EPS             : {info.get('trailingEps', 'N/A')}
Dividend Yield  : {info.get('dividendYield', 'N/A')}
"""
    except Exception as e:
        return f"Error fetching {ticker}: {str(e)}"

@tool
def get_stock_history(ticker: str, period: str = "1mo") -> str:
    """Get historical price data for a stock.
    Period options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y
    Example: ticker=INFY.NS period=3mo"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        if hist.empty:
            return f"No historical data for {ticker}"

        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        total_return = ((end_price - start_price) / start_price) * 100
        high = hist['High'].max()
        low = hist['Low'].min()

        return f"""
=== {ticker} Historical Data ({period}) ===
Start Price     : {start_price:.2f}
End Price       : {end_price:.2f}
Total Return    : {total_return:+.2f}%
Period High     : {high:.2f}
Period Low      : {low:.2f}
Trading Days    : {len(hist)}
"""
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def compare_stocks(tickers: str) -> str:
    """Compare multiple stocks side by side.
    Input tickers as comma-separated string.
    Example: INFY.NS,TCS.NS,WIPRO.NS"""
    try:
        ticker_list = [t.strip() for t in tickers.split(",")]
        results = []

        for ticker in ticker_list:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1mo")

            if hist.empty:
                results.append(f"{ticker}: No data")
                continue

            current = hist['Close'].iloc[-1]
            month_return = ((current - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100

            results.append(f"""
{ticker}:
  Price     : {current:.2f}
  1M Return : {month_return:+.2f}%
  P/E       : {info.get('trailingPE', 'N/A')}
  Mkt Cap   : {info.get('marketCap', 'N/A')}""")

        return "\n=== Stock Comparison ===" + "".join(results)
    except Exception as e:
        return f"Comparison error: {str(e)}"


if __name__ == "__main__":
    print("Testing market tools...\n")
    print(get_stock_price.invoke("INFY.NS"))
    print(get_stock_history.invoke({"ticker": "INFY.NS", "period": "1mo"}))
    print(compare_stocks.invoke("INFY.NS,TCS.NS"))