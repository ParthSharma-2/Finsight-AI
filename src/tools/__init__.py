from src.tools.market import get_stock_price, get_stock_history, compare_stocks
from src.tools.news import get_company_news, get_analyst_recommendations
from src.tools.sql_tool import query_financial_database, get_revenue_breakdown, init_database

ALL_FINANCIAL_TOOLS = [
    get_stock_price,
    get_stock_history,
    compare_stocks,
    get_company_news,
    get_analyst_recommendations,
    query_financial_database,
    get_revenue_breakdown,
]