import os
import sys
sys.path.insert(0, os.path.abspath("."))

import sqlite3
import json
from langchain.tools import tool
from pathlib import Path

DB_PATH = "data/processed/finsight.db"

def init_database():
    """Create and populate the SQLite database with sample financial data."""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS company_financials (
            id INTEGER PRIMARY KEY,
            company TEXT,
            year INTEGER,
            revenue REAL,
            net_income REAL,
            eps REAL,
            operating_margin REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS segment_revenue (
            id INTEGER PRIMARY KEY,
            company TEXT,
            year INTEGER,
            segment TEXT,
            revenue REAL,
            growth_pct REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS geography_revenue (
            id INTEGER PRIMARY KEY,
            company TEXT,
            year INTEGER,
            region TEXT,
            revenue REAL
        )
    """)

    financials = [
        ("Infosys", 2024, 153670, 26248, 63.07, 20.1),
        ("Infosys", 2023, 146767, 24108, 57.62, 21.0),
        ("Infosys", 2022, 121641, 22110, 52.52, 22.5),
        ("TCS",     2024, 240893, 46099, 125.43, 24.1),
        ("TCS",     2023, 225458, 42147, 114.61, 23.8),
        ("Wipro",   2024, 89829,  11006, 20.39, 16.5),
        ("Wipro",   2023, 90488,  11375, 20.88, 16.2),
    ]

    segments = [
        ("Infosys", 2024, "Financial Services", 33560, 4.2),
        ("Infosys", 2024, "Retail",             18820, 5.1),
        ("Infosys", 2024, "Communication",      18440, -2.1),
        ("Infosys", 2024, "Energy & Utilities",  12840, 8.3),
        ("Infosys", 2024, "Manufacturing",       18940, 9.2),
        ("Infosys", 2024, "Hi-Tech",             16580, 3.1),
        ("Infosys", 2024, "Life Sciences",       13490, 7.4),
    ]

    geography = [
        ("Infosys", 2024, "North America", 92411),
        ("Infosys", 2024, "Europe",        42267),
        ("Infosys", 2024, "India",          3881),
        ("Infosys", 2024, "Rest of World", 15111),
        ("Infosys", 2023, "North America", 90724),
        ("Infosys", 2023, "Europe",        37675),
        ("Infosys", 2023, "India",          3861),
        ("Infosys", 2023, "Rest of World", 14507),
    ]

    cursor.executemany(
        "INSERT OR IGNORE INTO company_financials VALUES (NULL,?,?,?,?,?,?)",
        financials
    )
    cursor.executemany(
        "INSERT OR IGNORE INTO segment_revenue VALUES (NULL,?,?,?,?,?)",
        segments
    )
    cursor.executemany(
        "INSERT OR IGNORE INTO geography_revenue VALUES (NULL,?,?,?,?)",
        geography
    )

    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

@tool
def query_financial_database(sql_query: str) -> str:
    """Execute a SQL query on the financial database.
    Tables available:
    - company_financials(company, year, revenue, net_income, eps, operating_margin)
    - segment_revenue(company, year, segment, revenue, growth_pct)
    - geography_revenue(company, year, region, revenue)
    All revenue values are in INR crore.
    Example: SELECT company, revenue FROM company_financials WHERE year=2024"""
    try:
        if not Path(DB_PATH).exists():
            init_database()

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()

        if not results:
            return "Query returned no results."

        header = " | ".join(columns)
        separator = "-" * len(header)
        rows = "\n".join([" | ".join(str(v) for v in row) for row in results])

        return f"=== Query Results ===\n{header}\n{separator}\n{rows}"
    except Exception as e:
        return f"SQL Error: {str(e)}"

@tool
def get_revenue_breakdown(company: str, year: int) -> str:
    """Get complete revenue breakdown by segment and geography for a company.
    Example: company=Infosys year=2024"""
    try:
        if not Path(DB_PATH).exists():
            init_database()

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT segment, revenue, growth_pct FROM segment_revenue WHERE company=? AND year=? ORDER BY revenue DESC",
            (company, year)
        )
        segments = cursor.fetchall()

        cursor.execute(
            "SELECT region, revenue FROM geography_revenue WHERE company=? AND year=? ORDER BY revenue DESC",
            (company, year)
        )
        geography = cursor.fetchall()
        conn.close()

        result = f"=== {company} Revenue Breakdown ({year}) ===\n"
        result += "\nBy Segment (INR Crore):\n"
        for seg, rev, growth in segments:
            result += f"  {seg:<25} {rev:>10,.0f}  ({growth:+.1f}%)\n"

        result += "\nBy Geography (INR Crore):\n"
        for region, rev in geography:
            result += f"  {region:<25} {rev:>10,.0f}\n"

        return result
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    init_database()
    print(query_financial_database.invoke(
        "SELECT company, year, revenue, net_income FROM company_financials ORDER BY year DESC"
    ))
    print(get_revenue_breakdown.invoke({"company": "Infosys", "year": 2024}))