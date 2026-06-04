import sys
import os
sys.path.insert(0, os.path.abspath("."))

from langchain.tools import tool

@tool
def calculate_financial_ratios(data: str) -> str:
    """Calculate key financial ratios from provided numbers.
    Input as JSON string with keys: revenue, net_income, total_assets, 
    total_equity, total_debt, current_assets, current_liabilities
    Example: {"revenue": 153670, "net_income": 26248, "total_equity": 85000}"""
    try:
        import json
        d = json.loads(data)

        results = ["=== Financial Ratios ==="]

        if "net_income" in d and "revenue" in d:
            margin = (d["net_income"] / d["revenue"]) * 100
            results.append(f"Net Profit Margin    : {margin:.2f}%")

        if "net_income" in d and "total_equity" in d:
            roe = (d["net_income"] / d["total_equity"]) * 100
            results.append(f"Return on Equity     : {roe:.2f}%")

        if "net_income" in d and "total_assets" in d:
            roa = (d["net_income"] / d["total_assets"]) * 100
            results.append(f"Return on Assets     : {roa:.2f}%")

        if "total_debt" in d and "total_equity" in d:
            de = d["total_debt"] / d["total_equity"]
            results.append(f"Debt-to-Equity       : {de:.2f}")

        if "current_assets" in d and "current_liabilities" in d:
            cr = d["current_assets"] / d["current_liabilities"]
            results.append(f"Current Ratio        : {cr:.2f}")

        if "revenue" in d and "total_assets" in d:
            at = d["revenue"] / d["total_assets"]
            results.append(f"Asset Turnover       : {at:.2f}")

        return "\n".join(results)
    except json.JSONDecodeError:
        return "Invalid JSON input. Check format."
    except Exception as e:
        return f"Calculation error: {str(e)}"

@tool
def calculate_growth_rate(values: str) -> str:
    """Calculate CAGR and year-on-year growth from a series of values.
    Input as comma-separated numbers from oldest to newest.
    Example: 121641,146767,153670 (revenue for 2022,2023,2024)"""
    try:
        nums = [float(x.strip()) for x in values.split(",")]
        if len(nums) < 2:
            return "Need at least 2 values."

        results = ["=== Growth Analysis ==="]

        for i in range(1, len(nums)):
            growth = ((nums[i] - nums[i-1]) / nums[i-1]) * 100
            results.append(f"Year {i} to {i+1} growth : {growth:+.2f}%")

        years = len(nums) - 1
        cagr = ((nums[-1] / nums[0]) ** (1/years) - 1) * 100
        results.append(f"CAGR ({years}Y)           : {cagr:.2f}%")

        return "\n".join(results)
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    print(calculate_financial_ratios.invoke(
        '{"revenue": 153670, "net_income": 26248, "total_equity": 85000, "total_assets": 120000}'
    ))
    print(calculate_growth_rate.invoke("121641,146767,153670"))