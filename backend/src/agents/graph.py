import os
import sys

sys.path.insert(0, os.path.abspath("."))

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_core.messages import HumanMessage

from langgraph.graph import (
    StateGraph,
    END,
    MessagesState
)

from langgraph.prebuilt import ToolNode

from src.tools import (
    ALL_FINANCIAL_TOOLS as ALL_TOOLS
)

from src.agents.memory import (
    get_recent_context
)

load_dotenv()


def create_agent():

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.1
    )

    return llm.bind_tools(ALL_TOOLS)


def agent_node(state):

    llm = create_agent()

    response = llm.invoke(state["messages"])

    return {
        "messages": [response]
    }


def should_use_tools(state):

    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return END


def build_graph():

    tool_node = ToolNode(ALL_TOOLS)

    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", agent_node)

    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_use_tools
    )

    workflow.add_edge("tools", "agent")

    return workflow.compile()


GRAPH = build_graph()


def run_agent(question: str) -> dict:

    small_talk = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "good morning",
        "good evening"
    }

    if question.lower().strip() in small_talk:
        return {
            "question": question,
            "answer": "Hello! I'm FinSight AI. How can I help with stocks, markets, financial reports, or company research?"
        }

    context = get_recent_context(5)

    result = GRAPH.invoke({
        "messages": [
            HumanMessage(
                content=f"""
You are FinSight, an expert financial AI assistant.

You have access to specialized tools:
- Use search_uploaded_documents for questions about financial filings, 10-K, 10-Q, annual reports, or uploaded documents.
- Use stock tools (get_stock_price, get_stock_history, compare_stocks) for market data and price metrics.
- Use news tools (get_company_news, get_analyst_recommendations) for company headlines and analyst ratings.
- Use SQL tools (query_financial_database, get_revenue_breakdown) for structured queries and revenue breakdowns.

Always provide accurate, grounded answers and cite sources whenever available.

Previous conversation:
{context}

Question:
{question}
"""
            )
        ]
    })

    return {
        "question": question,
        "answer": result["messages"][-1].content,
        "messages": result["messages"]
    }
