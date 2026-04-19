import os
import sys
sys.path.insert(0, os.path.abspath("."))

from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode

from src.tools import ALL_FINANCIAL_TOOLS as ALL_TOOLS
from src.agents.memory import get_recent_context

load_dotenv()


def create_agent():
    llm = OllamaLLM(model="phi", temperature=0.1)
    return llm.bind_tools(ALL_TOOLS)


def should_use_tools(state):
    """Check if the last message contains tool calls safely"""
    last_message = state["messages"][-1]
    return hasattr(last_message, "tool_calls") and last_message.tool_calls


def build_graph():
    llm = create_agent()
    tool_node = ToolNode(ALL_TOOLS)

    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", llm)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        lambda state: "tools" if should_use_tools(state) else END,
    )

    workflow.add_edge("tools", "agent")

    return workflow.compile()


def run_agent(question: str) -> dict:
    graph = build_graph()

    context = get_recent_context(5)

    result = graph.invoke({
        "messages": [
            HumanMessage(content=f"""
You are FinSight, an expert financial AI.

You can use tools:
- Use document search for annual report questions
- Use stock tools for market data
- Use SQL tools for structured queries

Previous conversation:
{context}

Question: {question}
""")
        ]
    })

    return {
        "question": question,
        "answer": result["messages"][-1].content,
        "messages": result["messages"]
    }