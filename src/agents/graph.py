import os
import sys
sys.path.insert(0, os.path.abspath("."))

from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from src.agents.tools import ALL_TOOLS

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "The conversation messages"]
    question: str
    final_answer: str

def create_agent():
    llm = OllamaLLM(model="phi", temperature=0.1)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    return llm_with_tools

def agent_node(state: AgentState) -> AgentState:
    llm = OllamaLLM(model="phi", temperature=0.1)
    
    question = state["question"]
    messages = state["messages"]
    
    system_prompt = """You are FinSight, an expert financial analyst AI.
You have access to these tools:
1. search_financial_document — search the Infosys Annual Report
2. get_stock_price — get live stock prices  
3. calculate_financial_metric — do financial calculations
4. get_company_summary — get company overview

Always use search_financial_document first for questions about the annual report.
Use get_stock_price for current market data.
Be specific, cite page numbers, and show calculations."""

    full_prompt = f"{system_prompt}\n\nQuestion: {question}"
    response = llm.invoke(full_prompt)
    
    new_messages = messages + [
        HumanMessage(content=question),
        AIMessage(content=response)
    ]
    
    return {
        "messages": new_messages,
        "question": question,
        "final_answer": response
    }

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)
    graph = workflow.compile()
    return graph

def run_agent(question: str) -> dict:
    graph = build_graph()
    
    initial_state = {
        "messages": [],
        "question": question,
        "final_answer": ""
    }
    
    print(f"Running FinSight agent for: {question}")
    result = graph.invoke(initial_state)
    
    return {
        "question": question,
        "answer": result["final_answer"],
        "messages": result["messages"]
    }

if __name__ == "__main__":
    questions = [
        "What was Infosys revenue growth from 2023 to 2024?",
        "What is the current stock price of Infosys?",
        "What are the main risks mentioned in the Infosys annual report?"
    ]
    
    for q in questions:
        print(f"\n{'='*60}")
        result = run_agent(q)
        print(f"Q: {result['question']}")
        print(f"A: {result['answer']}")