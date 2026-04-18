import os
import sys
sys.path.insert(0, os.path.abspath("."))

import streamlit as st
from dotenv import load_dotenv
from src.rag.qa_chain import answer_question
from src.agents.graph import run_agent
from src.tools.market import get_stock_price, compare_stocks
from src.tools.sql_tool import init_database, get_revenue_breakdown, query_financial_database
from src.agents.memory import add_to_memory, get_recent_context, clear_memory

load_dotenv()
init_database()

st.set_page_config(
    page_title="FinSight AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .answer-box {
        background: #f0f7ff;
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid #b8d4f0;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">FinSight AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Agentic Financial Intelligence — RAG + Multi-Agent + Live Market Data</div>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/financial-analytics.png", width=80)
    st.title("Navigation")
    mode = st.radio(
        "Choose Mode",
        ["Document Q&A", "Live Market Data", "Financial Database", "AI Agent", "Conversation History"],
        index=0
    )
    st.divider()
    st.caption("Powered by Ollama (phi) + LangChain + ChromaDB")
    st.caption("Embeddings: BAAI/bge-small-en-v1.5 (local)")
    st.caption("LLM: phi (local, no API costs)")

if mode == "Document Q&A":
    st.header("Document Q&A")
    st.caption("Ask questions about the Infosys Annual Report 2023-24")

    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input(
            "Ask a question",
            placeholder="What was Infosys total revenue in 2024?",
            key="doc_question"
        )
    with col2:
        k_docs = st.slider("Sources", 3, 10, 5)

    example_questions = [
        "What was Infosys total revenue in FY2024?",
        "Who is the CEO of Infosys?",
        "What are the main business segments?",
        "What risks are mentioned in the report?",
        "What is Infosys strategy for AI adoption?"
    ]

    st.caption("Example questions:")
    cols = st.columns(len(example_questions))
    for i, eq in enumerate(example_questions):
        if cols[i].button(eq[:30] + "...", key=f"eq_{i}"):
            question = eq

    if question:
        with st.spinner("Searching document and generating answer..."):
            result = answer_question(question, k=k_docs)

        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        st.markdown(f"**Answer:**\n\n{result['answer']}")
        st.markdown('</div>', unsafe_allow_html=True)

        add_to_memory(question, result['answer'])

        with st.expander(f"View {result['num_sources']} source chunks"):
            for i, doc in enumerate(result['source_documents']):
                st.markdown(f"**Chunk {i+1} — Page {doc.metadata.get('page', '?')}**")
                st.text(doc.page_content[:400])
                st.divider()

elif mode == "Live Market Data":
    st.header("Live Market Data")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Single Stock")
        ticker = st.text_input("Ticker Symbol", value="INFY.NS")
        if st.button("Get Price", type="primary"):
            with st.spinner("Fetching data..."):
                result = get_stock_price.invoke(ticker)
            st.code(result)

    with col2:
        st.subheader("Compare Stocks")
        tickers = st.text_input("Tickers (comma-separated)", value="INFY.NS,TCS.NS,WIPRO.NS")
        if st.button("Compare", type="primary"):
            with st.spinner("Comparing stocks..."):
                result = compare_stocks.invoke(tickers)
            st.code(result)

    st.divider()
    st.subheader("IT Sector Watchlist")
    watchlist = ["INFY.NS", "TCS.NS", "WIPRO.NS", "HCLTECH.NS"]
    cols = st.columns(len(watchlist))

    for i, tick in enumerate(watchlist):
        with cols[i]:
            with st.spinner(f"Loading {tick}..."):
                try:
                    import yfinance as yf
                    stock = yf.Ticker(tick)
                    hist = stock.history(period="2d")
                    if not hist.empty:
                        price = hist['Close'].iloc[-1]
                        prev = hist['Close'].iloc[-2] if len(hist) > 1 else price
                        change = ((price - prev) / prev) * 100
                        st.metric(
                            label=tick,
                            value=f"₹{price:.2f}",
                            delta=f"{change:+.2f}%"
                        )
                except Exception:
                    st.metric(label=tick, value="N/A")

elif mode == "Financial Database":
    st.header("Financial Database")

    tab1, tab2 = st.tabs(["Revenue Breakdown", "SQL Query"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            company = st.selectbox("Company", ["Infosys", "TCS", "Wipro"])
        with col2:
            year = st.selectbox("Year", [2024, 2023, 2022])

        if st.button("Get Breakdown", type="primary"):
            result = get_revenue_breakdown.invoke({"company": company, "year": year})
            st.code(result)

    with tab2:
        st.caption("Query the financial database directly")
        default_query = "SELECT company, year, revenue, net_income FROM company_financials ORDER BY year DESC, revenue DESC"
        sql = st.text_area("SQL Query", value=default_query, height=100)

        if st.button("Run Query", type="primary"):
            result = query_financial_database.invoke(sql)
            st.code(result)

        st.caption("Available tables: company_financials, segment_revenue, geography_revenue")

elif mode == "AI Agent":
    st.header("AI Agent")
    st.caption("The agent reasons across document + market data + database")

    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []

    for msg in st.session_state.agent_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask the FinSight agent anything..."):
        st.session_state.agent_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agent thinking..."):
                result = run_agent(prompt)
                answer = result["answer"]
            st.write(answer)

        st.session_state.agent_messages.append({"role": "assistant", "content": answer})
        add_to_memory(prompt, answer)

elif mode == "Conversation History":
    st.header("Conversation History")

    context = get_recent_context(10)
    if context == "No previous conversation history.":
        st.info("No conversation history yet. Start asking questions in other modes.")
    else:
        st.text_area("Recent conversations", value=context, height=400)

    if st.button("Clear History", type="secondary"):
        clear_memory()
        st.success("Memory cleared.")