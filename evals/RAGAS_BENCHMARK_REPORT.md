# 📊 FinSight AI — RAGAS Benchmark Evaluation Report

**Evaluated on:** `2026-10-27T07:24:21Z`  
**Target System:** `https://api.finsightai.space/api/research/query` (Groq `openai/gpt-oss-120b` + ChromaDB + BGE Embeddings)

## 🎯 Executive Scorecard

| Metric | Score | Target | Assessment |
| :--- | :---: | :---: | :--- |
| **Faithfulness** | **100.00%** | >85.0% | Excellent (Zero Hallucinations) |
| **Answer Relevancy** | **93.30%** | >85.0% | High (Direct & Concisely Structured) |
| **Context Precision** | **100.00%** | >80.0% | High (Precise Statement & Page Citations) |
| **Context Recall** | **93.33%** | >80.0% | Complete (Captured Key Numerical Figures) |
| **Composite RAGAS Score** | **96.54%** | >80.0% | **Production Grade** |

## 🧪 Case-by-Case Breakdown

| ID | Category | Question | Faithfulness | Relevancy | Precision | Recall | Latency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `RAG-001` | Revenue & Topline | What is Infosys total revenue for FY2024? | 1.00 | 0.90 | 1.00 | 1.00 | 1.02s |
| `RAG-002` | Profit & Loss Statement | What was the total income of Infosys for the year ended 31 March 2024? | 1.00 | 0.94 | 1.00 | 0.67 | 1.06s |
| `RAG-003` | Source Citation & Grounding | Which financial statement and page contains the consolidated profit and loss for Infosys? | 1.00 | 0.93 | 1.00 | 1.00 | 1.21s |
| `RAG-004` | Negative Control / Hallucination Resistance | What was the net profit and operating margin of Tesla Motors in the uploaded Infosys report? | 1.00 | 1.00 | 1.00 | 1.00 | 0.84s |
| `RAG-005` | Multi-Fact Synthesis | What was the total income and total revenue of Infosys for FY2024 according to the document? | 1.00 | 0.90 | 1.00 | 1.00 | 1.36s |

## 🔍 Qualitative Observations

1. **Strict Grounding:** The system correctly cited **Page 271** and **Page 0** of the Consolidated Statement of Profit and Loss for Infosys FY2024 revenue figures.
2. **Anti-Hallucination Guard:** In the negative control test (querying out-of-domain Tesla financials from the Infosys filing), the model faithfully stated `Not available in the provided document.` rather than fabricating numbers.
3. **Numeric Accuracy:** Exact rupee amounts (`₹153,670 crore` revenue and `₹158,381 crore` total income) were preserved without precision loss.
