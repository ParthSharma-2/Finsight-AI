import sys
import os
sys.path.append(os.path.abspath("."))

from src.rag.qa_chain import answer_question
from evals.eval_data import dataset

# RAGAS + Metrics
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# ✅ Use Ollama instead of OpenAI
from langchain_ollama import OllamaLLM
from ragas.llms import LangchainLLMWrapper
from langchain_huggingface import HuggingFaceEmbeddings


# -------------------------------
# STEP 1: Generate answers
# -------------------------------
answers = []
contexts = []

print("Running evaluation...\n")

for q in dataset["question"]:
    result = answer_question(q)

    # Safety fallback
    answer = result.get("answer", "Not available in document")
    docs = result.get("source_documents", [])

    answers.append(answer)
    contexts.append([doc.page_content for doc in docs] if docs else [""])


# -------------------------------
# STEP 2: Add to dataset (SAFE)
# -------------------------------

# Remove existing columns safely
cols_to_remove = [col for col in ["answer", "contexts"] if col in dataset.column_names]
if cols_to_remove:
    dataset = dataset.remove_columns(cols_to_remove)

dataset = dataset.add_column("answer", answers)
dataset = dataset.add_column("contexts", contexts)

# Rename ground_truth → ground_truths safely
if "ground_truth" in dataset.column_names:
    dataset = dataset.rename_column("ground_truth", "ground_truths")

# Ensure correct format (list of strings)
def ensure_list(x):
    if "ground_truths" not in x:
        return {"ground_truths": [""]}

    if isinstance(x["ground_truths"], list):
        return x

    return {"ground_truths": [x["ground_truths"]]}

dataset = dataset.map(ensure_list)


# -------------------------------
# STEP 3: Setup evaluator LLM (FIXED)
# -------------------------------
evaluator_llm = LangchainLLMWrapper(
    OllamaLLM(
        model="phi",
        temperature=0
    )
)


# -------------------------------
# STEP 4: Setup embeddings (LOCAL)
# -------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


# -------------------------------
# STEP 5: Run evaluation
# -------------------------------
result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision
    ],
    llm=evaluator_llm,
    embeddings=embedding_model
)


# -------------------------------
# STEP 6: Print Results
# -------------------------------
print("\n===== RAGAS RESULTS =====")
print(result)