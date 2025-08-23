# 🩺 Healthcare RAG Chatbot (CSV-based)

End-to-end Retrieval-Augmented Generation app using **LangChain + Streamlit** that
loads a two-column CSV (`question,answer`) via **CSVLoader**, builds a **FAISS** vector index,
and answers user questions grounded **only** in your CSV content — with medical safety guardrails.

> **Disclaimer**: Educational information only. Not a substitute for professional medical advice.

---

## ✨ Features
- **CSVLoader** ingestion → chunks with structure
- **FAISS** vector store (local, fast, no server needed)
- **Hybrid LLM options**:
  - **OpenAI** (`gpt-4o-mini` + `text-embedding-3-small`)
  - **Hugging Face Inference** (e.g., `HuggingFaceH4/zephyr-7b-beta`)
  - **Local embeddings** via `sentence-transformers` if you don’t want remote calls
- **Citations**: every answer requests [#]-style citations
- **Guardrails**: detects urgent symptoms and returns safety notice
- **Streamlit UI**: upload CSV, build index, ask questions

---

## 📦 Setup

1) **Install**
```bash
pip install -r requirements.txt
```

2) **Configure environment**
Copy `.env.example` → `.env` and set at least one provider.

OpenAI (recommended):
```env
MODEL_PROVIDER=openai
OPENAI_API_KEY=sk-...          # https://platform.openai.com/
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Hugging Face Inference (optional):
```env
MODEL_PROVIDER=hf
HF_TOKEN=hf-...                 # https://huggingface.co/settings/tokens
HF_LLM_MODEL=HuggingFaceH4/zephyr-7b-beta
```

3) **Run the app**
```bash
streamlit run app.py
```

4) **Use it**
- Upload your CSV (two columns: `question,answer` — exact names not required).
- Click **Build / Rebuild Index**.
- Ask questions on the **Ask a Question** panel.

---

## 🧠 How it works
- **Ingestion** (`src/ingest.py`): Loads CSV via `CSVLoader`, rewrites each row as a
  `Document` in format: `Question: ...\nAnswer: ...`. Chunks into ~800 token docs and indexes into FAISS.
- **RAG Pipeline** (`src/rag_pipeline.py`):
  - Retrieves top-K chunks with FAISS.
  - Formats sources and constructs a strict grounding prompt.
  - Calls the selected LLM (OpenAI or HF Inference) to generate an answer with [#] citations.
  - Safety gate: if the query looks emergent (e.g., “chest pain”), returns a caution message.
- **Streamlit UI** (`app.py`): Upload CSV, build index, ask questions, show citations.

---

## 🧪 Tips
- Keep your CSV answers **concise and factual** — this improves retrieval quality.
- Add more rows instead of very long answers; the index works best on smaller chunks.
- To hard-refuse low-confidence answers, you can modify the pipeline to use
  `similarity_search_with_score` and set a distance threshold.

---

## 🔐 Notes on medical safety
This demo is information-only and not clinical advice. Always include the visible
disclaimer and consider expanding guardrails for your region (triage prompts, hotline numbers, etc.).
