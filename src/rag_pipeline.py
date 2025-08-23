import os
import re
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

load_dotenv()

RED_FLAG_PATTERNS = [
    r"\b(chest pain|shortness of breath|difficulty breathing|severe headache|loss of consciousness|stroke|fainting|suicidal|self[- ]harm)\b",
    r"\b(bleeding that won't stop|seizure|anaphylaxis|severe allergic)\b",
]

DISCLAIMER = (
    "This assistant provides educational information only and does not substitute for professional medical advice. "
    "For emergencies, seek immediate local medical care."
)

def safety_gate(user_query: str) -> bool:
    q = user_query.lower()
    return any(re.search(p, q) for p in RED_FLAG_PATTERNS)

def load_vectorstore(index_dir: str):
    return FAISS.load_local(index_dir, allow_dangerous_deserialization=True)

def get_llm():
    provider = os.getenv("MODEL_PROVIDER", "openai").lower()
    if provider == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=0.1)
    repo_id = os.getenv("HF_LLM_MODEL", "HuggingFaceH4/zephyr-7b-beta")
    return HuggingFaceEndpoint(repo_id=repo_id, temperature=0.1, max_new_tokens=512)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a careful clinical information assistant. Answer ONLY from the provided context. "
         "Be concise and neutral. If the answer is not present, say you don't know and suggest consulting a clinician. "
         "Always include numbered citations like [1], [2] referring to the sources listed at the end. "
         + DISCLAIMER),
        ("human",
         "Question: {question}\n\n"
         "Context:\n{context}\n\n"
         "Instructions:\n"
         "- Start with a 2–5 sentence direct answer.\n"
         "- Then provide bullet-point details.\n"
         "- Every factual statement must be supported by citations [#].\n"
         "- If sources conflict or are outdated, say so.\n"
         "- If insufficient evidence, say 'insufficient evidence'.")
    ]
)

def _format_context(docs: List[Document]) -> Tuple[str, List[str]]:
    lines = []
    sources = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata.copy()
        title = meta.get("source", "CSV")
        tag = f"[{i}] {title} — row:{meta.get('row', meta.get('line', 'n/a'))}"
        snippet = d.page_content.replace("\n", " ").strip()
        lines.append(f"{tag}\n{snippet}\n")
        sources.append(tag)
    return "\n".join(lines), sources

def answer_question(index_dir: str, question: str, k: int = 8) -> Dict:
    if safety_gate(question):
        return {
            "answer": (
                f"{DISCLAIMER}\n\n"
                "Your query mentions potentially urgent symptoms. I can't provide medical advice. "
                "Please seek immediate care from a healthcare professional or your local emergency services."
            ),
            "sources": [],
            "refused": True,
        }

    vs = load_vectorstore(index_dir)
    docs_scores = vs.similarity_search_with_score(question, k=k)
    docs = [d for d, _ in docs_scores]
    if not docs:
        return {"answer": "I couldn't find relevant information in the knowledge base.", "sources": [], "refused": False}

    context, source_tags = _format_context(docs)
    llm = get_llm()
    chain = PROMPT | llm
    resp = chain.invoke({"question": question, "context": context})
    text = resp.content if hasattr(resp, "content") else str(resp)

    return {"answer": text, "sources": source_tags, "refused": False}
