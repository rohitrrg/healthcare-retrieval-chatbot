import streamlit as st
from llm import LLM
from retriever import MedicalRetriever
from chain import Chain

K_RETRIEVE = 4
SIMILARITY_THRESHOLD = 0.6   # lower distance is better; we will check scores where available
MAX_NEW_TOKENS = 384
TEMPERATURE = 0.2
TOP_P = 0.9

INDEX_DIR = "data/faiss_index"


st.set_page_config(page_title="Healthcare RAG System", page_icon="🩺", layout="centered")
st.title("🩺  Healthcare Chatbot with Medical Document Retrieval")

@st.cache_resource(show_spinner=True, )
def load_and_index():
    return MedicalRetriever().retriever()

@st.cache_resource(show_spinner=True, )
def load_llm():
    llm = LLM()
    return llm

vector_store = load_and_index()
llm = load_llm()
chain = Chain().build_chain(vector_store, llm)

with st.form("qa_form"):
    user_q = st.text_area("Ask a medical question (non-emergency):", value="What are early symptoms of dehydration?", height=120, placeholder="e.g., What are early symptoms of dehydration?")
    submitted = st.form_submit_button("Ask")

if submitted and user_q.strip():
    with st.spinner("Thinking..."):
        #ans, used_docs = user_q.strip()
        ans = chain.invoke(user_q.strip())
    st.markdown("### Answer")
    st.write(ans)

    # if used_docs:
    #     st.markdown("---")
    #     st.markdown("##### Sources (retrieved)")
    #     for i, d in enumerate(used_docs, 1):
    #         q = d.metadata.get("question", "source")
    #         st.markdown(f"- **[{i}]** _{q}_")
else:
    st.info("Tip: This app works best for questions related to the provided CSV knowledge base.", icon="💡")

st.markdown("---")
st.caption("If you think you may have a medical emergency, call your local emergency number immediately.")
