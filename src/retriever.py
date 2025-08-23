import pandas as pd
import os
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_DIR = os.environ.get("INDEX_DIR", "data/faiss_index")

class MedicalRetriever:
    def __init__(self):
        self.csv_path = './data/medquad.csv'
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    def csv_to_docs(self):
        df = pd.read_csv(self.csv_path)
        df.dropna(inplace=True)
        loader = DataFrameLoader(df, page_content_column='answer')
        docs = loader.load()
        return docs

    def vector_store_as_retriver(self, chunks):
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        vector_store.save_local(INDEX_DIR)
        return vector_store #.as_retriever(search_type="similarity", search_kwargs={"k":4})

    def vector_store(self):
        if not os.path.isdir(INDEX_DIR):
            chunks = self.csv_to_docs()
            vector_store = FAISS.from_documents(chunks, self.embeddings)
            vector_store.save_local(INDEX_DIR)
            return vector_store
        
        return FAISS.load_local('data/faiss_index', embeddings=self.embeddings, allow_dangerous_deserialization=True)

    def retriever(self):
        vector_store = self.vector_store()
        return vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})
