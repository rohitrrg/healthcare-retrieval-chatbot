from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


class Chain:
    def __init__(self):
        print('Building Chain !!')
        
    
    def format_docs(self, retrieved_docs):
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        return context_text
    
    def build_chain(self, retriever, llm):

        parser = StrOutputParser()

        prompt = PromptTemplate.from_template(
                "You are a careful clinical information assistant. Answer ONLY from the provided context. "
                "Be concise and neutral. If the answer is not present, say you don't know and suggest consulting a clinician. "
                #"Always include numbered citations like [1], [2] referring to the sources listed at the end.\n\n"
                "Question:\n{question}\n\n"
                "Context:\n{context}\n\n"
                "Instructions:\n"
                "- Start with a 2–5 sentence direct answer.\n"
                "- Then provide bullet-point details.\n"
                "- Every factual statement must be supported by citations [#].\n"
                #"- If sources conflict or are outdated, say so.\n"
                "- If insufficient evidence, say 'insufficient evidence'.\n\n"
            )
        
        parallel_chain = RunnableParallel({
           'context': retriever | RunnableLambda(self.format_docs) ,
           'question': RunnablePassthrough()
           })
        
        main_chain = parallel_chain | prompt | llm | parser

        return main_chain