from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
import streamlit as st

def getDB():
    embeddings = (
        OllamaEmbeddings(model="gemma2:2b")
    )
    db = FAISS.load_local("VectorDB/index",
                          embeddings,
                          allow_dangerous_deserialization=True)
    
    return db

#---- creating prompt-----

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant. If the answer cannot be found in the context, say "The exact answer is not defined, However, the relevant context is as follows:" and answer the the following question based on the below context.
    <context>
    {context}
    </context>
    
    """
    
)

llm = Ollama(model="gemma2:2b")
doc_chain = create_stuff_documents_chain(llm,prompt)

db = getDB()

retriver = db.as_retriever()
# retriever = db.as_retriever(search_kwargs={"k": 3, "score_threshold": 0.7})

ret_chin = create_retrieval_chain(retriver,doc_chain)

# streamlit code

st.title("Basic Model Testing")
input_text = st.text_input("Ask a question")

if input_text:
    res = ret_chin.invoke({"input": input_text})
    st.write(res["answer"])
