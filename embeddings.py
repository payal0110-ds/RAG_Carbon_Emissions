from dataIngestion import PDF_Loader

from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS

def create_embeddings():
    docs=PDF_Loader('Data/Impact on Indian Cities.pdf')
    embeddings=(
        OllamaEmbeddings(model="gemma2:2b")
    )
    vectorStore=FAISS.from_documents(docs,embeddings)
    vectorStore.save_local("VectorDB/index")


create_embeddings()
