from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def PDF_Loader(path):
    loader=PyPDFLoader(path)
    docs=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    text=text_splitter.split_documents(docs)
    return text

# result=PDF_Loader('Data/Impact on Indian Cities.pdf')
# print(result[1])