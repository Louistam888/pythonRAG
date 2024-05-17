from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loaders = [PyPDFLoader("./report.pdf")]

docs = []

for file in loaders:
    docs.extend(file.load())

# split content into chunks and generate vectorized chunks for each aka embeddings

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chuck_overlap = 100)
docs = text_splitter.split_document(docs)
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={"device" : "cpu"})

vectorstore = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db_nccn" )

# test of how many embeddings we have in our vectorstore
print(vectorstore._collection.count())


