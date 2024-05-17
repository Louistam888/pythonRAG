from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load the PDF document
loaders = [PyPDFLoader("./report.pdf")]

docs = []

for loader in loaders:
    docs.extend(loader.load())

# Split content into chunks and generate vectorized chunks for each aka embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(docs)  

# Initialize the HuggingFaceEmbeddings with the model
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

# Initialize the Chroma vector store
vectorstore = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db_nccn")

# Test how many embeddings we have in our vectorstore
print(vectorstore._collection.count())
