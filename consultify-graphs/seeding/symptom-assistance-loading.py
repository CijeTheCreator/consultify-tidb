from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import TiDBVectorStore
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import os

from dotenv import load_dotenv

load_dotenv()
tidb_connection_string = os.getenv("TIDB_CONN_STRING")
 
loader = PyPDFLoader(
    "./datasets/Microbiology, Pharmacology, and Immunology for Pre-Clinical Students.pdf",
    mode="page",
)
docs = loader.load()
docs_list = docs

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)
print(f"Length of documents is {len(doc_splits)}")

print("Beginning vector embedding process")
embeddings = MistralAIEmbeddings()
vector_store = TiDBVectorStore.from_documents(
    documents=doc_splits,
    embedding=embeddings,
    table_name="microbiology_pharmacology_immunology_textbook",
    connection_string=tidb_connection_string,
    distance_strategy="cosine",
)
print("Finished vector embedding process")
