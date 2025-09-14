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
 
loader2 = PyPDFLoader(
    "./datasets/BNF-78-1.pdf",
    mode="page",
)
docs2 = loader2.load()
docs_list2 = docs2

text_splitter2 = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits2 = text_splitter2.split_documents(docs_list2)
print(f"Length of documents is {len(doc_splits2)}")

print("Beginning vector embedding process")
embeddings2 = MistralAIEmbeddings()
vector_store2 = TiDBVectorStore.from_documents(
    documents=doc_splits2,
    embedding=embeddings2,
    table_name="british-formulary",
    connection_string=tidb_connection_string,
    distance_strategy="cosine",
)
print("Finished vector embedding process")
