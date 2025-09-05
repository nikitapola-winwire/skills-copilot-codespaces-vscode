import os  
from dotenv import load_dotenv  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings  # <-- updated import
from langchain_community.vectorstores import Chroma

load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")

# --- Loader: load documents from ./data ---
loader = DirectoryLoader("./data", glob="**/*.*")
docs = loader.load()

# --- Split into chunks ---
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# --- Embeddings (Azure) ---
emb = OpenAIEmbeddings(
    model="text-embedding-3-small",
    azure_deployment="OpenAICreate-20250905124015",  # from your .env
)
# --- Vectorstore (Chroma) ---
vectorstore = Chroma.from_documents(chunks, embedding=emb, persist_directory=CHROMA_DIR)
vectorstore.persist()
print(f"Indexed {len(chunks)} chunks into Chroma at {CHROMA_DIR}")