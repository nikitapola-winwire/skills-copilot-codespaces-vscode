# app.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

load_dotenv()

app = FastAPI()
CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
AZ_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# load vectorstore
vectorstore = Chroma(persist_directory=CHROMA_DIR)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# LLM (Azure Chat)
llm = AzureChatOpenAI(deployment_name=AZ_DEPLOYMENT, temperature=0)

# Retrieval QA chain (returns sources)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="map_rerank", retriever=retriever, return_source_documents=True)

class Query(BaseModel):
    q: str

@app.post("/chat")
def chat(query: Query):
    res = qa({"query": query.q})
    answer = res.get("result") or res.get("output_text") or res
    docs = res.get("source_documents", [])
    sources = []
    for d in docs:
        sources.append({"source": d.metadata.get("source"), "page": d.metadata.get("page"), "text_snippet": d.page_content[:400]})
    return {"answer": answer, "sources": sources}
