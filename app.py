# app.py
import os
import json
import pickle
from typing import List, Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import numpy as np
import pandas as pd

# LangChain & OpenAI
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "Gemini-2.5-flash-lite")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "./faiss_index")
METADATA_STORE_PATH = os.environ.get("METADATA_STORE_PATH", "./faiss_metadata.pkl")
ASSIST_SKEY = os.environ.get("ASSIST_SKEY", "")
ASSIST_AGENT_SESSION_ID = os.environ.get("ASSIST_AGENT_SESSION_ID", "")
ASSIST_TOKEN = os.environ.get("ASSIST_TOKEN", "")
TOP_K = int(os.environ.get("TOP_K", "5"))

if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in environment")

app = FastAPI(title="RAG + FAISS Table Retriever")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class IngestRequest(BaseModel):
    table_name: str = "interactions_table"

class QueryRequest(BaseModel):
    question: str
    table_name: str = None
    top_k: int = TOP_K

# Initialize embeddings and LLM

llm = OpenAI(
    model_name=CHAT_MODEL,
    openai_api_key=OPENAI_API_KEY,
    base_url=os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1"),
    temperature=0.0,
    max_tokens=4000,  # Set a reasonable max token limit
    request_timeout=120,  # Increase timeout to 2 minutes
)

embeddings = OpenAIEmbeddings(
    model=EMBED_MODEL,
    openai_api_key=OPENAI_API_KEY,
    base_url=os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1"),
)
# We'll keep a metadata list mapping doc_ids -> metadata to persist alongside FAISS
metadata_store: Dict[str, Dict[str, Any]] = {}

# Helper to serialize a row into readable text for embedding
def row_to_text(row: Dict[str, Any], table_name: str = "") -> str:
    # simple deterministic serialization: "col1: val1 col2: val2 ..."
    parts = []
    for k, v in row.items():
        parts.append(f"{k}:{v}")
    return f"table:{table_name} " + " ".join(parts)

# Ingest function: build Document list and add to FAISS
import requests

def ingest_url(table_name: str = "interactions_table"):
    global metadata_store, faiss_index
    try:
        # Convert curl command to requests
        headers = {
            'Accept': '*/*',
            'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Content-Type': 'application/x-www-form-urlencoded',
            'Origin': 'https://consoleusw1.portal.assist.staging.247-inc.net',
            'Pragma': 'no-cache',
            'Referer': f'https://consoleusw1.portal.assist.staging.247-inc.net/en/console?_skey={ASSIST_SKEY}&locale=en_US&status=default',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36',
            'X-Requested-With': 'XMLHttpRequest',
            'sec-ch-ua': '"Chromium";v="140", "Not=A?Brand";v="24", "Google Chrome";v="140"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"'
        }
        
        cookies = {
            'JSESSIONID': 'node0e9d6xojaxtyozvmyklj8wf9r1630.node0',
            '_tokenId': ASSIST_TOKEN,
            '_clientId': 'nemo-client-pedemo',
            '_domainId': 'pedemo',
            '_locale': 'en_US',
            '_userAuthType': 'local-auth-user',
            '_dc': 'consoleusw1',
            '_jsVersion': '3.30.0',
            '__asId': ASSIST_AGENT_SESSION_ID,
            'SERVER': 'usw1_1',
            'BAYEUX_BROWSER': '1gtmcff3fh9gr13f'
        }
        
        data = 'queryCriteria=%7B%22orderBy%22%3A%5B%7B%22sortingOrder%22%3A%22DESC%22%2C%22fieldName%22%3A%22endTime%22%7D%5D%2C%22filters%22%3A%5B%7B%22fieldName%22%3A%22interactionEndState%22%2C%22operation%22%3A%22EQ%22%2C%22value%22%3A%22DISPOSED%22%7D%5D%7D&queryMetadata=%7B%7D&timeType=endTime&dateRangeStart=1757442600000&dateRangeEnd=1758133799000&pageSize=25&pageNumber=0&isFilteringDone=false&navigationDirection=FIRST&timeBasedMarkerObj=null&isTranscriptRequested=false&mode=search&searchType=team&locale=en-us'
        
        response = requests.post(
            f'https://consoleusw1.portal.assist.staging.247-inc.net/en/enhanced_interaction_history/rest/team/pedemo-account-default-team-default-allteams?_skey={ASSIST_SKEY}&session={ASSIST_AGENT_SESSION_ID}&diagSeq=147&clientId=nemo-client-pedemo',
            headers=headers,
            cookies=cookies,
            data=data
        )
        response.raise_for_status()
        # Try to parse as JSON
        try:
            data = response.json()
            data = data['data']
            if isinstance(data, dict) and "items" in data:
                items = data["items"]
                # Create documents for each item using interactionId as doc_id
                docs = []
                for item in items:
                    if isinstance(item, dict) and "interactionId" in item:
                        doc_id = f"table:{table_name}:interaction:{item['interactionId']}"
                        text = json.dumps(item, indent=2)
                        meta = {"doc_id": doc_id}
                        doc = Document(page_content=text, metadata=meta)
                        docs.append(doc)
                        metadata_store[doc_id] = meta
                
                # Logging for ingestion
                print(f"Ingested {len(docs)} interactions:")
                for doc in docs[:3]:  # Show first 3 for brevity
                    print(f"doc_id: {doc.metadata['doc_id']}")
                    print(f"content (first 200 chars): {doc.page_content[:200]}")
                
                # Add to FAISS
                if docs:
                    if os.path.exists(FAISS_INDEX_PATH):
                        faiss_index = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
                        faiss_index.add_documents(docs)
                    else:
                        faiss_index = FAISS.from_documents(docs, embeddings)
                    
                    faiss_index.save_local(FAISS_INDEX_PATH)
                    with open(METADATA_STORE_PATH, "wb") as f:
                        pickle.dump(metadata_store, f)
                
                return {"ingested_count": len(docs), "table_name": table_name}
            else:
                text = json.dumps(data, indent=2)
        except Exception:
            text = response.text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch or parse URL: {e}")



    if os.path.exists(FAISS_INDEX_PATH):
        faiss_index = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        faiss_index.add_documents(docs)
    else:
        faiss_index = FAISS.from_documents(docs, embeddings)

    faiss_index.save_local(FAISS_INDEX_PATH)
    with open(METADATA_STORE_PATH, "wb") as f:
        pickle.dump(metadata_store, f)

    return {"ingested_count": len(docs), "table_name": table_name}

# Load persisted metadata & faiss on startup if available
faiss_index = None
if os.path.exists(METADATA_STORE_PATH) and os.path.exists(FAISS_INDEX_PATH):
    try:
        with open(METADATA_STORE_PATH, "rb") as f:
            metadata_store = pickle.load(f)
        faiss_index = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("Loaded FAISS index and metadata store from disk.")
    except Exception as e:
        print("Could not load persisted FAISS index:", e)


@app.post("/ingest")
def ingest(req: IngestRequest):
    result = ingest_url(req.table_name)
    return {"status": "ok", **result}


def build_prompt(question: str, metas: List[Dict[str, Any]]) -> str:
    context_blocks = []
    for meta in metas:
        doc_id = meta["doc_id"]
        # Extract just the interaction ID for cleaner display
        if ":interaction:" in doc_id:
            display_id = doc_id.split(":interaction:")[-1]
            display_label = f"Interaction_{display_id}"
        else:
            display_label = doc_id
            
        content = meta.get("page_content", "")
        # Try to parse content as JSON and show keys/values for interactions
        try:
            json_obj = json.loads(content)
            if isinstance(json_obj, dict):
                keys = list(json_obj.keys())
                keys_str = ", ".join(keys)
                context_blocks.append(f"[{display_label}] CONTENT:\n{content}")
            else:
                context_blocks.append(f"[{display_label}]\nCONTENT TYPE: {type(json_obj).__name__}\nCONTENT:\n{content}")
        except Exception:
            context_blocks.append(f"[{display_label}]\nCONTENT (not JSON):\n{content[:500]}")

    context_text = "\n\n".join(context_blocks)
    prompt = f"""You are a Lead supervisor for all customer care executives in a customer care center.
You are responsible for analyzing customer interactions and support data. Use the provided interaction information to answer the question accurately.
You are eligible to do very detailed analysis and provide information in HTML, JSON or any other format as required.
no explanation needed on how the html or image was generated

IMPORTANT: Always provide a complete response. If generating HTML or code, ensure all tags are properly closed and the response is fully formed.



Context Content:
{context_text}

Question:
{question}

Please provide a complete and detailed answer. If creating visualizations or HTML, ensure the response is fully complete with all closing tags.

SOURCES: [List the doc_id(s) you used]
"""
    return prompt

@app.post("/query")
def query(req: QueryRequest):
    if faiss_index is None:
        raise HTTPException(status_code=400, detail="No FAISS index loaded. Please ingest data first.")

    top_k = req.top_k or TOP_K
    docs = faiss_index.similarity_search(req.question, k=top_k)

    # Include both metadata and page_content for prompt
    related_metas = []
    for d in docs:
        meta = dict(d.metadata)
        meta["page_content"] = d.page_content
        related_metas.append(meta)
    
    prompt = build_prompt(req.question, related_metas)
    print(f"DEBUG: Prompt length: {len(prompt)} characters")
    print(f"DEBUG: Number of related documents: {len(related_metas)}")
    
    try:
        llm_resp = llm.invoke(prompt)
        print(f"DEBUG: LLM response type: {type(llm_resp)}")
        print(f"DEBUG: LLM response length: {len(str(llm_resp))} characters")
        print(f"DEBUG: LLM response preview (first 200 chars): {str(llm_resp)[:200]}")
        print(f"DEBUG: LLM response preview (last 200 chars): {str(llm_resp)[-200:]}")
        
        answer_text = llm_resp.strip() if isinstance(llm_resp, str) else str(llm_resp)
        
        # Check if response seems incomplete
        if len(answer_text) > 1000 and not answer_text.endswith(('.', '!', '?', '```')):
            print("WARNING: Response might be incomplete - doesn't end with proper punctuation")
            
    except Exception as e:
        print(f"ERROR in LLM call: {e}")
        raise HTTPException(status_code=500, detail=f"LLM processing failed: {e}")

    sources = [m["doc_id"] for m in related_metas]

    return {
        "answer": answer_text,
        "response_length": len(answer_text),
        #"sources": sources,
        #"related_urls": [m["url"] for m in related_metas],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
