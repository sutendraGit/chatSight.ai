# app.py
import os
import json
import pickle
import time
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
    max_tokens=65536,  # Increased for complex HTML/JSON responses
    request_timeout=180,  # Increased timeout for longer responses
)

embeddings = OpenAIEmbeddings(
    model=EMBED_MODEL,
    openai_api_key=OPENAI_API_KEY,
    base_url=os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1"),
    chunk_size=100,  # Process embeddings in smaller batches
    max_retries=3,   # Reduce retries for faster processing
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

def fetch_transcripts(interaction_ids: List[str], items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fetch transcripts for individual interaction IDs using the enhanced transcript API
    """
    transcripts_map = {}
    
    # Create a mapping of interaction_id -> vsId for visitor ID lookup
    interaction_to_vsid = {}
    for item in items:
        if isinstance(item, dict) and "interactionId" in item and "vsId" in item:
            interaction_to_vsid[item["interactionId"]] = item["vsId"]
    
    for interaction_id in interaction_ids:
        try:
            # Get visitor ID from the item data
            time.sleep(0.1)  # Sleep for 100ms between API calls
            visitor_id = interaction_to_vsid.get(interaction_id, "")  # fallback to default if not found
            
            # Prepare headers for the transcript API call
            headers = {
                'Accept': '*/*',
                'Accept-Language': 'en-US,en;q=0.9,en-IN;q=0.8',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Content-Type': 'application/x-www-form-urlencoded',
                'Pragma': 'no-cache',
                'Referer': f'https://consoleusw1.portal.assist.staging.247-inc.net/en/console?_skey={ASSIST_SKEY}&locale=en_US&status=default',
                'Sec-Fetch-Dest': 'empty',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Site': 'same-origin',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36 Edg/140.0.0.0',
                'X-Requested-With': 'XMLHttpRequest',
                'sec-ch-ua': '"Chromium";v="140", "Not=A?Brand";v="24", "Microsoft Edge";v="140"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"'
            }
            
            cookies = {
                'JSESSIONID': 'node0ep66oynbz8wc1vrra0l9odifk1738.node0',
                '_tokenId': ASSIST_TOKEN,
                '_clientId': 'nemo-client-pedemo',
                '_domainId': 'pedemo',
                '_locale': 'en_US',
                '_userAuthType': 'local-auth-user',
                '_dc': 'consoleusw1',
                '_jsVersion': '3.30.0',
                '__asId': ASSIST_AGENT_SESSION_ID,
                'SERVER': 'usw1_1',
                'BAYEUX_BROWSER': '1fol0vdixnsrl1ic',
                'CSRF-TOKEN': '16975072348901221187012439763872'
            }
            
            # Build the URL with interaction ID in the path and visitor ID from item data
            url = f'https://consoleusw1.portal.assist.staging.247-inc.net/en/enhanced_interaction_transcript/rest/transcript/{interaction_id}/visitor/{visitor_id}'
            
            # Query parameters
            params = {
                'context': '',
                '_skey': ASSIST_SKEY,
                'session': ASSIST_AGENT_SESSION_ID,
                'diagSeq': '417',
                'clientId': 'nemo-client-pedemo',
                'locale': 'en-us',
                'dateRangeStart': '1',
                'dateRangeEnd': '1758208078000',
                'pageSize': '1',
                'pageNumber': '0',
                'isHistoryDataRequested': 'true',
                'queryCriteria': f'{{"orderBy":[{{"sortingOrder":"DESC","fieldName":"startTime"}}],"filters":[{{"fieldName":"vsId","operation":"EQ","value":"{visitor_id}"}},{{"fieldName":"InteractionId","operation":"EQ","value":"{interaction_id}"}}]}}'
            }
            
            response = requests.get(
                url,
                headers=headers,
                cookies=cookies,
                params=params
            )
            response.raise_for_status()
            
            transcript_response = response.json()
            print(f"DEBUG: Transcript API response for {interaction_id}: {list(transcript_response.keys()) if isinstance(transcript_response, dict) else 'Not a dict'}")
            
            # Extract transcript data from response
            if isinstance(transcript_response, dict):
                # Look for transcript data in various possible locations
                transcript_data = transcript_response.get('data', transcript_response.get('items', transcript_response.get('messages', transcript_response)))
                transcripts_map[interaction_id] = transcript_data
            else:
                transcripts_map[interaction_id] = transcript_response
                
        except Exception as e:
            print(f"ERROR: Failed to fetch transcript for {interaction_id}: {e}")
            transcripts_map[interaction_id] = None
    
    print(f"DEBUG: Extracted transcripts for {len([k for k, v in transcripts_map.items() if v is not None])} out of {len(interaction_ids)} interactions")
    return transcripts_map

def fetch_realtime_metrics() -> Dict[str, Any]:
    """
    Fetch real-time monitoring metrics from the 247-inc.net API
    Returns real-time metrics data for current queue status, agent status, etc.
    """
    try:
        headers = {
            'Authorization': 'J4XAL5tESUrhhoYBetTuZiysClFUGjDZthkKHuRu8WKFV8oJ'
        }
        
        url = 'https://staging.api.cloud.247-inc.net/RealTimeMonitoring/v1/clients/nemo-client-pedemo/accounts/pedemo-account-default/metrics'
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        rtm_data = response.json()
        print(f"DEBUG: Real-time metrics API response keys: {list(rtm_data.keys()) if isinstance(rtm_data, dict) else 'Not a dict'}")
        
        return rtm_data
        
    except Exception as e:
        print(f"ERROR: Failed to fetch real-time metrics: {e}")
        return {}

def create_rtm_documents(rtm_data: Dict[str, Any], table_name: str = "rtm_metrics") -> List[Document]:
    """
    Create Document objects from real-time monitoring data
    Returns a list of Document objects with rtm_ prefixed metadata for real-time info
    """
    global metadata_store
    docs = []
    
    if not rtm_data:
        return docs
    
    try:
        # Handle different possible structures of RTM data
        if isinstance(rtm_data, dict):
            # If rtm_data has queues, agents, or other structured data
            for category, items in rtm_data.items():
                if isinstance(items, list):
                    # Handle arrays of items (queues, agents, etc.)
                    for i, item in enumerate(items):
                        if isinstance(item, dict):
                            # Create unique doc_id with rtm prefix
                            doc_id = f"rtm:{table_name}:{category}:{i}"
                            
                            # Add RTM-specific metadata
                            if "queueId" in item:
                                doc_id = f"rtm:{table_name}:queue:{item['queueId']}"
                            elif "agentId" in item:
                                doc_id = f"rtm:{table_name}:agent:{item['agentId']}"
                            elif "id" in item:
                                doc_id = f"rtm:{table_name}:{category}:{item['id']}"
                            
                            # Add timestamp for freshness
                            item["rtm_timestamp"] = time.time()
                            item["rtm_category"] = category
                            
                            text = json.dumps(item, indent=2)
                            meta = {"doc_id": doc_id, "rtm_category": category}
                            doc = Document(page_content=text, metadata=meta)
                            docs.append(doc)
                            metadata_store[doc_id] = meta
                            
                elif isinstance(items, dict):
                    # Handle single objects
                    doc_id = f"rtm:{table_name}:{category}"
                    items["rtm_timestamp"] = time.time()
                    items["rtm_category"] = category
                    
                    text = json.dumps(items, indent=2)
                    meta = {"doc_id": doc_id, "rtm_category": category}
                    doc = Document(page_content=text, metadata=meta)
                    docs.append(doc)
                    metadata_store[doc_id] = meta
            
            # If the whole response should be treated as one document
            if not docs:
                doc_id = f"rtm:{table_name}:overall_metrics"
                rtm_data["rtm_timestamp"] = time.time()
                
                text = json.dumps(rtm_data, indent=2)
                meta = {"doc_id": doc_id, "rtm_category": "overall"}
                doc = Document(page_content=text, metadata=meta)
                docs.append(doc)
                metadata_store[doc_id] = meta
        
        print(f"DEBUG: Created {len(docs)} RTM documents")
        for doc in docs[:3]:  # Show first 3 for brevity
            print(f"RTM doc_id: {doc.metadata['doc_id']}")
            print(f"RTM content (first 200 chars): {doc.page_content[:200]}")
            
    except Exception as e:
        print(f"ERROR: Failed to create RTM documents: {e}")
    
    return docs

def create_documents_from_api_response(table_name: str = "interactions_table") -> List[Document]:
    """
    Fetch interactions from API and create Document objects with transcripts
    Returns a list of Document objects ready for FAISS indexing
    """
    global metadata_store
    
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
    
    data = 'queryCriteria=%7B%22orderBy%22%3A%5B%7B%22sortingOrder%22%3A%22DESC%22%2C%22fieldName%22%3A%22endTime%22%7D%5D%2C%22filters%22%3A%5B%7B%22fieldName%22%3A%22interactionEndState%22%2C%22operation%22%3A%22EQ%22%2C%22value%22%3A%22DISPOSED%22%7D%5D%7D&queryMetadata=%7B%7D&timeType=endTime&dateRangeStart=1757442600000&dateRangeEnd=1758208078000&pageSize=10&pageNumber=0&isFilteringDone=false&navigationDirection=FIRST&timeBasedMarkerObj=null&isTranscriptRequested=false&mode=search&searchType=team&locale=en-us'
    
    response = requests.post(
        f'https://consoleusw1.portal.assist.staging.247-inc.net/en/enhanced_interaction_history/rest/team/pedemo-account-default-team-default-allteams?_skey={ASSIST_SKEY}&session={ASSIST_AGENT_SESSION_ID}&diagSeq=147&clientId=nemo-client-pedemo',
        headers=headers,
        cookies=cookies,
        data=data
    )
    response.raise_for_status()
    
    # Try to parse as JSON
    data = response.json()
    data = data['data']
    
    if not isinstance(data, dict) or "items" not in data:
        raise ValueError("API response does not contain expected 'items' structure")
    
    items = data["items"]
    
    # Extract interaction IDs for transcript fetching
    interaction_ids = []
    for item in items:
        if isinstance(item, dict) and "interactionId" in item:
            interaction_ids.append(item["interactionId"])
    
    # Fetch transcripts for all interactions
    transcripts_data = {}
    if interaction_ids:
        try:
            transcripts_data = fetch_transcripts(interaction_ids, items)
            print(f"DEBUG: Fetched transcripts for {len(transcripts_data)} interactions")
        except Exception as e:
            print(f"WARNING: Failed to fetch transcripts: {e}")
    
    # Create documents for each item using interactionId as doc_id
    docs = []
    for item in items:
        if isinstance(item, dict) and "interactionId" in item:
            # Add transcript to the item if available
            interaction_id = item["interactionId"]
            if interaction_id in transcripts_data:
                item["transcript"] = transcripts_data[interaction_id]
            
            doc_id = f"table:{table_name}:interaction:{item['interactionId']}"
            text = json.dumps(item, indent=2)
            meta = {"doc_id": doc_id}
            doc = Document(page_content=text, metadata=meta)
            docs.append(doc)
            metadata_store[doc_id] = meta
    
    # Fetch and add real-time monitoring data
    print("DEBUG: Fetching real-time metrics...")
    rtm_data = fetch_realtime_metrics()
    if rtm_data:
        rtm_docs = create_rtm_documents(rtm_data, "rtm_metrics")
        docs.extend(rtm_docs)
        print(f"DEBUG: Added {len(rtm_docs)} real-time monitoring documents")
    
    # Logging for ingestion
    print(f"Ingested {len(docs)} total documents (interactions + RTM):")
    for doc in docs[:3]:  # Show first 3 for brevity
        print(f"doc_id: {doc.metadata['doc_id']}")
        print(f"content (first 200 chars): {doc.page_content[:200]}")
    
    return docs

def ingest_url(url: str, table_name: str = "interactions_table"):
    global metadata_store, faiss_index
    try:
        # Get documents from API
        docs = create_documents_from_api_response(table_name)
        
        # Add to FAISS
        if docs:
            print(f"DEBUG: Starting FAISS indexing for {len(docs)} documents...")
            start_time = time.time()
            
            # Process documents in smaller batches to improve performance
            batch_size = 5  # Process 5 documents at a time
            
            if os.path.exists(FAISS_INDEX_PATH):
                print("DEBUG: Loading existing FAISS index...")
                faiss_index = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
                
                # Add documents in batches
                for i in range(0, len(docs), batch_size):
                    batch = docs[i:i + batch_size]
                    print(f"DEBUG: Processing batch {i//batch_size + 1}/{(len(docs) + batch_size - 1)//batch_size} ({len(batch)} documents)")
                    faiss_index.add_documents(batch)
            else:
                print("DEBUG: Creating new FAISS index...")
                faiss_index = FAISS.from_documents(docs, embeddings)
            
            end_time = time.time()
            print(f"DEBUG: FAISS indexing completed in {end_time - start_time:.2f} seconds")
            
            print("DEBUG: Saving FAISS index to disk...")
            faiss_index.save_local(FAISS_INDEX_PATH)
            with open(METADATA_STORE_PATH, "wb") as f:
                pickle.dump(metadata_store, f)
            print("DEBUG: FAISS index saved successfully")
        
        return {"ingested_count": len(docs), "table_name": table_name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch or parse URL: {e}")

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
    added_interaction_ids = set()  # Track which interaction IDs have been added
    rtm_blocks = []  # Separate blocks for real-time data
    
    for meta in metas:
        doc_id = meta["doc_id"]
        content = meta.get("page_content", "")
        
        # Handle RTM (Real-time monitoring) documents
        if doc_id.startswith("rtm:"):
            rtm_category = meta.get("rtm_category", "unknown")
            display_label = f"RTM_{rtm_category}"
            
            try:
                json_obj = json.loads(content)
                if isinstance(json_obj, dict):
                    rtm_blocks.append(f"[{display_label}] REAL-TIME DATA:\n{content}")
                else:
                    rtm_blocks.append(f"[{display_label}] REAL-TIME DATA (type: {type(json_obj).__name__}):\n{content}")
            except Exception:
                rtm_blocks.append(f"[{display_label}] REAL-TIME DATA (not JSON):\n{content[:500]}")
            continue
        
        # Handle interaction documents
        if ":interaction:" in doc_id:
            interaction_id = doc_id.split(":interaction:")[-1]
            display_label = f"Interaction_{interaction_id}"
            
            # Check if this interaction ID has already been added
            if interaction_id in added_interaction_ids:
                print(f"DEBUG: Skipping duplicate interaction ID: {interaction_id}")
                continue
            
            # Mark this interaction ID as added
            added_interaction_ids.add(interaction_id)
        else:
            display_label = doc_id
            
        # Try to parse content as JSON and show keys/values for interactions
        try:
            json_obj = json.loads(content)
            if isinstance(json_obj, dict):
                # Check if this content appears to be raw item data and skip it
                if "items" in json_obj and isinstance(json_obj["items"], list):
                    print(f"DEBUG: Skipping item data for {display_label}")
                    continue
                
                context_blocks.append(f"[{display_label}] CONTENT:\n{content}")
            else:
                context_blocks.append(f"[{display_label}]\nCONTENT TYPE: {type(json_obj).__name__}\nCONTENT:\n{content}")
        except Exception:
            context_blocks.append(f"[{display_label}]\nCONTENT (not JSON):\n{content[:500]}")

    # Combine RTM and interaction data
    all_context_blocks =  context_blocks + rtm_blocks
    context_text = "\n\n".join(all_context_blocks)
    
    # Check if question asks for current/real-time information
    is_realtime_query = any(keyword in question.lower() for keyword in ['current', 'now', 'real-time', 'realtime', 'live', 'present', 'today', 'right now'])
    
    realtime_note = ""
    if rtm_blocks and is_realtime_query:
        realtime_note = """
REAL-TIME DATA PRIORITY: The question asks for current information. Prioritize data from RTM (Real-time monitoring) sources which contain the most up-to-date metrics, queue status, agent status, and live system information."""

    prompt = f"""You are a Lead supervisor for all customer care executives in a customer care center.
You are responsible for analyzing customer interactions and support data. Use the provided interaction information to answer the question accurately.
You are eligible to do very detailed analysis and provide information in HTML, JSON or any other format as required.
no explanation needed on how the html or image was generated

IMPORTANT: Always provide a complete response. If generating HTML or code, ensure all tags are properly closed and the response is fully formed.

JSON FORMATTING RULES:
- When providing JSON responses, format them cleanly without code block markers
- Do NOT wrap JSON in ```json or ``` code blocks
- Present JSON directly as formatted text that looks clean in chat bubbles
- Use proper indentation and spacing for readability{realtime_note}

Whenever question had "render" keyword, provide HTML output only html no descriptions or explanations.
when question has "visualize" keyword, provide charts or graphs in HTML format only html no descriptions or explanations.
When question has "show" keyword, provide tabular data in HTML format only html no descriptions or explanations.
When question has "graph" keyword, provide charts or graphs in HTML format only html no descriptions or explanations.
When question has "chart" keyword, provide charts or graphs in HTML format only html no descriptions or explanations.
When question has "table" keyword, provide tabular data in HTML format only html no descriptions or explanations.
When question has "html" keyword, provide HTML output only html no descriptions or explanations.
When question has "json" keyword, provide JSON output only json no descriptions or explanations.
when question has "code" keyword, provide code output only code no descriptions or explanations.
When question has "image" keyword, provide html image output only image no descriptions or explanations.

(STRICT) When tables are displayed, ensure they are in proper HTML table format with <table>, <tr>, <td>, and <th> tags.

whenever transcripts are asked, (STRICT)provide chat bubbles with purple based theme of agent message should be on left side and customer message should be on right side and of different colors
Context:
{context_text}

Question:
{question}

Please provide a complete and detailed answer. If creating visualizations or HTML, ensure the response is fully complete with all closing tags.

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
        content = d.page_content
        
        # Truncate very long content to prevent token overflow
        # if len(content) > 3000:  # Limit individual document content
        #     content = content[:3000] + "... [TRUNCATED]"
        #     print(f"DEBUG: Truncated long content for doc {meta.get('doc_id', 'unknown')}")
        
        meta["page_content"] = content
        related_metas.append(meta)
    
    prompt = build_prompt(req.question, related_metas)
    
    # Check total prompt length and warn if too long
    if len(prompt) > 12000:  # Reserve 4k tokens for response
        print(f"WARNING: Prompt is very long ({len(prompt)} chars) - may cause token overflow")
        # Optionally reduce context here
    
    print(f"DEBUG: Prompt length: {len(prompt)} characters")
    print(f"DEBUG: Number of related documents: {len(related_metas)}")
    
    try:
        llm_resp = llm.invoke(prompt)
        print(f"DEBUG: LLM response type: {type(llm_resp)}")
        print(f"DEBUG: LLM response length: {len(str(llm_resp))} characters")
        print(f"DEBUG: LLM response preview (first 200 chars): {str(llm_resp)[:200]}")
        print(f"DEBUG: LLM response preview (last 200 chars): {str(llm_resp)[-200:]}")
        
        answer_text = llm_resp.strip() if isinstance(llm_resp, str) else str(llm_resp)
        
        # Enhanced response completeness check
        is_html_response = '<html>' in answer_text.lower() or '<table>' in answer_text.lower()
        is_json_response = answer_text.strip().startswith(('{', '['))
        
        # Check for incomplete responses based on content type
        if is_html_response:
            if not (answer_text.endswith('</html>') or answer_text.endswith('</table>') or answer_text.endswith('</div>')):
                print("WARNING: HTML response might be incomplete - missing closing tags")
        elif is_json_response:
            try:
                json.loads(answer_text)  # Validate JSON completeness
            except json.JSONDecodeError:
                print("WARNING: JSON response might be incomplete - invalid JSON structure")
        elif len(answer_text) > 1000 and not answer_text.endswith(('.', '!', '?', '>', '}', ']')):
            print("WARNING: Response might be incomplete - doesn't end with proper punctuation")
        
        # Check if response was likely truncated due to token limits
        if len(str(llm_resp)) >= 15500:  # Close to our 16k limit
            print("WARNING: Response may be truncated due to token limits - consider reducing context or increasing max_tokens")
            
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
