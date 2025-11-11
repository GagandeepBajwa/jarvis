# main.py
import base64
import json
import logging
import os
import time
import hashlib
from math import exp
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from google.auth.exceptions import DefaultCredentialsError, RefreshError
from google.api_core.exceptions import PermissionDenied, Forbidden, Unauthenticated, NotFound

from fastapi import FastAPI, Request, Response, HTTPException, Body, Header, status
from pydantic import BaseModel, Field
from google.cloud import bigquery
from google.cloud import aiplatform
from google.cloud import aiplatform_v1

# ----- Config via env -----
PROJECT_ID = os.getenv("PROJECT_ID", "")
LOCATION = os.getenv("LOCATION", "")
BQ_DATASET_RAW = os.getenv("BQ_DATASET_RAW", "")
BQ_DATASET_DERIVED = os.getenv("BQ_DATASET_DERIVED", "")
BQ_DATASET_ACTIONS = os.getenv("BQ_DATASET_ACTIONS", "")

# Embeddings / Vector Search
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")  # or text-embedding-005
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "3072"))
VERTEX_INDEX_ENDPOINT = os.getenv("VERTEX_INDEX_ENDPOINT", "")  # full resource name
VERTEX_INDEX_ID = os.getenv("VERTEX_INDEX_ID", "")              # deployed index id on endpoint
VERTEX_INDEX_NAME = os.getenv("VERTEX_INDEX_NAME", "")   # index resource name
VERTEX_API_ENDPOINT = os.getenv("VERTEX_API_ENDPOINT", "")

# Optional basic protection for Pub/Sub & Chat
VERIFY_PUBSUB_HEADER = os.getenv("VERIFY_PUBSUB_HEADER", "false").lower() == "true"
CHAT_BOT_TOKEN = os.getenv("CHAT_BOT_TOKEN", "")

# ----- Clients -----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis")

# ---------- Lazy, auth-safe Google clients ----------
_bq_client: Optional[bigquery.Client] = None
_aiplatform = None         # module handle after import
_embedding_model = None    # TextEmbeddingModel instance
_index_ep = None           # MatchingEngineIndexEndpoint instance (bound if configured)
_index = None              # Index instance (not used directly here)

def _resolution_hint(prefix: str, ex: Exception) -> str:
    if isinstance(ex, DefaultCredentialsError):
        return (f"{prefix}: No Application Default Credentials. "
                "Cloud Run: attach a service account with required roles. "
                "Local: 'gcloud auth application-default login' or set GOOGLE_APPLICATION_CREDENTIALS.")
    if isinstance(ex, (PermissionDenied, Forbidden)):
        return (f"{prefix}: Permission denied. "
                "Grant the Cloud Run service account the needed roles "
                "(BigQuery: roles/bigquery.jobUser + dataset perms; "
                "Vertex: roles/aiplatform.user and Matching Engine perms).")
    if isinstance(ex, Unauthenticated):
        return f"{prefix}: Unauthenticated—provide ADC or refresh credentials."
    if isinstance(ex, RefreshError):
        return f"{prefix}: Credential refresh failed—re-authenticate or rotate key."
    if isinstance(ex, NotFound):
        return f"{prefix}: Resource not found—verify dataset/table names and full Vertex Index Endpoint in the right project/region."
    return f"{prefix}: {type(ex).__name__}: {ex}"

def get_bq() -> Optional[bigquery.Client]:
    global _bq_client
    if _bq_client is not None:
        return _bq_client
    try:
        _bq_client = bigquery.Client(project=PROJECT_ID)
        logger.info("BigQuery client initialized for project=%s", PROJECT_ID)
        return _bq_client
    except Exception as ex:
        logger.warning(_resolution_hint("BigQuery init failed", ex))
        _bq_client = None
        return None

def init_vertex_clients() -> None:
    global _aiplatform, _embedding_model, _index_ep, _index
    if _aiplatform is not None:
        return
    # aiplatform
    try:
        from google.cloud import aiplatform as _aip
        _aip.init(project=PROJECT_ID, location=LOCATION)
        _aiplatform = _aip
        logger.info("Vertex AI initialized for project=%s location=%s", PROJECT_ID, LOCATION)
    except Exception as ex:
        logger.warning(_resolution_hint("Vertex AI init failed", ex))
        _aiplatform = None
    # embeddings
    try:
        import vertexai
        from vertexai.preview.language_models import TextEmbeddingModel
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        _embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
        logger.info("Embedding model ready: %s", EMBEDDING_MODEL)
    except Exception as ex:
        logger.warning(ex)
        logger.warning(_resolution_hint("Vertex embeddings init failed", ex))
        _embedding_model = None
    # Matching Engine endpoint
    if _aiplatform and VERTEX_INDEX_ENDPOINT:
        try:
            _index_ep = _aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name=VERTEX_INDEX_ENDPOINT
            )
            logger.info("Vector Search endpoint bound: %s", VERTEX_INDEX_ENDPOINT)
        except Exception as ex:
            logger.warning(_resolution_hint("Vector Search endpoint bind failed", ex))
            _index_ep = None
    # Matching Engine index (writes)
    if _aiplatform and VERTEX_INDEX_NAME:
        try:
            _index = _aiplatform.MatchingEngineIndex(index_name=VERTEX_INDEX_NAME)
            logger.info("Vector Search index bound: %s", VERTEX_INDEX_NAME)
        except Exception as ex:
            logger.warning(_resolution_hint("Vector Search index bind failed", ex))

try:
    init_vertex_clients()
except Exception as _ex:
    logger.warning("Non-fatal Vertex init exception: %s", _ex)

# ---------- Gemini LLM (for enrichment) ----------
def _init_gemini_model():
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        return GenerativeModel("gemini-2.5-flash")
    except Exception as ex:
        logger.warning("LLM init failed; continuing without enrichment: %s", ex)
        return None

_gemini = _init_gemini_model()

# ---------- FastAPI ----------
app = FastAPI(title="Personal Jarvis (Workspace edition)", version="0.1.0")

# ---------- Models ----------
class PubSubEnvelope(BaseModel):
    message: Dict[str, Any]
    subscription: Optional[str] = None

class IngestRequest(BaseModel):
    id: Optional[str] = None
    source: Optional[str] = "workspace"
    owner: Optional[str] = None
    pointer: Optional[str] = None
    text: Optional[str] = None
    created: Optional[str] = None
    labels: Optional[List[str]] = None

class ChatSpace(BaseModel):
    name: Optional[str] = None

class ChatMessage(BaseModel):
    text: Optional[str] = None

class ChatEvent(BaseModel):
    type: Optional[str] = None
    token: Optional[str] = None
    message: Optional[ChatMessage] = None
    space: Optional[ChatSpace] = None
    user: Optional[Dict[str, Any]] = None

class SearchRequest(BaseModel):
    q: str = Field(..., description="User query to search")
    owner: Optional[str] = Field(None, description="Owner email or ID to scope results")
    k: int = Field(3, ge=1, le=10, description="How many items to return (1-10)")
    labels: Optional[List[str]] = Field(None, description="Optional label filters (future use)")
    last_n_days: Optional[int] = Field(7, ge=0, le=90, description="Recency window for results")

class SearchHit(BaseModel):
    id: str
    text: str
    score: float  # 0..1 (approx similarity)

class SearchResponse(BaseModel):
    hits: List[SearchHit] = []

# ---------- Utilities ----------
def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()

def _bq_table(dataset: str, table: str) -> str:
    return f"{PROJECT_ID}.{dataset}.{table}"

def embed_text(text: str) -> List[float]:
    """Return an embedding vector for the given text. Falls back to zeros if model not loaded or auth fails."""
    if not text:
        return [0.0] * EMBEDDING_DIM
    if _embedding_model:
        try:
            embeddings = _embedding_model.get_embeddings([text])
            vec = embeddings[0].values
            if EMBEDDING_DIM and len(vec) != EMBEDDING_DIM:
                if len(vec) > EMBEDDING_DIM:
                    vec = vec[:EMBEDDING_DIM]
                else:
                    vec = vec + [0.0] * (EMBEDDING_DIM - len(vec))
            return vec
        except (PermissionDenied, Forbidden, Unauthenticated, DefaultCredentialsError, RefreshError) as ex:
            logger.warning(_resolution_hint("Embedding call auth error", ex))
        except Exception as ex:
            logger.exception("Embedding error: %s", ex)
    return [0.0] * EMBEDDING_DIM

def vector_upsert(
    vec: List[float],
    datapoint_id: str,
    restricts: Dict[str, str],
    numeric_restricts: Optional[Dict[str, float]] = None,
) -> None:
    """Upsert a datapoint into Vertex AI Vector Search (Streaming Index)."""
    # Use the INDEX (writes), not the ENDPOINT.
    if not _index:
        logger.info("Vector Search index not bound; skipping upsert.")
        return

    # Build restrictions for the datapoint payload (use allow_list for writes).
    dp_restricts = [
        {"namespace": k, "allow_list": [str(v)]}
        for k, v in (restricts or {}).items()
        if v is not None
    ]

    dp_numeric = []
    for k, v in (numeric_restricts or {}).items():
        dp_numeric.append({"namespace": k, "range": {"start": float(v), "end": float(v)}})

    try:
        _index.upsert_datapoints(
            datapoints=[{
                "datapoint_id": datapoint_id,
                "feature_vector": [float(x) for x in vec],
                # "restricts": dp_restricts,
                # "numeric_restricts": dp_numeric,
            }]
        )
        logger.info("Upserted datapoint id=%s", datapoint_id)
    except Exception as e:
        logger.exception("Vector upsert failed: %s", e)


def vector_query(
    query_vec: List[float],
    k: int,
    restricts: Dict[str, Any],
    numeric_min_ts: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Query nearest neighbors with metadata filters (auth-safe)."""
    if not VERTEX_INDEX_ENDPOINT or not VERTEX_INDEX_ID:
        return []
    if not query_vec:
        logger.warning("Query vector is empty; returning empty results.")
        return []
    try:
        if isinstance(query_vec, (int, float)):
            raise TypeError("Scalar provided; expected an iterable of floats.")
        if hasattr(query_vec, "ravel"):
            clean_query_vec = [float(x) for x in query_vec.ravel().tolist()]
        else:
            clean_query_vec = [float(x) for x in query_vec]
    except Exception as ex:
        logger.exception("Query vector contains non-numeric values: %s", ex)
        return []
    try:
        INDEX_DIM = globals().get("VERTEX_INDEX_DIM")
        if INDEX_DIM and len(clean_query_vec) != int(INDEX_DIM):
            logger.error("Embedding dim mismatch: got %d, expected %d.", len(clean_query_vec), int(INDEX_DIM))
            return []
    except Exception:
        pass

    cat_restricts: List[aiplatform_v1.IndexDatapoint.Restriction] = []
    for ns, val in (restricts or {}).items():
        if val is None:
            continue
        if isinstance(val, str):
            allow = [val]
        elif isinstance(val, (list, tuple)):
            allow = [str(x) for x in val if x is not None]
        else:
            allow = [str(val)]
        if not allow:
            continue
        cat_restricts.append(
            aiplatform_v1.IndexDatapoint.Restriction(
                namespace=str(ns),
                allow_tokens=allow,
            )
        )

    num_restricts: List[aiplatform_v1.IndexDatapoint.NumericRestriction] = []
    if numeric_min_ts is not None:
        num_restricts.append(
            aiplatform_v1.IndexDatapoint.NumericRestriction(
                namespace="event_epoch",
                op=aiplatform_v1.IndexDatapoint.NumericRestriction.Operator.GREATER_EQUAL,
                value_float=float(numeric_min_ts),
            )
        )

    datapoint = aiplatform_v1.IndexDatapoint(
        datapoint_id="q-0",
        feature_vector=clean_query_vec,
        # restricts=cat_restricts,
        # numeric_restricts=num_restricts,
    )

    query_proto = aiplatform_v1.FindNeighborsRequest.Query(
        datapoint=datapoint,
        neighbor_count=int(k),
    )

    request = aiplatform_v1.FindNeighborsRequest(
        index_endpoint=VERTEX_INDEX_ENDPOINT,
        deployed_index_id=VERTEX_INDEX_ID,
        queries=[query_proto],
        return_full_datapoint=False,
    )

    try:
        client = aiplatform_v1.MatchServiceClient(
            client_options={"api_endpoint": VERTEX_API_ENDPOINT}
        )
    except Exception as ex:
        logger.exception("Failed to create MatchServiceClient: %s", ex)
        return []

    try:
        res = client.find_neighbors(request=request)
        out: List[Dict[str, Any]] = []
        for q in res.nearest_neighbors:
            for n in q.neighbors:
                dp = n.datapoint
                out.append({
                    "datapoint_id": getattr(dp, "datapoint_id", None),
                    "distance": float(n.distance),
                    "restricts": [
                        {"namespace": r.namespace, "allowTokens": list(r.allow_tokens)}
                        for r in getattr(dp, "restricts", []) or []
                    ],
                })
        return out
    except Exception as ex:
        logger.exception("Vector query failed: %s", ex)
        return []

def bq_insert_raw_event(source: str, body: Dict[str, Any]) -> None:
    client = get_bq()
    if not client:
        logger.info("Skipping BQ insert (raw_event): no credentials/client.")
        return
    tbl = _bq_table(BQ_DATASET_RAW, "events")
    rows = [{"event_time": datetime.utcnow().isoformat(), "source": source, "body": json.dumps(body)}]
    try:
        client.insert_rows_json(tbl, rows)
    except (PermissionDenied, Forbidden, Unauthenticated, DefaultCredentialsError, RefreshError, NotFound) as ex:
        logger.warning(_resolution_hint("BQ insert raw_event error", ex))
    except Exception as ex:
        logger.exception("BQ insert raw_event failed: %s", ex)

def bq_insert_artifact(
    artifact_id: str,
    kind: str,
    owner: Optional[str],
    source: str,
    pointer: Optional[str],
    text: str,
    created_ts: Optional[str] = None,
) -> None:
    client = get_bq()
    if not client:
        logger.info("Skipping BQ insert (artifact): no credentials/client.")
        return
    tbl = _bq_table(BQ_DATASET_DERIVED, "artifacts")
    rows = [{
        "id": artifact_id,
        "kind": kind,
        "owner": owner,
        "created": created_ts or datetime.utcnow().isoformat(),
        "source": source,
        "pointer": pointer,
        "text": text[:100000],
    }]
    try:
        client.insert_rows_json(tbl, rows)
    except (PermissionDenied, Forbidden, Unauthenticated, DefaultCredentialsError, RefreshError, NotFound) as ex:
        logger.warning(_resolution_hint("BQ insert artifact error", ex))
    except Exception as ex:
        logger.exception("BQ insert artifact failed: %s", ex)

def bq_log_tool(tool: str, user: str, request_obj: Dict[str, Any], response_obj: Dict[str, Any]) -> None:
    client = get_bq()
    if not client:
        logger.info("Skipping BQ insert (tool_log): no credentials/client.")
        return
    tbl = _bq_table(BQ_DATASET_ACTIONS, "tool_invocations")
    rows = [{
        "ts": datetime.utcnow().isoformat(),
        "user": user,
        "tool": tool,
        "request": json.dumps(request_obj)[:100000],
        "response": json.dumps(response_obj)[:100000],
    }]
    try:
        client.insert_rows_json(tbl, rows)
    except (PermissionDenied, Forbidden, Unauthenticated, DefaultCredentialsError, RefreshError, NotFound) as ex:
        logger.warning(_resolution_hint("BQ insert tool_log error", ex))
    except Exception as ex:
        logger.exception("BQ insert tool_log failed: %s", ex)

# ---------- LLM enrichment ----------
def llm_enrich(text: str) -> Dict[str, Any]:
    """
    Return {'summary','priority','due','labels'} using Gemini structured output.
    Falls back gracefully if model unavailable.
    """
    result = {"summary": text[:120] if text else "", "priority": "low", "due": None, "labels": []}
    if not _gemini or not text:
        return result
    try:
        from vertexai.generative_models import GenerationConfig
        schema = {
            "type": "object",
            "properties": {
                "summary":  {"type": "string", "description": "≤15 words, imperative"},
                "priority": {"type": "string", "enum": ["high","medium","low"]},
                "due":      {"type": "string", "nullable": True, "description": "ISO8601 date-time or null"},
                "labels":   {"type": "array", "items": {"type": "string"}}
            },
            "required": ["summary","priority"]
        }
        prompt = (
            "Summarize the event as an action.\n"
            "Return JSON with keys: summary, priority∈{high,medium,low}, due (ISO8601 or null), labels[].\n\n"
            f"Event:\n{text}"
        )
        out = _gemini.generate_content(
            [prompt],
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                response_schema=schema
            ),
        )
        parsed = json.loads(out.candidates[0].content.parts[0].text or "{}")
        result.update({k: v for k, v in parsed.items() if k in result})
        return result
    except Exception as ex:
        logger.warning("LLM enrichment failed; using fallback: %s", ex)
        return result

# ---------- Raw → Artifact normalizer (covers your sample sources) ----------
def normalize_raw_to_artifact(event: Dict[str, Any]) -> Optional[Tuple[str, str, Optional[str], Optional[str], str, float, str]]:
    """
    → (artifact_id, kind, owner, pointer, text, event_epoch, source)
    """
    src = (event.get("source") or "").lower()

    if src == "google-chat":
        msg = event.get("message", {})
        text = (msg.get("text") or "").strip()
        if not text: return None
        owner  = (event.get("user") or "").replace("users/", "") or None
        pointer = msg.get("name")
        art_id  = f"gchat#{pointer or (event.get('space') or '')}"
        ts = event.get("created")
        epoch = datetime.fromisoformat(ts.replace("Z","+00:00")).timestamp() if ts else time.time()
        body = f"Chat: {text}"
        return (art_id, "chat_message", owner, pointer, body, epoch, "google-chat")

    if src == "google-calendar":
        ev = event.get("event", {})
        start, end = ev.get("start"), ev.get("end")
        title = ev.get("summary") or "Calendar event"
        attendees = ", ".join(ev.get("attendees") or [])
        pointer = ev.get("id"); owner = None
        art_id = f"gcal#{pointer or title}"
        epoch = datetime.fromisoformat((start or "").replace("Z","+00:00")).timestamp() if start else time.time()
        body = f"Calendar: {title}\nWhen: {start}–{end}\nAttendees: {attendees}"
        return (art_id, "calendar_event", owner, pointer, body, epoch, "google-calendar")

    if src == "gmail":
        subj = event.get("subject") or "(no subject)"
        frm = event.get("from") or ""
        to  = ", ".join(event.get("to") or [])
        snip = event.get("snippet") or ""
        pointer = event.get("messageId") or event.get("threadId")
        art_id  = f"gmail#{pointer or subj}"
        ts = event.get("received")
        epoch = datetime.fromisoformat(ts.replace("Z","+00:00")).timestamp() if ts else time.time()
        body = f"Email: {subj}\nFrom: {frm}\nTo: {to}\nSnippet: {snip}"
        owner = None
        return (art_id, "email", owner, pointer, body, epoch, "gmail")

    if src == "google-drive":
        title = event.get("title") or "(untitled)"
        action = event.get("action") or "updated"
        actor  = event.get("actor") or ""
        pointer = event.get("fileId")
        art_id  = f"gdrive#{pointer or title}"
        ts = event.get("updated")
        epoch = datetime.fromisoformat(ts.replace("Z","+00:00")).timestamp() if ts else time.time()
        body = f"Drive: {title} was {action} by {actor}"
        owner = actor or None
        return (art_id, "doc_activity", owner, pointer, body, epoch, "google-drive")

    if src == "msteams":
        msg = event.get("text") or ""
        pointer = event.get("messageId")
        art_id  = f"msteams#{pointer or msg[:24]}"
        ts = event.get("created"); epoch = datetime.fromisoformat(ts.replace("Z","+00:00")).timestamp() if ts else time.time()
        owner = event.get("user") or None
        body = f"Teams: {msg}"
        return (art_id, "chat_message", owner, pointer, body, epoch, "msteams")

    if src == "outlook-calendar":
        ev = event.get("event", {})
        title = ev.get("subject") or "Meeting"
        start = ev.get("start"); end = ev.get("end")
        org   = ev.get("organizer") or ""
        pointer = ev.get("id")
        art_id  = f"outlookcal#{pointer or title}"
        epoch = datetime.fromisoformat((start or "").replace("Z","+00:00")).timestamp() if start else time.time()
        body = f"Calendar: {title}\nWhen: {start}–{end}\nOrganizer: {org}"
        owner = org or None
        return (art_id, "calendar_event", owner, pointer, body, epoch, "outlook-calendar")

    return None

# ---------- Endpoints ----------
@app.get("/healthz")
def health():
    return {
        "ok": True,
        "time": utcnow(),
        "service": "jarvis",
        "bq_ready": get_bq() is not None,
        "vertex_ready": _aiplatform is not None,
        "embeddings_ready": _embedding_model is not None,
        "vector_endpoint_bound": _index_ep is not None,
        "project": PROJECT_ID,
        "location": LOCATION,
    }

@app.post("/pubsub")
async def pubsub_push(envelope: PubSubEnvelope, request: Request):
    """
    Pub/Sub push endpoint (Workspace Events → Pub/Sub → Cloud Run).
    Stores the raw event in BigQuery (raw.events).
    """
    if VERIFY_PUBSUB_HEADER:
        topic = request.headers.get("X-Cloud-Pubsub-Topic")
        if not topic:
            raise HTTPException(status_code=400, detail="Missing Pub/Sub topic header")
    try:
        msg = envelope.message
        data64 = msg.get("data", "")
        data = json.loads(base64.b64decode(data64 or b"{}"))
        source = data.get("source", "workspace")
        bq_insert_raw_event(source=source, body=data)
        return Response(status_code=204)
    except Exception as e:
        logger.exception("Pub/Sub push handling failed: %s", e)
        raise HTTPException(status_code=500, detail="pubsub error")

# --- POST /ingest (single artifact) ---
def _artifact_id_from_key(key: str) -> str:
    return "art#" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]

@app.post("/ingest", status_code=status.HTTP_201_CREATED)
async def ingest(
    req: IngestRequest,
    response: Response,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    """
    Ingest a single artifact (posted text) with idempotency.
    - LLM-enrich (summary/priority/due/labels) inlined for posted text
    - Embed
    - Upsert to Vector Search with restricts
    - Write metadata to BigQuery (derived.artifacts)
    """
    if not req.text:
        raise HTTPException(status_code=400, detail="text is required for MVP ingest")

    artifact_id = _artifact_id_from_key(idempotency_key) if idempotency_key else (req.id or f"art#{int(time.time()*1000)}")
    created_ts = req.created or datetime.utcnow().isoformat()

    # Enrich + embed
    enrich = llm_enrich(req.text)
    enriched_text = (
        f"{enrich.get('summary', req.text[:120])}\n"
        f"[priority: {enrich.get('priority','low')}"
        f"{', due: '+enrich['due'] if enrich.get('due') else ''}"
        f"{', labels: '+', '.join(enrich.get('labels',[])) if enrich.get('labels') else ''}]"
        f"\n\n{req.text}"
    )
    vec = embed_text(enriched_text)

    restricts = {"source": req.source or "workspace"}
    if req.owner:
        restricts["owner"] = req.owner

    event_epoch = time.time()
    vector_upsert(vec, artifact_id, restricts, numeric_restricts={"event_epoch": event_epoch})

    bq_insert_artifact(
        artifact_id=artifact_id,
        kind="text",
        owner=req.owner,
        source=req.source or "workspace",
        pointer=req.pointer,
        text=enriched_text,
        created_ts=created_ts,
    )

    response.headers["Location"] = f"/artifacts/{artifact_id}"
    return {"ok": True, "id": artifact_id}

# --- Periodic sweep over raw.events (LLM-enrich → embed → upsert → artifacts) ---
@app.post("/sweep")
async def sweep(hours: int = 24):
    """
    Backfill/refresh: read raw events from the last N hours, enrich with LLM, embed, upsert to VS,
    and persist enriched text to derived.artifacts. Safe to run repeatedly.
    """
    client = get_bq()
    if not client:
        raise HTTPException(status_code=500, detail="BigQuery client unavailable")

    raw_tbl = _bq_table(BQ_DATASET_RAW, "events")
    sql = f"""
      SELECT body
      FROM `{raw_tbl}`
      WHERE TIMESTAMP(event_time) >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @h HOUR)
    """
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("h", "INT64", hours)]
        ),
    )

    processed = 0
    for row in job:
        try:
            body = json.loads(row["body"])
        except Exception:
            continue

        norm = normalize_raw_to_artifact(body)
        if not norm:
            continue

        artifact_id, kind, owner, pointer, base_text, epoch, source = norm

        # LLM enrichment
        enrich = llm_enrich(base_text)
        enriched_text = (
            f"{enrich.get('summary', base_text[:120])}\n"
            f"[priority: {enrich.get('priority','low')}"
            f"{', due: '+enrich['due'] if enrich.get('due') else ''}"
            f"{', labels: '+', '.join(enrich.get('labels',[])) if enrich.get('labels') else ''}]"
            f"\n\n{base_text}"
        )

        # Embed
        vec = embed_text(enriched_text)

        # Upsert to Vector Search with restricts
        restricts = {"source": source}
        if owner:
            restricts["owner"] = owner
        vector_upsert(vec, artifact_id, restricts, numeric_restricts={"event_epoch": epoch})

        # Write enriched artifact
        bq_insert_artifact(
            artifact_id=artifact_id,
            kind=kind,
            owner=owner,
            source=source,
            pointer=pointer,
            text=enriched_text,
            created_ts=datetime.utcfromtimestamp(epoch).isoformat() + "Z",
        )

        processed += 1

    return {"ok": True, "processed": processed}

@app.post("/chat")
async def chat_webhook(evt: ChatEvent):
    if CHAT_BOT_TOKEN and evt.token != CHAT_BOT_TOKEN:
        return {"text": "unauthorized bot caller"}

    user = (evt.user or {}).get("name", "user")
    text = (evt.message or ChatMessage()).text or ""

    if text.strip().lower().startswith("/top3"):
        reply = ("Top 3 priorities (demo):\n"
                 "1) Follow up on client escalations\n"
                 "2) Prepare agenda for 2pm sync\n"
                 "3) Draft budget update email")
        return {"text": reply}

    if text.strip().lower().startswith("/help"):
        return {"text": "Commands: /top3, say anything to echo.\n(I'll soon summarize threads, create meetings, and draft emails.)"}

    return {"text": f"You said: {text}\nTry /top3"}

@app.post("/tasks/daily-digest")
async def daily_digest():
    return {"ok": True, "message": "Daily digest triggered (demo)"}

# ---------- /search Endpoint ----------
@app.post("/search", response_model=SearchResponse)
async def search_endpoint(req: SearchRequest = Body(...)):
    """
    Search previously ingested artifacts using Vector Search.
    Adds recency filter (numeric restrict) and lightweight re-rank:
      score = 0.70*similarity + 0.20*time_decay + 0.10*source_boost
    """
    q = (req.q or "").strip()
    if not q:
        return SearchResponse(hits=[])

    # 1) Embed the query
    try:
        q_vec = embed_text(q)
    except Exception as ex:
        logger.exception("search: embedding failed: %s", ex)
        return SearchResponse(hits=[])

    # 2) Build restricts (owner, labels later)
    restricts: Dict[str, str] = {}
    if req.owner:
        restricts["owner"] = req.owner

    # 3) Recency filter (last_n_days)
    numeric_min_ts = None
    if req.last_n_days and req.last_n_days > 0:
        numeric_min_ts = time.time() - req.last_n_days * 86400

    # 4) Vector Search
    try:
        neighbors = vector_query(
            query_vec=q_vec,
            k=max(req.k, 10),  # fetch a few extra for re-rank
            restricts=restricts,
            numeric_min_ts=numeric_min_ts
        )
    except Exception as ex:
        logger.exception("search: vector_query failed: %s", ex)
        return SearchResponse(hits=[])

    if not neighbors:
        return SearchResponse(hits=[])

    # 5) Hydrate from BigQuery: text + source + created
    ids = [n.get("datapoint_id") for n in neighbors if n.get("datapoint_id")]
    meta_by_id: Dict[str, Dict[str, Any]] = {}

    try:
        if ids:
            tbl = _bq_table(BQ_DATASET_DERIVED, "artifacts")
            sql = f"""
              SELECT id, text, source, created
              FROM `{tbl}`
              WHERE id IN UNNEST(@ids)
            """
            client = get_bq()
            job = client.query(
                sql,
                job_config=bigquery.QueryJobConfig(
                    query_parameters=[bigquery.ArrayQueryParameter("ids", "STRING", ids)]
                ),
            )
            for row in job:
                meta_by_id[row["id"]] = {
                    "text": row["text"],
                    "source": row.get("source"),
                    "created": row.get("created"),
                }
    except Exception as ex:
        logger.exception("search: BigQuery join failed: %s", ex)

    # 6) Re-rank by similarity + recency + source boost
    NOW = time.time()
    def time_decay(created_iso: Optional[str], half_life_hours=24):
        if not created_iso:
            return 0.0
        try:
            dt = datetime.fromisoformat(created_iso.replace("Z", "+00:00"))
            age_h = max(0.0, (NOW - dt.timestamp()) / 3600.0)
            return exp(-age_h / half_life_hours)
        except Exception:
            return 0.0

    SOURCE_BOOST = {
        "google-calendar": 0.15,
        "outlook-calendar": 0.15,
        "gmail": 0.10,
        "google-chat": 0.08,
        "msteams": 0.08,
        "google-drive": 0.06,
    }

    ranked = []
    for n in neighbors:
        dp_id = n.get("datapoint_id")
        if not dp_id:
            continue
        dist = float(n.get("distance", 0.0))
        sim = max(0.0, min(1.0, 1.0 - dist))
        meta = meta_by_id.get(dp_id, {})
        src = (meta.get("source") or "workspace").lower()
        freshness = time_decay(meta.get("created"))
        sboost = SOURCE_BOOST.get(src, 0.0)
        score = 0.70 * sim + 0.20 * freshness + 0.10 * sboost
        ranked.append((score, dp_id))

    ranked.sort(reverse=True, key=lambda x: x[0])
    top_ids = [dp for _, dp in ranked[: req.k]]

    # 7) Build response
    hits: List[SearchHit] = []
    for dp_id in top_ids:
        meta = meta_by_id.get(dp_id, {})
        txt = (meta.get("text") or "").strip()
        if len(txt) > 1000:
            txt = txt[:1000] + " …"
        # Use the combined score we computed
        sc = next((s for s, i in ranked if i == dp_id), 0.0)
        hits.append(SearchHit(id=dp_id, text=txt, score=sc))

    return SearchResponse(hits=hits)

# ---------- Entrypoint ----------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, workers=1)
