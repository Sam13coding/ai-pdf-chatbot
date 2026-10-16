import hashlib
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Cookie, Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import APIStatusError
from pydantic import BaseModel, Field

from .config import EMBEDDING_MODEL, FRONTEND_ORIGIN, MAX_PDF_BYTES, UPLOAD_DIR
from .auth import Credentials, current_user, login, logout, register
from .db import connection, init_db
from .pdf import chunk_pages, extract_pages
from .rag import MissingNvidiaKeyError, answer_question, answer_with_background, embeddings, retrieve, standalone_question
from .chat_graph import run_chat
from .workflows import router as workflow_router

logger = logging.getLogger(__name__)


def nvidia_error_response(exc: APIStatusError) -> HTTPException:
    if exc.status_code == 401:
        return HTTPException(502, "NVIDIA rejected the API key. Check NVIDIA_API_KEY in backend/.env.")
    if exc.status_code == 402:
        return HTTPException(402, "NVIDIA API credits or free endpoint access are unavailable. Check your NVIDIA Build account.")
    if exc.status_code == 429:
        return HTTPException(429, "NVIDIA API rate limit reached. Wait and try again.")
    if exc.status_code in {400, 404, 422}:
        return HTTPException(502, "NVIDIA rejected the configured model or request. Check NVIDIA model settings.")
    return HTTPException(502, "NVIDIA API request failed. Check the server log for details.")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="PAGESUTRA API", lifespan=lifespan)
app.include_router(workflow_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def monitor_request(request: Request, call_next):
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        route = request.scope.get("route")
        # Route templates avoid recording IDs, query strings, bodies or credentials.
        logging.getLogger("uvicorn.error").info(json.dumps({
            "event": "http_request", "request_id": request_id,
            "method": request.method, "route": getattr(route, "path", "unmatched"),
            "status": status_code,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        }))


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    conversation_id: str | None = None
    document_ids: list[str] = Field(default_factory=list, max_length=50)
    answer_mode: Literal["documents", "background"] = "documents"


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/auth/signup", status_code=201)
def signup(credentials: Credentials, response: Response):
    return register(credentials, response)


@app.post("/api/auth/signin")
def signin(credentials: Credentials, response: Response):
    return login(credentials, response)


@app.post("/api/auth/signout", status_code=204)
def signout(response: Response, token: str | None = Cookie(default=None, alias="pagesutra_session")):
    logout(response, token)


@app.get("/api/auth/me")
def me(user: dict = Depends(current_user)):
    return user


@app.get("/api/documents")
def list_documents(user: dict = Depends(current_user)):
    with connection() as db:
        return [dict(row) for row in db.execute(
            "SELECT id, name, pages, chunks, created_at FROM documents WHERE owner_id = ? ORDER BY created_at DESC, id DESC",
            (user["id"],)
        )]


@app.post("/api/documents", status_code=201)
async def upload_document(file: UploadFile = File(...), user: dict = Depends(current_user)):
    name = (file.filename or "document.pdf").replace("\\", "/").split("/")[-1][:180]
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")
    data = await file.read(MAX_PDF_BYTES + 1)
    await file.close()
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(413, "PDF exceeds the upload size limit.")
    if not data.startswith(b"%PDF-"):
        raise HTTPException(400, "The file is not a valid PDF.")
    digest = hashlib.sha256(user["id"].encode() + b":" + data).hexdigest()
    legacy_digest = hashlib.sha256(data).hexdigest()
    with connection() as db:
        existing = db.execute("SELECT id, name FROM documents WHERE owner_id = ? AND sha256 IN (?, ?)",
                              (user["id"], digest, legacy_digest)).fetchone()
    if existing:
        raise HTTPException(409, f"This PDF is already uploaded as {existing['name']}.")
    try:
        pages = extract_pages(data)
        chunks = list(chunk_pages(pages))
        vectors = embeddings([text for _, text in chunks], "passage")
    except MissingNvidiaKeyError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except APIStatusError as exc:
        logger.warning("NVIDIA upload request failed: %s", exc.status_code)
        raise nvidia_error_response(exc) from exc
    except Exception as exc:
        logger.exception("Upload processing failed")
        raise HTTPException(502, "Could not process the PDF. Check the server log and API key.") from exc
    document_id = str(uuid.uuid4())
    path = UPLOAD_DIR / f"{document_id}.pdf"
    try:
        path.write_bytes(data)
        with connection() as db:
            db.execute(
                "INSERT INTO documents (id, name, sha256, pages, chunks, embedding_model, owner_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (document_id, name, digest, len(pages), len(chunks), EMBEDDING_MODEL, user["id"]),
            )
            db.executemany(
                "INSERT INTO chunks (id, document_id, page, text, embedding) VALUES (?, ?, ?, ?, ?)",
                ((str(uuid.uuid4()), document_id, page, text, json.dumps(vector))
                 for (page, text), vector in zip(chunks, vectors)),
            )
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return {"id": document_id, "name": name, "pages": len(pages), "chunks": len(chunks)}


@app.get("/api/documents/{document_id}/file")
def get_document_file(document_id: uuid.UUID, user: dict = Depends(current_user)):
    with connection() as db:
        document = db.execute("SELECT name FROM documents WHERE id = ? AND owner_id = ?", (str(document_id), user["id"])).fetchone()
    path = UPLOAD_DIR / f"{document_id}.pdf"
    if not document or not path.is_file():
        raise HTTPException(404, "Document not found.")
    return FileResponse(path, media_type="application/pdf", filename=document["name"], content_disposition_type="inline")


@app.delete("/api/documents/{document_id}", status_code=204)
def delete_document(document_id: uuid.UUID, user: dict = Depends(current_user)):
    with connection() as db:
        deleted = db.execute("DELETE FROM documents WHERE id = ? AND owner_id = ?", (str(document_id), user["id"])).rowcount
    if not deleted:
        raise HTTPException(404, "Document not found.")
    (UPLOAD_DIR / f"{document_id}.pdf").unlink(missing_ok=True)


@app.get("/api/conversations")
def list_conversations(user: dict = Depends(current_user)):
    with connection() as db:
        return [dict(row) for row in db.execute(
            "SELECT id, title, created_at FROM conversations WHERE owner_id = ? ORDER BY created_at DESC, id DESC",
            (user["id"],)
        )]


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: uuid.UUID, user: dict = Depends(current_user)):
    with connection() as db:
        conversation = db.execute(
            "SELECT id, title, created_at FROM conversations WHERE id = ? AND owner_id = ?", (str(conversation_id), user["id"])
        ).fetchone()
        if not conversation:
            raise HTTPException(404, "Conversation not found.")
        messages = [
            {**dict(row), "citations": json.loads(row["citations"])}
            for row in db.execute(
                "SELECT id, role, content, citations, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY created_at, id", (str(conversation_id),)
            )
        ]
    return {**dict(conversation), "messages": messages}


@app.delete("/api/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: uuid.UUID, user: dict = Depends(current_user)):
    with connection() as db:
        deleted = db.execute("DELETE FROM conversations WHERE id = ? AND owner_id = ?", (str(conversation_id), user["id"])).rowcount
    if not deleted:
        raise HTTPException(404, "Conversation not found.")


@app.post("/api/chat")
def chat(request: ChatRequest, user: dict = Depends(current_user)):
    question = request.question.strip()
    if len(question) < 2:
        raise HTTPException(400, "Please enter a question.")
    with connection() as db:
        count = db.execute("SELECT COUNT(*) AS total FROM documents WHERE owner_id = ?", (user["id"],)).fetchone()["total"]
        if count == 0:
            raise HTTPException(400, "Upload a PDF before asking a question.")
        if request.document_ids:
            marks = ",".join("?" for _ in request.document_ids)
            selected = db.execute(
                f"SELECT COUNT(*) AS total FROM documents WHERE owner_id = ? AND id IN ({marks})", [user["id"], *request.document_ids]
            ).fetchone()["total"]
            if selected != len(set(request.document_ids)):
                raise HTTPException(400, "One or more selected documents were not found.")
        history = []
        if request.conversation_id:
            exists = db.execute(
                "SELECT 1 FROM conversations WHERE id = ? AND owner_id = ?", (request.conversation_id, user["id"])
            ).fetchone()
            if not exists:
                raise HTTPException(404, "Conversation not found.")
            history = [dict(row) for row in db.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at DESC, id DESC LIMIT 8",
                (request.conversation_id,),
            )][::-1]
    try:
        answer_fn = answer_with_background if request.answer_mode == "background" else answer_question
        answer, citations = run_chat(question, history, request.document_ids, user["id"], retrieve, answer_fn, standalone_question)
    except MissingNvidiaKeyError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except APIStatusError as exc:
        logger.warning("NVIDIA chat request failed: %s", exc.status_code)
        raise nvidia_error_response(exc) from exc
    except Exception as exc:
        logger.exception("Chat failed")
        raise HTTPException(502, "Could not generate an answer. Check the server log and API key.") from exc
    conversation_id = request.conversation_id or str(uuid.uuid4())
    with connection() as db:
        if not request.conversation_id:
            db.execute(
                "INSERT INTO conversations (id, title, owner_id) VALUES (?, ?, ?)",
                (conversation_id, question[:70], user["id"]),
            )
        db.execute(
            "INSERT INTO messages (id, conversation_id, role, content) VALUES (?, ?, 'user', ?)",
            (str(uuid.uuid4()), conversation_id, question),
        )
        db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, citations) "
            "VALUES (?, ?, 'assistant', ?, ?)",
            (str(uuid.uuid4()), conversation_id, answer, json.dumps(citations)),
        )
    return {"conversation_id": conversation_id, "answer": answer, "citations": citations}
