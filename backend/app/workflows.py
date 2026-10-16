"""Authenticated workflow lifecycle with durable checkpoints and execution leases."""

import json
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, Field, StrictBool, field_validator

from .auth import current_user
from .config import DATA_DIR
from .db import connection
from .workflow_graph import build_graph

router = APIRouter(prefix="/api/workflows", tags=["Workflows"])
CHECKPOINT_PATH = DATA_DIR / "workflow-checkpoints.sqlite"
LEASE_SECONDS = 600
logger = logging.getLogger(__name__)


class WorkflowRequest(BaseModel):
    kind: Literal["research", "summary", "extract"]
    goal: str = Field(min_length=2, max_length=2000)
    document_ids: list[str] = Field(min_length=1, max_length=50)

    @field_validator("goal")
    @classmethod
    def trim_goal(cls, value):
        if len(value.strip()) < 2:
            raise ValueError("Enter a goal for the workflow.")
        return value.strip()


class Decision(BaseModel):
    approved: StrictBool


def owned(workflow_id, owner_id):
    with connection() as db:
        row = db.execute("SELECT * FROM workflows WHERE id = ? AND owner_id = ?", (str(workflow_id), owner_id)).fetchone()
    if not row:
        raise HTTPException(404, "Workflow not found.")
    return dict(row)


def check_documents(db, document_ids, owner_id):
    marks = ",".join("?" for _ in document_ids)
    count = db.execute(f"SELECT COUNT(*) AS total FROM documents WHERE owner_id = ? AND id IN ({marks})",
                       [owner_id, *document_ids]).fetchone()["total"]
    if count != len(set(document_ids)):
        raise HTTPException(400, "One or more selected documents were not found.")


def save_report(state, run_token=None):
    if state.get("approved") is not True or not state.get("supported"):
        raise ValueError("A report requires evidence review and human approval.")
    report = {"id": state["id"], "kind": state["kind"], "goal": state["goal"],
              "answer": state["draft"], "citations": state["citations"], "records": state.get("records", [])}
    with connection() as db:
        check_documents(db, state["document_ids"], state["owner_id"])
        if run_token:
            active = db.execute("SELECT 1 FROM workflows WHERE id = ? AND owner_id = ? AND run_token = ? AND lease_until > ?",
                                (state["id"], state["owner_id"], run_token, time.time())).fetchone()
            if not active:
                raise RuntimeError("Workflow lease lost before saving.")
        db.execute("INSERT INTO reports (id, owner_id, content, created_at) VALUES (?, ?, ?, ?) ON CONFLICT (id) DO NOTHING",
                   (state["id"], state["owner_id"], json.dumps(report), datetime.now(timezone.utc).isoformat()))


@contextmanager
def graph_session(save_fn=save_report):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(CHECKPOINT_PATH)) as saver:
        yield build_graph(saver, save_fn)


def config(workflow_id):
    return {"configurable": {"thread_id": str(workflow_id)}, "recursion_limit": 60}


def snapshot_view(row, graph):
    snapshot = graph.get_state(config(row["id"]))
    state = snapshot.values or {}
    return {"id": row["id"], "kind": row["kind"], "goal": row["goal"], "status": row["status"],
            "created_at": row["created_at"], "error": row["error"],
            "can_retry": row["status"] in {"pending", "failed"} or (row["status"] == "running" and row["lease_until"] < time.time()),
            "draft": state.get("draft", ""), "citations": state.get("citations", []),
            "records": state.get("records", []), "feedback": state.get("feedback", ""),
            "steps": state.get("steps", [])}


def execute(row, *, initial=None, decision=None):
    token = str(uuid.uuid4())
    now = time.time()
    with connection() as db:
        claimed = db.execute(
            "UPDATE workflows SET status = 'running', run_token = ?, lease_until = ?, error = NULL "
            "WHERE id = ? AND owner_id = ? AND (status IN ('pending', 'paused', 'failed') "
            "OR (status = 'running' AND lease_until < ?))",
            (token, now + LEASE_SECONDS, row["id"], row["owner_id"], now),
        ).rowcount
    if not claimed:
        raise HTTPException(409, "This workflow is already running or finished. Refresh its status.")
    try:
        with graph_session(lambda state: save_report(state, token)) as graph:
            snapshot = graph.get_state(config(row["id"]))
            is_interrupted = any(task.interrupts for task in snapshot.tasks)
            if decision is not None and not is_interrupted:
                raise ValueError("This workflow is not waiting for approval.")
            if is_interrupted and decision is None:
                # Recovery of a crash after persisting an interrupt must not auto-approve.
                status = "paused"
            else:
                graph_input = Command(resume=decision) if decision is not None else initial
                if not snapshot.values and initial is None:
                    graph_input = {"id": row["id"], "owner_id": row["owner_id"], "kind": row["kind"],
                                   "goal": row["goal"], "document_ids": json.loads(row["document_ids"])}
                for _ in graph.stream(graph_input, config(row["id"]), stream_mode="values"):
                    with connection() as db:
                        renewed = db.execute("UPDATE workflows SET lease_until = ? WHERE id = ? AND run_token = ?",
                                             (time.time() + LEASE_SECONDS, row["id"], token)).rowcount
                    if not renewed:
                        raise RuntimeError("Workflow lease lost.")
                snapshot = graph.get_state(config(row["id"]))
                status = "paused" if any(task.interrupts for task in snapshot.tasks) else snapshot.values.get("status", "failed")
        with connection() as db:
            db.execute("UPDATE workflows SET status = ?, run_token = NULL, lease_until = 0 WHERE id = ? AND run_token = ?",
                       (status, row["id"], token))
    except Exception as exc:
        # Only known validation errors are exposed; provider payloads can contain private data.
        detail = str(exc) if isinstance(exc, ValueError) and not hasattr(exc, "errors") else "Workflow step failed. You can retry from the last checkpoint."
        logger.warning("Workflow step failed: %s", type(exc).__name__)
        with connection() as db:
            db.execute("UPDATE workflows SET status = 'failed', error = ?, run_token = NULL, lease_until = 0 WHERE id = ? AND run_token = ?",
                       (detail[:300], row["id"], token))
    return get_workflow(uuid.UUID(row["id"]), {"id": row["owner_id"]})


@router.get("")
def list_workflows(user: dict = Depends(current_user)):
    with connection() as db:
        return [dict(row) for row in db.execute(
            "SELECT id, kind, goal, status, created_at FROM workflows WHERE owner_id = ? ORDER BY created_at DESC LIMIT 100",
            (user["id"],),
        )]


@router.post("", status_code=201)
def create_workflow(request: WorkflowRequest, user: dict = Depends(current_user)):
    workflow_id = str(uuid.uuid4())
    document_ids = list(dict.fromkeys(request.document_ids))
    with connection() as db:
        check_documents(db, document_ids, user["id"])
        db.execute("INSERT INTO workflows (id, owner_id, kind, goal, document_ids, status, created_at) "
                   "VALUES (?, ?, ?, ?, ?, 'pending', ?)",
                   (workflow_id, user["id"], request.kind, request.goal, json.dumps(document_ids), datetime.now(timezone.utc).isoformat()))
    row = owned(workflow_id, user["id"])
    return execute(row, initial={"id": workflow_id, "owner_id": user["id"], "kind": request.kind,
                                 "goal": request.goal, "document_ids": document_ids})


@router.get("/{workflow_id}")
def get_workflow(workflow_id: uuid.UUID, user: dict = Depends(current_user)):
    row = owned(workflow_id, user["id"])
    with graph_session() as graph:
        return snapshot_view(row, graph)


@router.post("/{workflow_id}/decision")
def decide_workflow(workflow_id: uuid.UUID, request: Decision, user: dict = Depends(current_user)):
    row = owned(workflow_id, user["id"])
    if row["status"] != "paused":
        raise HTTPException(409, "This workflow is not waiting for approval.")
    return execute(row, decision=request.approved)


@router.post("/{workflow_id}/retry")
def retry_workflow(workflow_id: uuid.UUID, user: dict = Depends(current_user)):
    row = owned(workflow_id, user["id"])
    if row["status"] not in {"pending", "failed"} and not (row["status"] == "running" and row["lease_until"] < time.time()):
        raise HTTPException(409, "Only failed or interrupted runs can be retried.")
    return execute(row)


@router.get("/{workflow_id}/report")
def get_report(workflow_id: uuid.UUID, user: dict = Depends(current_user)):
    owned(workflow_id, user["id"])
    with connection() as db:
        row = db.execute("SELECT content FROM reports WHERE id = ? AND owner_id = ?", (str(workflow_id), user["id"])).fetchone()
    if not row:
        raise HTTPException(404, "No approved report exists for this workflow.")
    return json.loads(row["content"])


@router.delete("/{workflow_id}", status_code=204)
def delete_workflow(workflow_id: uuid.UUID, user: dict = Depends(current_user)):
    owned(workflow_id, user["id"])
    with connection() as db:
        deleted = db.execute("DELETE FROM workflows WHERE id = ? AND owner_id = ? AND status != 'running'",
                             (str(workflow_id), user["id"])).rowcount
    if not deleted:
        raise HTTPException(409, "A running workflow cannot be deleted. Wait for it to finish or recover it first.")
    with SqliteSaver.from_conn_string(str(CHECKPOINT_PATH)) as saver:
        saver.delete_thread(str(workflow_id))
