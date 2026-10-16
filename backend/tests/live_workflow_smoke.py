"""Opt-in NVIDIA smoke test; uses synthetic documents and temporary databases only."""

import json
import argparse
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app import db, rag
from app.config import EMBEDDING_MODEL
from app.workflow_graph import build_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["all", "summary", "extract", "research"], default="all")
    args = parser.parse_args()
    text = "Project Atlas launches June 15. Employees receive 20 days of annual leave, approved by the line manager."
    vector = rag.embeddings([text], "passage")[0]
    with ExitStack() as stack:
        root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        stack.enter_context(patch.object(db, "DATABASE_URL", ""))
        stack.enter_context(patch.object(db, "DB_PATH", root / "app.db"))
        stack.enter_context(patch.object(db, "UPLOAD_DIR", root / "uploads"))
        db.init_db()
        with db.connection() as conn:
            conn.execute("INSERT INTO users (id, email, password_hash) VALUES ('smoke-owner', 'smoke@example.com', 'not-a-login')")
            conn.execute("INSERT INTO documents (id, name, sha256, pages, chunks, embedding_model, owner_id) VALUES ('smoke-doc', 'atlas.pdf', 'synthetic', 1, 1, ?, 'smoke-owner')", (EMBEDDING_MODEL,))
            conn.execute("INSERT INTO chunks (id, document_id, page, text, embedding) VALUES ('smoke-chunk', 'smoke-doc', 1, ?, ?)", (text, json.dumps(vector)))
        for kind, goal in [
            ("summary", "Summarize the launch and annual leave policy."),
            ("extract", "Extract launch date, annual leave allowance, and leave approver."),
            ("research", "Research when Project Atlas launches and how annual leave works."),
        ]:
            if args.kind != "all" and args.kind != kind:
                continue
            saved = []
            config = {"configurable": {"thread_id": kind}, "recursion_limit": 60}
            checkpoint_path = str(root / "checkpoints.sqlite")
            with SqliteSaver.from_conn_string(checkpoint_path) as saver:
                graph = build_graph(saver, lambda state: saved.append(state["draft"]))
                result = graph.invoke({"id": kind, "owner_id": "smoke-owner", "kind": kind,
                                       "goal": goal, "document_ids": ["smoke-doc"]}, config)
                if not result.get("__interrupt__"):
                    raise AssertionError(f"{kind} did not reach human review: {result.get('feedback')}; notes={result.get('notes')}; draft={result.get('draft')}")
                assert not saved, "Report was saved before approval"
            # Recreate the graph/checkpointer to exercise restart recovery with the real model.
            with SqliteSaver.from_conn_string(checkpoint_path) as saver:
                graph = build_graph(saver, lambda state: saved.append(state["draft"]))
                result = graph.invoke(Command(resume=True), config)
                assert result["status"] == "completed" and len(saved) == 1
                assert "June 15" in result["draft"] and result["citations"]
                tool_rounds = result.get("tool_rounds", 0)
                print(f"{kind}: draft -> persisted review -> approved report OK (tool rounds: {tool_rounds})", flush=True)
        answer, citations = rag.answer_with_background(
            "What is this leave policy, and what is annual leave generally?", [],
            [{"document_id": "smoke-doc", "document_name": "atlas.pdf", "page": 1, "text": text}],
        )
        assert "20" in answer and citations and "Related background" in answer
        print("background: PDF evidence and general explanation separated OK", flush=True)


if __name__ == "__main__":
    main()
