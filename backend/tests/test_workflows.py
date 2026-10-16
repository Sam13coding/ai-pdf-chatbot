import json
import tempfile
import unittest
import uuid
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app import db, main, workflow_graph as graph, workflows


class WorkflowTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        for obj, name, value in [(db, "DATABASE_URL", ""), (db, "DB_PATH", root / "app.db"),
                                  (db, "UPLOAD_DIR", root / "uploads"),
                                  (workflows, "CHECKPOINT_PATH", root / "checkpoints.sqlite")]:
            self.stack.enter_context(patch.object(obj, name, value))
        self.client = self.stack.enter_context(TestClient(main.app))
        user = self.client.post("/api/auth/signup", json={"email": "owner@example.com", "password": "strongpassword1"}).json()
        self.owner = user["id"]
        self.document_id = str(uuid.uuid4())
        with db.connection() as conn:
            conn.execute("INSERT INTO documents (id, name, sha256, pages, chunks, embedding_model, owner_id) VALUES (?, ?, ?, 1, 1, ?, ?)",
                         (self.document_id, "launch.pdf", "unique-hash", main.EMBEDDING_MODEL, self.owner))
            conn.execute("INSERT INTO chunks (id, document_id, page, text, embedding) VALUES ('chunk', ?, 1, ?, '[1,0]')",
                         (self.document_id, "The project launch is June 15."))
        self.map_calls = 0
        self.writer_calls = 0
        self.fail_writer = False
        self.review_supported = True
        self.bad_extraction = False
        self.stack.enter_context(patch.object(graph, "chain", side_effect=self.fake_chain))

    def fake_chain(self, system, *, structured=False):
        def invoke(_):
            if "evidence reviewer" in system:
                return {"supported": self.review_supported, "feedback": "Checked against page 1."}
            if system.startswith("Extract concise evidence notes"):
                self.map_calls += 1
                return "Launch: June 15. [1]"
            self.writer_calls += 1
            if self.fail_writer:
                self.fail_writer = False
                raise RuntimeError("Simulated provider interruption")
            if structured:
                return {"records": [{"field": "launch", "value": "June 15", "sources": [99 if self.bad_extraction else 1]}]}
            return "The project launches June 15. [1]"
        return RunnableLambda(invoke)

    def start(self, kind="summary"):
        response = self.client.post("/api/workflows", json={"kind": kind, "goal": "When is launch?", "document_ids": [self.document_id]})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def test_approval_survives_new_checkpoint_session_and_is_required_for_report(self):
        run = self.start()
        self.assertEqual(run["status"], "paused", run)
        self.assertEqual(self.client.get(f"/api/workflows/{run['id']}/report").status_code, 404)
        restored = self.client.get(f"/api/workflows/{run['id']}").json()
        self.assertEqual(restored["draft"], run["draft"])
        self.assertEqual(self.client.post(f"/api/workflows/{run['id']}/decision", json={"approved": "yes"}).status_code, 422)
        decision = self.client.post(f"/api/workflows/{run['id']}/decision", json={"approved": True})
        self.assertEqual(decision.json()["status"], "completed", decision.text)
        report = self.client.get(f"/api/workflows/{run['id']}/report").json()
        self.assertEqual(report["citations"][0]["document_id"], self.document_id)
        self.assertEqual(self.client.post(f"/api/workflows/{run['id']}/decision", json={"approved": True}).status_code, 409)
        self.assertEqual(self.map_calls, 1)
        self.assertEqual(self.writer_calls, 1)

    def test_rejection_does_not_save_report(self):
        run = self.start()
        result = self.client.post(f"/api/workflows/{run['id']}/decision", json={"approved": False})
        self.assertEqual(result.json()["status"], "rejected", result.text)
        self.assertEqual(self.client.get(f"/api/workflows/{run['id']}/report").status_code, 404)

    def test_failed_writer_retries_without_repeating_completed_map_step(self):
        self.fail_writer = True
        run = self.start()
        self.assertEqual(run["status"], "failed", run)
        self.assertTrue(run["can_retry"])
        result = self.client.post(f"/api/workflows/{run['id']}/retry")
        self.assertEqual(result.json()["status"], "paused", result.text)
        self.assertEqual(self.map_calls, 1)
        self.assertEqual(self.writer_calls, 2)

    def test_evidence_failure_is_bounded_and_cannot_be_approved(self):
        self.review_supported = False
        run = self.start()
        self.assertEqual(run["status"], "blocked", run)
        self.assertEqual(self.writer_calls, 2)
        self.assertEqual(self.client.post(f"/api/workflows/{run['id']}/decision", json={"approved": True}).status_code, 409)
        self.assertEqual(self.client.get(f"/api/workflows/{run['id']}/report").status_code, 404)

    def test_extraction_validates_sources_and_exports_structured_records(self):
        self.bad_extraction = True
        run = self.start("extract")
        self.assertEqual(run["status"], "failed", run)
        self.bad_extraction = False
        resumed = self.client.post(f"/api/workflows/{run['id']}/retry").json()
        self.assertEqual(resumed["status"], "paused", resumed)
        self.assertEqual(resumed["records"][0]["value"], "June 15")
        self.client.post(f"/api/workflows/{run['id']}/decision", json={"approved": True})
        report = self.client.get(f"/api/workflows/{run['id']}/report").json()
        self.assertEqual(report["records"][0]["sources"], [1])

    def test_workflows_and_tools_enforce_owner_and_document_scope(self):
        run = self.start()
        other = self.stack.enter_context(TestClient(main.app))
        other.post("/api/auth/signup", json={"email": "other@example.com", "password": "strongpassword2"})
        self.assertEqual(other.get("/api/workflows").json(), [])
        for suffix in ["", "/report"]:
            self.assertEqual(other.get(f"/api/workflows/{run['id']}{suffix}").status_code, 404)
        self.assertEqual(other.post(f"/api/workflows/{run['id']}/decision", json={"approved": True}).status_code, 404)
        self.assertEqual(other.post(f"/api/workflows/{run['id']}/retry").status_code, 404)
        self.assertEqual(other.post("/api/workflows", json={"kind": "summary", "goal": "Read it", "document_ids": [self.document_id]}).status_code, 400)
        tools = graph.document_tools(self.owner, [self.document_id])
        self.assertEqual(json.loads(tools["read_document_page"].invoke({"document_id": "foreign", "page": 1})), [])
        allowed = json.loads(tools["read_document_page"].invoke({"document_id": self.document_id, "page": 1}))
        self.assertEqual(allowed[0]["text"], "The project launch is June 15.")

    def test_research_executes_model_selected_read_only_tool(self):
        model = MagicMock()
        model.bind_tools.return_value = model
        model.invoke.side_effect = [
            AIMessage(content="", tool_calls=[{"name": "read_document_page", "args": {"document_id": self.document_id, "page": 1}, "id": "call1"}]),
            AIMessage(content="Evidence gathered."),
        ]
        source = {"document_id": self.document_id, "document_name": "launch.pdf", "page": 1, "text": "Launch June 15"}
        with patch.object(graph, "chat_model", return_value=model), patch.object(graph, "answer_question", return_value=("June 15. [1]", [{"number": 1, **source, "snippet": "Launch June 15"}])):
            run = self.start("research")
        self.assertEqual(run["status"], "paused", run)
        self.assertIn("Read-only document tools executed", run["steps"])
        self.assertEqual(model.invoke.call_count, 2)

    def test_summary_size_limit_is_explicit(self):
        with db.connection() as conn:
            conn.executemany("INSERT INTO chunks (id, document_id, page, text, embedding) VALUES (?, ?, 1, 'more', '[1,0]')",
                             [(f"extra-{n}", self.document_id) for n in range(24)])
        run = self.start()
        self.assertEqual(run["status"], "failed", run)
        self.assertIn("24 chunks", run["error"])

    def test_delete_removes_report_and_checkpoint(self):
        run = self.start()
        self.client.post(f"/api/workflows/{run['id']}/decision", json={"approved": True})
        self.assertEqual(self.client.delete(f"/api/workflows/{run['id']}").status_code, 204)
        self.assertEqual(self.client.get(f"/api/workflows/{run['id']}").status_code, 404)
        with workflows.graph_session() as compiled:
            self.assertFalse(compiled.get_state(workflows.config(run["id"])).values)
        with db.connection() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) AS n FROM reports").fetchone()["n"], 0)

    def test_pending_run_recovers_if_process_stopped_before_first_checkpoint(self):
        run = self.start()
        with workflows.graph_session() as compiled:
            compiled.checkpointer.delete_thread(run["id"])
        with db.connection() as conn:
            conn.execute("UPDATE workflows SET status = 'pending' WHERE id = ?", (run["id"],))
        restored = self.client.get(f"/api/workflows/{run['id']}").json()
        self.assertTrue(restored["can_retry"])
        resumed = self.client.post(f"/api/workflows/{run['id']}/retry").json()
        self.assertEqual(resumed["status"], "paused", resumed)


if __name__ == "__main__":
    unittest.main()
