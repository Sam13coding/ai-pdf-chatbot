import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pymupdf
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from openai import RateLimitError

from app import db, main, pdf as pdf_tools, rag


def sample_pdf():
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "The project launch date is June 15. This test PDF has enough text to verify extraction and citations.")
    data = document.tobytes()
    document.close()
    return data


class ApiFlowTest(unittest.TestCase):
    def test_first_account_receives_existing_local_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (patch.object(db, "DATABASE_URL", ""),
                  patch.object(db, "DB_PATH", root / "test.db"),
                  patch.object(db, "UPLOAD_DIR", root / "uploads"),
                  patch.object(main, "UPLOAD_DIR", root / "uploads")):
                db.init_db()
                with db.connection() as connection:
                    connection.execute(
                        "INSERT INTO documents (id, name, sha256, pages, chunks, embedding_model) VALUES (?, ?, ?, ?, ?, ?)",
                        ("legacy-doc", "old.pdf", "legacy-hash", 1, 0, "legacy"),
                    )
                    connection.execute("INSERT INTO conversations (id, title) VALUES (?, ?)", ("legacy-chat", "Old question"))
                with TestClient(main.app) as client:
                    self.assertEqual(client.post("/api/auth/signup", json={"email": "owner@example.com", "password": "strongpassword1"}).status_code, 201)
                    self.assertEqual(client.get("/api/documents").json()[0]["id"], "legacy-doc")
                    self.assertEqual(client.get("/api/conversations").json()[0]["id"], "legacy-chat")
                with TestClient(main.app) as other:
                    self.assertEqual(other.post("/api/auth/signup", json={"email": "other@example.com", "password": "strongpassword2"}).status_code, 201)
                    self.assertEqual(other.get("/api/documents").json(), [])
                    self.assertEqual(other.get("/api/conversations").json(), [])

    def test_pdf_hyperlink_annotation_is_extracted(self):
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((72, 72), "GitHub", fontsize=12)
        page.insert_link({
            "kind": pymupdf.LINK_URI,
            "from": pymupdf.Rect(70, 58, 125, 78),
            "uri": "https://github.com/example/profile",
        })
        content = document.tobytes()
        document.close()

        extracted = pdf_tools.extract_pages(content)[0][1]
        self.assertIn("GitHub", extracted)
        self.assertIn("https://github.com/example/profile", extracted)

    @unittest.skipUnless(pdf_tools.TESSDATA_DIR.is_dir(), "Tesseract language data is not installed")
    def test_scanned_pdf_ocr(self):
        source = pymupdf.open()
        source_page = source.new_page()
        source_page.insert_text((72, 120), "SCANNED INVOICE TOTAL 125 DOLLARS", fontsize=22)
        image = source_page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
        scanned = pymupdf.open()
        scanned_page = scanned.new_page(width=source_page.rect.width, height=source_page.rect.height)
        scanned_page.insert_image(scanned_page.rect, pixmap=image)
        content = scanned.tobytes()
        source.close()
        scanned.close()

        pages = pdf_tools.extract_pages(content)
        self.assertIn("125", pages[0][1])

    def test_nvidia_embedding_modes_and_chat(self):
        provider = MagicMock()
        provider.embed_documents.return_value = [[1.0, 0.0]]
        provider.embed_query.return_value = [1.0, 0.0]
        chat_model = RunnableLambda(lambda _: AIMessage(content="The launch is June 15. [1]"))
        source = {"document_id": "doc", "document_name": "sample.pdf", "page": 1,
                  "text": "The launch is June 15."}
        with (patch.object(rag, "embedding_model", return_value=provider),
              patch.object(rag, "require_nvidia_key"),
              patch.object(rag, "ChatOpenAI", return_value=chat_model)):
            self.assertEqual(rag.embeddings(["PDF text"], "passage"), [[1.0, 0.0]])
            provider.embed_documents.assert_called_once_with(["PDF text"])
            self.assertEqual(rag.embeddings(["launch?"], "query"), [[1.0, 0.0]])
            provider.embed_query.assert_called_once_with("launch?")
            answer, citations = rag.answer_question("When?", [], [source])
        self.assertIn("June 15", answer)
        self.assertEqual(citations[0]["page"], 1)

    def test_nvidia_rate_limit_message(self):
        request = httpx.Request("POST", "https://integrate.api.nvidia.com/v1/embeddings")
        response = httpx.Response(429, request=request)
        error = RateLimitError(
            "Rate limit reached", response=response, body={"detail": "rate limited"},
        )
        mapped = main.nvidia_error_response(error)
        self.assertEqual(mapped.status_code, 429)
        self.assertIn("NVIDIA", mapped.detail)

    def test_upload_chat_citation_and_delete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (patch.object(db, "DATABASE_URL", ""),
                  patch.object(db, "DB_PATH", root / "test.db"),
                  patch.object(db, "UPLOAD_DIR", root / "uploads"),
                  patch.object(main, "UPLOAD_DIR", root / "uploads"),
                  patch.object(main, "embeddings", side_effect=lambda texts, input_type: [[1.0, 0.0] for _ in texts]),
                  patch.object(rag, "embeddings", side_effect=lambda texts, input_type: [[1.0, 0.0] for _ in texts])):
                with TestClient(main.app) as client:
                    pdf_bytes = sample_pdf()
                    self.assertEqual(client.get("/api/documents").status_code, 401)
                    signup = client.post("/api/auth/signup", json={"email": "first@example.com", "password": "strongpassword1"})
                    self.assertEqual(signup.status_code, 201, signup.text)
                    self.assertEqual(client.get("/api/auth/me").json()["email"], "first@example.com")
                    invalid = client.post("/api/documents", files={"file": ("bad.pdf", b"hello", "application/pdf")})
                    self.assertEqual(invalid.status_code, 400)

                    uploaded = client.post("/api/documents", files={"file": ("sample.pdf", pdf_bytes, "application/pdf")})
                    self.assertEqual(uploaded.status_code, 201, uploaded.text)
                    document = uploaded.json()
                    self.assertEqual(document["pages"], 1)
                    self.assertGreater(document["chunks"], 0)
                    duplicate = client.post("/api/documents", files={"file": ("sample.pdf", pdf_bytes, "application/pdf")})
                    self.assertEqual(duplicate.status_code, 409)

                    def fake_answer(question, history, sources):
                        self.assertEqual(sources[0]["page"], 1)
                        return "June 15. [1]", [{
                            "number": 1, "document_id": sources[0]["document_id"],
                            "document_name": sources[0]["document_name"],
                            "page": sources[0]["page"], "snippet": sources[0]["text"][:320],
                        }]

                    with patch.object(main, "answer_question", side_effect=fake_answer):
                        chat = client.post("/api/chat", json={"question": "When is launch?", "document_ids": [document["id"]]})
                    self.assertEqual(chat.status_code, 200, chat.text)
                    self.assertEqual(chat.json()["citations"][0]["page"], 1)
                    conversation = client.get(f"/api/conversations/{chat.json()['conversation_id']}")
                    self.assertEqual(len(conversation.json()["messages"]), 2)
                    self.assertEqual(client.get(f"/api/documents/{document['id']}/file").status_code, 200)
                    with TestClient(main.app) as other:
                        self.assertEqual(other.post("/api/auth/signup", json={"email": "second@example.com", "password": "strongpassword2"}).status_code, 201)
                        self.assertEqual(other.get("/api/documents").json(), [])
                        self.assertEqual(other.get("/api/conversations").json(), [])
                        self.assertEqual(other.get(f"/api/documents/{document['id']}/file").status_code, 404)
                        self.assertEqual(other.get(f"/api/conversations/{chat.json()['conversation_id']}").status_code, 404)
                        self.assertEqual(other.delete(f"/api/documents/{document['id']}").status_code, 404)
                        self.assertEqual(other.post("/api/chat", json={"question": "When is launch?", "document_ids": [document["id"]]}).status_code, 400)
                        same_pdf = other.post("/api/documents", files={"file": ("sample.pdf", pdf_bytes, "application/pdf")})
                        self.assertEqual(same_pdf.status_code, 201, same_pdf.text)
                        self.assertEqual(other.post("/api/chat", json={"question": "When is launch?", "document_ids": [document["id"]]}).status_code, 400)
                        with patch.object(main, "answer_question", side_effect=lambda question, history, sources: ("June 15.", [])) as answer_mock:
                            other_chat = other.post("/api/chat", json={"question": "When is launch?"})
                        self.assertEqual(other_chat.status_code, 200, other_chat.text)
                        self.assertEqual({source["document_id"] for source in answer_mock.call_args.args[2]}, {same_pdf.json()["id"]})
                    self.assertEqual(client.delete(f"/api/documents/{document['id']}").status_code, 204)
                    self.assertEqual(client.get("/api/documents").json(), [])
                    self.assertEqual(client.get(f"/api/documents/{document['id']}/file").status_code, 404)
                    self.assertFalse((root / "uploads" / f"{document['id']}.pdf").exists())
                    with db.connection() as connection:
                        remaining = connection.execute(
                            "SELECT COUNT(*) AS total FROM chunks WHERE document_id = ?", (document["id"],)
                        ).fetchone()["total"]
                    self.assertEqual(remaining, 0)
                    self.assertEqual(client.post("/api/auth/signout").status_code, 204)
                    self.assertEqual(client.get("/api/documents").status_code, 401)
                    self.assertEqual(client.post("/api/auth/signin", json={"email": "first@example.com", "password": "wrongpassword"}).status_code, 401)
                    self.assertEqual(client.post("/api/auth/signin", json={"email": "first@example.com", "password": "strongpassword1"}).status_code, 200)
                    self.assertEqual(client.get(f"/api/conversations/{chat.json()['conversation_id']}").status_code, 200)


if __name__ == "__main__":
    unittest.main()
