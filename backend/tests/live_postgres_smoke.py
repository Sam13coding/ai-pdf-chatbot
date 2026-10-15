"""Run an end-to-end test with synthetic content against a running local API."""

import os
import uuid

import httpx
import pymupdf


def main():
    base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8001")
    marker = uuid.uuid4().hex[:10]
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text(
        (72, 72),
        f"Synthetic project {marker} launches on June 15. Its project manager is Maya.",
    )
    page.insert_text((72, 100), "GitHub profile", fontsize=12)
    expected_url = f"https://github.com/example/{marker}"
    page.insert_link({
        "kind": pymupdf.LINK_URI,
        "from": pymupdf.Rect(70, 86, 155, 106),
        "uri": expected_url,
    })
    content = pdf.tobytes()
    pdf.close()

    document_id = None
    conversation_id = None
    email = os.getenv("SMOKE_TEST_EMAIL")
    password = os.getenv("SMOKE_TEST_PASSWORD")
    if not email or not password:
        raise SystemExit("Set SMOKE_TEST_EMAIL and SMOKE_TEST_PASSWORD to an existing test account.")
    with httpx.Client(timeout=150) as client:
        try:
            client.post(
                f"{base_url}/api/auth/signin",
                json={"email": email, "password": password},
            ).raise_for_status()
            upload = client.post(
                f"{base_url}/api/documents",
                files={"file": (f"postgres-smoke-{marker}.pdf", content, "application/pdf")},
            )
            upload.raise_for_status()
            document_id = upload.json()["id"]

            chat = client.post(
                f"{base_url}/api/chat",
                json={"question": "What is the exact GitHub profile link?", "document_ids": [document_id]},
            )
            chat.raise_for_status()
            result = chat.json()
            conversation_id = result["conversation_id"]
            if expected_url not in result["answer"]:
                raise RuntimeError(f"Unexpected answer: {result['answer']}")
            if not result["citations"] or result["citations"][0]["page"] != 1:
                raise RuntimeError("The answer did not include the expected page 1 citation.")
            print("Synthetic PDF upload: passed")
            print("PostgreSQL pgvector retrieval: passed")
            print("NVIDIA exact GitHub-link answer with page citation: passed")
        finally:
            if conversation_id:
                client.delete(f"{base_url}/api/conversations/{conversation_id}").raise_for_status()
            if document_id:
                client.delete(f"{base_url}/api/documents/{document_id}").raise_for_status()
            print("Synthetic test records removed")


if __name__ == "__main__":
    main()
