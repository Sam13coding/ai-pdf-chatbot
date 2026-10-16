"""Evaluate real model answers against synthetic, fixed PDF contexts."""

import argparse
import json
import re
import time
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from app import db, rag
from app.config import EMBEDDING_MODEL
from app.guardrails import ABSTENTION
from app.prompts import PROMPT_VERSION
from app.rag import answer_question


def score_answer(case: dict, answer: str, citations: list[dict]) -> dict:
    abstained = answer.strip() == ABSTENTION
    markers = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    source_pages = {i: source["page"] for i, source in enumerate(case["sources"], 1)}
    citation_numbers = {citation["number"] for citation in citations}
    valid_citations = (
        markers == citation_numbers
        and markers.issubset(source_pages)
        and all(source_pages.get(c["number"]) == c["page"] for c in citations)
    )
    expected = case["expect_abstention"]
    content_ok = (
        abstained if expected else
        not abstained and all(phrase.casefold() in answer.casefold() for phrase in case["expected_phrases"])
    )
    citations_ok = valid_citations and (
        not markers and not citations if expected else
        bool(markers) and set(case["expected_pages"]).issubset({c["page"] for c in citations})
    )
    forbidden_ok = not any(
        phrase.casefold() in answer.casefold() for phrase in case.get("forbidden_phrases", [])
    )
    return {
        "content_ok": content_ok, "citations_ok": citations_ok,
        "forbidden_ok": forbidden_ok, "abstained": abstained,
        "passed": content_ok and citations_ok and forbidden_ok,
    }


def run(cases: list[dict], answer_fn=answer_question, source_fn=None) -> dict:
    results = []
    for case in cases:
        sources = [
            {**source, "document_id": "evaluation-document", "document_name": "evaluation.pdf"}
            for source in case["sources"]
        ]
        started = time.perf_counter()
        try:
            if source_fn:
                sources = source_fn(case)
            answer, citations = answer_fn(case["question"], case.get("history", []), sources)
            scoring_case = {**case, "sources": sources} if source_fn else case
            result = {"id": case["id"], **score_answer(scoring_case, answer, citations),
                      "answer": answer, "citations": citations}
            if source_fn:
                expected_pages = set(case["expected_pages"])
                retrieved_pages = {source["page"] for source in sources}
                result["retrieval_recall"] = len(expected_pages & retrieved_pages) / len(expected_pages) if expected_pages else None
                result["retrieved_pages"] = sorted(retrieved_pages)
                result["passed"] = result["passed"] and expected_pages.issubset(retrieved_pages)
        except Exception as exc:
            # Do not put provider errors (which may contain request data) in reports.
            result = {"id": case["id"], "passed": False, "error_type": type(exc).__name__}
        result["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
        results.append(result)
    passed = sum(result["passed"] for result in results)
    return {"total": len(results), "passed": passed,
            "pass_rate": passed / len(results) if results else 0, "results": results}


def run_with_retrieval(cases):
    """Use real NVIDIA embeddings in an isolated temporary SQLite database."""
    with ExitStack() as stack:
        root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        stack.enter_context(patch.object(db, "DATABASE_URL", ""))
        stack.enter_context(patch.object(db, "DB_PATH", root / "evaluation.db"))
        stack.enter_context(patch.object(db, "UPLOAD_DIR", root / "uploads"))
        db.init_db()
        passages = [(case, index, source) for case in cases for index, source in enumerate(case["sources"])]
        distractors = [
            "The kitchen coffee machine is cleaned every Friday.",
            "The building has four visitor parking spaces.",
            "The archived supplier catalog contains discontinued products.",
            "Shipping labels must be printed using the warehouse printer.",
            "The office garden is maintained by the facilities team.",
            "The cafeteria serves lunch at noon.",
            "Backup batteries are stored in the maintenance cupboard.",
            "Conference room bookings require a meeting title.",
        ]
        vectors = rag.embeddings([source["text"] for _, _, source in passages] + distractors, "passage")
        with db.connection() as conn:
            conn.execute("INSERT INTO users (id, email, password_hash) VALUES ('eval-owner', 'evaluation@example.com', 'not-a-login')")
            for case in cases:
                conn.execute("INSERT INTO documents (id, name, sha256, pages, chunks, embedding_model, owner_id) VALUES (?, 'evaluation.pdf', ?, 100, ?, ?, 'eval-owner')",
                             (case["id"], case["id"], len(case["sources"]) + len(distractors), EMBEDDING_MODEL))
            for (case, index, source), vector in zip(passages, vectors):
                conn.execute("INSERT INTO chunks (id, document_id, page, text, embedding) VALUES (?, ?, ?, ?, ?)",
                             (f"{case['id']}-{index}", case["id"], source["page"], source["text"], json.dumps(vector)))
            for case in cases:
                # Empty-context case remains genuinely empty.
                if not case["sources"]:
                    continue
                for index, (text, vector) in enumerate(zip(distractors, vectors[len(passages):])):
                    conn.execute("INSERT INTO chunks (id, document_id, page, text, embedding) VALUES (?, ?, ?, ?, ?)",
                                 (f"noise-{case['id']}-{index}", case["id"], 90 + index, text, json.dumps(vector)))

        def source_fn(case):
            query = rag.standalone_question(case["question"], case.get("history", []))
            return rag.retrieve(query, [case["id"]], "eval-owner", limit=6)

        return run(cases, source_fn=source_fn)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "evals/knowledge_agent.json")
    parser.add_argument("--output", type=Path, default=Path("../output/evals/report.json"))
    parser.add_argument("--with-retrieval", action="store_true", help="Also exercise real embeddings, hybrid retrieval, and follow-up rewriting against isolated synthetic documents.")
    args = parser.parse_args()
    cases = json.loads(args.dataset.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        parser.error("The dataset must contain at least one evaluation case.")
    report = run_with_retrieval(cases) if args.with_retrieval else run(cases)
    report["mode"] = "hybrid-retrieval-and-generation" if args.with_retrieval else "fixed-context-generation"
    report["prompt_version"] = PROMPT_VERSION
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Passed {report['passed']}/{report['total']} cases. Report: {args.output}")
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
