import json
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app import rag
from app.chat_graph import run_chat
from app.guardrails import ABSTENTION
from app.retrieval import cosine, rank_sources


def chunk(identifier, text, vector):
    return {"id": identifier, "document_id": "doc", "document_name": "manual.pdf", "page": 1,
            "text": text, "embedding": json.dumps(vector)}


class RetrievalTest(unittest.TestCase):
    def test_exact_identifier_beats_semantically_similar_irrelevant_chunks(self):
        rows = [chunk(f"noise-{n}", f"General invoice and finance procedures section {n}", [1, 0]) for n in range(20)]
        rows.append(chunk("target", "Invoice INV-8271 has a total of 125 dollars.", [.4, .9]))
        result = rank_sources("What is the total of INV-8271?", rows, [1, 0], 4)
        self.assertEqual(result[0]["id"], "target")
        self.assertGreater(result[0]["keyword_score"], 0)

    def test_duplicate_evidence_does_not_consume_context_slots(self):
        rows = [chunk("one", "Annual leave is 20 days.", [1, 0]),
                chunk("duplicate", "Annual leave is 20 days.", [1, 0]),
                chunk("two", "A manager approves leave requests.", [.8, .2])]
        result = rank_sources("annual leave approvals", rows, [1, 0], 3)
        self.assertEqual(len(result), 2)
        self.assertIn("two", {row["id"] for row in result})

    def test_embedding_dimension_mismatch_is_reported(self):
        with self.assertRaisesRegex(ValueError, "dimensions changed"):
            cosine([1, 0], [1, 0, 0])

    def test_chat_retries_abstention_once_and_uses_standalone_followup(self):
        retrieve = MagicMock(return_value=[{"text": "evidence"}])
        generate = MagicMock(side_effect=[(ABSTENTION, []), ("June 15. [1]", [{"number": 1}])])
        rewrite = MagicMock(return_value="When is Project Atlas launching?")
        answer, citations = run_chat("When is it launching?", [{"role": "user", "content": "Project Atlas"}], ["doc"], "owner", retrieve, generate, rewrite)
        self.assertEqual(retrieve.call_count, 2)
        self.assertEqual(retrieve.call_args_list[0].args[0], "When is Project Atlas launching?")
        self.assertEqual(retrieve.call_args_list[1].kwargs["limit"], 12)
        self.assertEqual(generate.call_args.args[0], "When is Project Atlas launching?")
        self.assertEqual(answer, "June 15. [1]")

    def test_background_is_separate_and_cannot_supply_pdf_citations(self):
        source = {"document_id": "doc", "document_name": "manual.pdf", "page": 1, "text": "Leave is 20 days."}
        output = {"document_answer": "Leave is 20 days. [1]", "background_answer": "Leave policies vary by employer. [99]"}
        model = RunnableLambda(lambda _: AIMessage(content=json.dumps(output)))
        with patch.object(rag, "chat_model", return_value=model):
            answer, citations = rag.answer_with_background("Explain the leave policy", [], [source])
        self.assertIn("Related background", answer)
        self.assertNotIn("[99]", answer)
        self.assertEqual([c["number"] for c in citations], [1])
        output["document_answer"] = "The salary is 999. [8]"
        with patch.object(rag, "chat_model", return_value=model):
            answer, citations = rag.answer_with_background("What is salary?", [], [source])
        self.assertIn(ABSTENTION, answer)
        self.assertEqual(citations, [])


if __name__ == "__main__":
    unittest.main()
