import json
import re

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_openai import ChatOpenAI

from .config import CHAT_MODEL, EMBEDDING_MODEL, NVIDIA_API_KEY, NVIDIA_BASE_URL
from .retrieval import hybrid_retrieve
from .guardrails import ABSTENTION, check_answer
from .prompts import ANSWER_SYSTEM, BACKGROUND_SYSTEM, CITATION_REPAIR_SYSTEM, REWRITE_SYSTEM, input_json, prompt


class MissingNvidiaKeyError(Exception):
    pass


def require_nvidia_key() -> None:
    if not NVIDIA_API_KEY or NVIDIA_API_KEY == "your_nvidia_api_key_here":
        raise MissingNvidiaKeyError("Set NVIDIA_API_KEY in backend/.env and restart the API.")


def embedding_model() -> NVIDIAEmbeddings:
    require_nvidia_key()
    return NVIDIAEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=NVIDIA_BASE_URL,
        api_key=NVIDIA_API_KEY,
        truncate="END",
        max_batch_size=16,
    )


def embeddings(texts: list[str], input_type: str) -> list[list[float]]:
    if input_type not in {"passage", "query"}:
        raise ValueError("Embedding input_type must be passage or query.")
    if not texts:
        return []
    model = embedding_model()
    if input_type == "query":
        return [model.embed_query(texts[0])]
    return model.embed_documents(texts)


def retrieve(query: str, document_ids: list[str], owner_id: str, limit: int = 8) -> list[dict]:
    return hybrid_retrieve(query, document_ids, owner_id, embeddings, limit)


def answer_question(question: str, history: list[dict], sources: list[dict]) -> tuple[str, list[dict]]:
    if not sources:
        return ABSTENTION, []
    evidence = [{"number": i, "document_name": source["document_name"],
                 "page": source["page"], "text": source["text"]}
                for i, source in enumerate(sources, 1)]
    request = input_json(question=question, recent_history=history[-8:], pdf_passages=evidence)
    model = chat_model()
    response = (prompt(ANSWER_SYSTEM) | model).invoke({"input": request})
    answer = response.content if isinstance(response.content, str) else str(response.content)
    answer = answer.strip()
    if not answer:
        raise ValueError("NVIDIA returned an empty answer.")
    checked, _ = cited_answer(answer, sources)
    if checked == ABSTENTION and answer != ABSTENTION:
        # Ask the model to correct its citations using original evidence; never append one ourselves.
        response = (prompt(CITATION_REPAIR_SYSTEM) | model).invoke({"input": input_json(
            question=question, recent_history=history[-8:], pdf_passages=evidence, rejected_draft=answer,
        )})
        answer = response.content.strip() if isinstance(response.content, str) else ""
    return cited_answer(answer, sources)


def chat_model():
    require_nvidia_key()
    return ChatOpenAI(
        model=CHAT_MODEL,
        base_url=NVIDIA_BASE_URL,
        api_key=NVIDIA_API_KEY,
        temperature=0.2,
        max_completion_tokens=1800,
        timeout=90,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


def cited_answer(answer: str, sources: list[dict]) -> tuple[str, list[dict]]:
    answer, cited_numbers = check_answer(answer, len(sources))
    citations = [
        {"number": i, "document_id": source["document_id"],
         "document_name": source["document_name"], "page": source["page"],
         "snippet": source["text"][:320]}
        for i, source in enumerate(sources, start=1) if i in cited_numbers
    ]
    return answer, citations


def standalone_question(question: str, history: list[dict]) -> str:
    if not history:
        return question
    rewritten = (prompt(REWRITE_SYSTEM) | chat_model() | StrOutputParser()).invoke({
        "input": input_json(history=history[-6:], latest_question=question),
    }).strip()
    return rewritten[:2000] if rewritten else question


def answer_with_background(question: str, history: list[dict], sources: list[dict]) -> tuple[str, list[dict]]:
    if not sources:
        return ABSTENTION, []
    evidence = [{"number": i, "document_name": source["document_name"],
                 "page": source["page"], "text": source["text"]}
                for i, source in enumerate(sources, 1)]
    output = (prompt(BACKGROUND_SYSTEM) | chat_model() | JsonOutputParser()).invoke({
        "input": input_json(question=question, recent_history=history[-6:], pdf_passages=evidence),
    })
    if not isinstance(output, dict) or not all(isinstance(output.get(key), str) for key in ["document_answer", "background_answer"]):
        raise ValueError("The model returned an invalid answer format. Please try again.")
    answer, citations = cited_answer(output["document_answer"], sources)
    background = re.sub(r"\[\d+\]", "", output["background_answer"]).strip()
    if background:
        answer = f"From your PDFs\n{answer}\n\nRelated background (model knowledge, not verified from your PDFs or the web)\n{background}"
    return answer, citations
