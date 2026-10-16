"""Document tools, sequential document chains, and a durable review graph."""

import json
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from .db import connection
from .guardrails import ABSTENTION
from .prompts import (EXTRACTION_SYSTEM, NOTES_SYSTEM, RESEARCH_SYSTEM,
                      REVIEW_SYSTEM, SUMMARY_SYSTEM, workflow_prompt)
from .rag import answer_question, chat_model, cited_answer, retrieve

MAX_CHUNKS = 24
MAX_TOOL_ROUNDS = 3


class SearchInput(BaseModel):
    query: str = Field(min_length=2, max_length=2000)


class PageInput(BaseModel):
    document_id: str = Field(min_length=1, max_length=100)
    page: int = Field(ge=1)


class ExtractedField(BaseModel):
    field: str = Field(min_length=1, max_length=200)
    value: str = Field(min_length=1, max_length=2000)
    sources: list[int] = Field(min_length=1, max_length=24)


class Extraction(BaseModel):
    records: list[ExtractedField] = Field(default_factory=list, max_length=50)


class Review(BaseModel):
    supported: bool
    feedback: str = Field(default="", max_length=2000)


class WorkflowState(TypedDict, total=False):
    id: str
    owner_id: str
    kind: Literal["research", "summary", "extract"]
    goal: str
    document_ids: list[str]
    sources: list[dict]
    messages: list
    tool_rounds: int
    cursor: int
    notes: list[str]
    draft: str
    citations: list[dict]
    records: list[dict]
    feedback: str
    revisions: int
    supported: bool
    approved: bool
    status: str
    steps: list[str]


def select_chunks(owner_id, document_ids, *, page=None, limit=MAX_CHUNKS + 1):
    marks = ",".join("?" for _ in document_ids)
    params = [owner_id, *document_ids]
    condition = f"d.owner_id = ? AND d.id IN ({marks})"
    if page is not None:
        condition += " AND c.page = ?"
        params.append(page)
    with connection() as db:
        return [dict(row) for row in db.execute(
            "SELECT c.id, c.document_id, d.name AS document_name, c.page, c.text "
            f"FROM chunks c JOIN documents d ON d.id = c.document_id WHERE {condition} "
            "ORDER BY d.id, c.page, c.id LIMIT ?", [*params, limit],
        )]


def document_tools(owner_id, document_ids):
    # Identity and document scope are closures, never model-supplied arguments.
    @tool(args_schema=SearchInput)
    def search_documents(query: str) -> str:
        """Search the user's selected PDFs for passages relevant to a question."""
        return json.dumps(retrieve(query, document_ids, owner_id, limit=4))

    @tool(args_schema=PageInput)
    def read_document_page(document_id: str, page: int) -> str:
        """Read a page of a selected PDF to inspect supporting evidence."""
        if document_id not in document_ids:
            return "[]"
        return json.dumps(select_chunks(owner_id, [document_id], page=page, limit=8))

    return {item.name: item for item in [search_documents, read_document_page]}


def source_context(sources):
    return "\n\n".join(
        f"[{i}] {s['document_name']}, page {s['page']}\n{s['text']}"
        for i, s in enumerate(sources, 1)
    )


def chain(system, *, structured=False):
    parser = JsonOutputParser() if structured else StrOutputParser()
    return workflow_prompt(system) | chat_model() | parser


def trace(state, name):
    return [*state.get("steps", []), name]


def build_graph(checkpointer, save_report):
    def prepare(state):
        if state["kind"] == "research":
            return {"sources": [], "messages": [
                SystemMessage(content=RESEARCH_SYSTEM),
                HumanMessage(content=state["goal"]),
            ], "tool_rounds": 0, "steps": trace(state, "Research prepared")}
        sources = select_chunks(state["owner_id"], state["document_ids"])
        if len(sources) > MAX_CHUNKS:
            raise ValueError(f"Summary/extraction supports up to {MAX_CHUNKS} chunks. Select fewer/smaller PDFs or use research.")
        if not sources:
            raise ValueError("The selected PDFs have no available text.")
        return {"sources": sources, "notes": [], "cursor": 0, "steps": trace(state, "Documents loaded")}

    def research(state):
        tools = document_tools(state["owner_id"], state["document_ids"])
        response = chat_model().bind_tools(list(tools.values()), parallel_tool_calls=False).invoke(state["messages"])
        return {"messages": [*state["messages"], response], "steps": trace(state, "Researcher selected tools")}

    def use_tools(state):
        tools = document_tools(state["owner_id"], state["document_ids"])
        messages = list(state["messages"])
        sources = list(state["sources"])
        for index, call in enumerate(messages[-1].tool_calls):
            result = []
            if index < 2 and call["name"] in tools:
                # Validation failures stop this run and can be inspected/retried, not executed unchecked.
                result = json.loads(tools[call["name"]].invoke(call["args"]))
            for source in result:
                if len(sources) < MAX_CHUNKS and not any(s["id"] == source["id"] for s in sources):
                    sources.append(source)
            messages.append(ToolMessage(content=json.dumps(result), tool_call_id=call["id"]))
        return {"sources": sources, "messages": messages, "tool_rounds": state["tool_rounds"] + 1,
                "steps": trace(state, "Read-only document tools executed")}

    def fallback(state):
        # Some compatible endpoints choose no tools. Always retrieve real evidence before writing.
        sources = state["sources"] or retrieve(state["goal"], state["document_ids"], state["owner_id"], limit=8)
        return {"sources": sources, "steps": trace(state, "Evidence gathered")}

    def map_document(state):
        start = state["cursor"]
        group = state["sources"][start:start + 4]
        context = "\n\n".join(
            f"[{start + i}] {s['document_name']}, page {s['page']}\n{s['text']}" for i, s in enumerate(group, 1)
        )
        notes = chain(NOTES_SYSTEM).invoke({
            "input": f"Goal: {state['goal']}\nEvidence:\n{context}",
        })
        return {"notes": [*state["notes"], notes], "cursor": start + len(group),
                "steps": trace(state, f"Processed passages {start + 1}-{start + len(group)}")}

    def write(state):
        if state["kind"] == "research":
            answer, citations = answer_question(
                f"Write a concise research report answering: {state['goal']}. "
                f"Explain evidence gaps. Reviewer feedback to consider: {state.get('feedback', '')}", [], state["sources"],
            )
            return {"draft": answer, "citations": citations, "records": [], "steps": trace(state, "Writer drafted report")}
        notes = "\n\n".join(state["notes"])
        source_index = "\n".join(f"[{i}] {s['document_name']}, page {s['page']}" for i, s in enumerate(state["sources"], 1))
        payload = {"input": f"Goal: {state['goal']}\nAllowed source numbers:\n{source_index}\nEvidence notes:\n{notes}\nReviewer feedback: {state.get('feedback', '')}"}
        if state["kind"] == "extract":
            output = chain(EXTRACTION_SYSTEM, structured=True).invoke(payload)
            records = Extraction.model_validate(output).model_dump()["records"]
            if any(n < 1 or n > len(state["sources"]) for r in records for n in r["sources"]):
                raise ValueError("Extraction returned invalid source references.")
            answer = "\n".join(f"{r['field']}: {r['value']} " + " ".join(f"[{n}]" for n in r["sources"]) for r in records)
        else:
            records = []
            answer = chain(SUMMARY_SYSTEM).invoke(payload)
        answer, citations = cited_answer(answer, state["sources"])
        return {"draft": answer, "citations": citations, "records": records, "steps": trace(state, "Writer drafted report")}

    def review(state):
        supported, feedback = False, "No supported, cited answer was produced."
        if state["draft"] != ABSTENTION and state["citations"]:
            output = chain(REVIEW_SYSTEM, structured=True).invoke({
                "input": f"Goal: {state['goal']}\nDraft:\n{state['draft']}\nOriginal evidence:\n{source_context(state['sources'])}",
            })
            result = Review.model_validate(output)
            supported, feedback = result.supported, result.feedback
        return {"supported": supported, "feedback": feedback, "revisions": state.get("revisions", 0) + 1,
                "steps": trace(state, "Reviewer checked evidence")}

    def human_review(state):
        approved = interrupt({"draft": state["draft"], "citations": state["citations"],
                              "action": "Save this report to your private workspace"})
        if not isinstance(approved, bool):
            raise ValueError("Approval must be a boolean.")
        return {"approved": approved, "steps": trace(state, "Human approved" if approved else "Human rejected")}

    def publish(state):
        if state["approved"]:
            save_report(state)  # Idempotent SQL insert, only reachable after interrupt approval.
        return {"status": "completed" if state["approved"] else "rejected", "steps": trace(state, "Workflow finished")}

    def blocked(state):
        return {"status": "blocked", "steps": trace(state, "Evidence check failed; report not saved")}

    graph = StateGraph(WorkflowState)
    for name, fn in [("prepare", prepare), ("researcher", research), ("tools", use_tools),
                     ("fallback", fallback), ("map", map_document), ("writer", write),
                     ("reviewer", review), ("human_review", human_review), ("publish", publish), ("blocked", blocked)]:
        graph.add_node(name, fn)
    graph.add_edge(START, "prepare")
    graph.add_conditional_edges("prepare", lambda s: "researcher" if s["kind"] == "research" else "map")
    graph.add_conditional_edges("researcher", lambda s: "tools" if s["messages"][-1].tool_calls else "fallback")
    graph.add_conditional_edges("tools", lambda s: "researcher" if s["tool_rounds"] < MAX_TOOL_ROUNDS else "fallback")
    graph.add_edge("fallback", "writer")
    graph.add_conditional_edges("map", lambda s: "map" if s["cursor"] < len(s["sources"]) else "writer")
    graph.add_edge("writer", "reviewer")
    graph.add_conditional_edges("reviewer", lambda s: "human_review" if s["supported"] else
                               "writer" if s["revisions"] < 2 else "blocked")
    graph.add_edge("human_review", "publish")
    graph.add_edge("publish", END)
    graph.add_edge("blocked", END)
    return graph.compile(checkpointer=checkpointer)
