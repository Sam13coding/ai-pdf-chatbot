"""A bounded retrieval/generation/validation loop for normal chat."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .guardrails import ABSTENTION


class ChatState(TypedDict, total=False):
    question: str
    search_query: str
    history: list[dict]
    document_ids: list[str]
    owner_id: str
    sources: list[dict]
    answer: str
    citations: list[dict]
    attempts: int


def run_chat(question, history, document_ids, owner_id, retrieve_fn, answer_fn, rewrite_fn=None):
    def search(state):
        attempt = state.get("attempts", 0) + 1
        # First use conversational context; retry the literal question with more evidence.
        query = state["search_query"] if attempt == 1 else f"{state['question']} {state['search_query']}"
        sources = retrieve_fn(query, state["document_ids"], state["owner_id"], limit=6 if attempt == 1 else 12)
        return {"sources": sources, "attempts": attempt}

    def generate(state):
        answer, citations = answer_fn(state["search_query"], state["history"], state["sources"])
        return {"answer": answer, "citations": citations}

    def validate(state):
        unsupported = state["answer"] == ABSTENTION or not state["citations"]
        return "search" if unsupported and state["attempts"] < 2 else END

    graph = StateGraph(ChatState)
    graph.add_node("search", search)
    graph.add_node("generate", generate)
    graph.add_edge(START, "search")
    graph.add_edge("search", "generate")
    graph.add_conditional_edges("generate", validate, ["search", END])
    graph = graph.compile()
    search_query = rewrite_fn(question, history) if rewrite_fn else question
    result = graph.invoke({
        "question": question, "history": history, "document_ids": document_ids, "owner_id": owner_id,
        "search_query": search_query,
    })
    return result["answer"], result["citations"]
