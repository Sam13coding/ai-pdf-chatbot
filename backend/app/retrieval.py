"""Hybrid semantic/keyword retrieval with rank fusion and evidence diversity."""

import json
import math
import re
from collections import Counter

from .config import EMBEDDING_MODEL
from .db import connection, using_postgres

STOPWORDS = set("a an the is are was were be been being of to in on at for from and or with this that it what which who when where how why do does did my me please tell about document pdf documents".split())


def tokens(text):
    return [word for word in re.findall(r"[^\W_]+", text.casefold()) if word not in STOPWORDS]


def cosine(a, b):
    if len(a) != len(b):
        raise ValueError("Embedding dimensions changed. Delete and upload the affected PDFs again.")
    norm = math.sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / norm if norm else 0.0


def keyword_scores(query, rows):
    query_terms = set(tokens(query))
    counts = [Counter(tokens(row["document_name"] + " " + row["text"])) for row in rows]
    lengths = [sum(count.values()) for count in counts]
    average = sum(lengths) / len(lengths) if lengths else 1
    frequencies = Counter(term for count in counts for term in query_terms if term in count)
    scores = {}
    for row, count, length in zip(rows, counts, lengths):
        score = 0.0
        for term in query_terms:
            freq = count[term]
            if freq:
                idf = math.log(1 + (len(rows) - frequencies[term] + .5) / (frequencies[term] + .5))
                score += idf * freq * 2.5 / (freq + 1.5 * (.25 + .75 * length / (average or 1)))
        scores[row["id"]] = score
    return scores


def rank_sources(query, rows, query_vector, limit):
    if not rows:
        return []
    lexical = keyword_scores(query, rows)
    semantic = {row["id"]: float(row["semantic_score"]) if "semantic_score" in row else
                cosine(query_vector, json.loads(row["embedding"])) for row in rows}
    by_id = {row["id"]: row for row in rows}
    fused = {key: 0.0 for key in by_id}
    for ranking in [sorted(semantic, key=semantic.get, reverse=True),
                    sorted((key for key in lexical if lexical[key] > 0), key=lexical.get, reverse=True)]:
        for rank, key in enumerate(ranking, 1):
            fused[key] += 1 / (60 + rank)
    candidates = sorted(by_id, key=fused.get, reverse=True)[:80]
    selected = []
    term_sets = {key: set(tokens(row["text"])) for key, row in by_id.items()}
    while candidates and len(selected) < limit:
        def utility(key):
            overlaps = [len(term_sets[key] & term_sets[other]) / max(1, len(term_sets[key] | term_sets[other])) for other in selected]
            return fused[key] - .012 * max(overlaps, default=0)
        best = max(candidates, key=utility)
        candidates.remove(best)
        if any(by_id[best]["text"].strip() == by_id[other]["text"].strip() for other in selected):
            continue
        selected.append(best)
    return [{"id": key, "document_id": by_id[key]["document_id"], "document_name": by_id[key]["document_name"],
             "page": by_id[key]["page"], "text": by_id[key]["text"], "score": fused[key],
             "semantic_score": semantic[key], "keyword_score": lexical[key]} for key in selected]


def hybrid_retrieve(query, document_ids, owner_id, embedding_fn, limit=8):
    limit = max(1, min(limit, 16))
    condition = "d.owner_id = ?"
    params = [owner_id]
    if document_ids:
        condition += " AND d.id IN (" + ",".join("?" for _ in document_ids) + ")"
        params.extend(document_ids)
    with connection() as db:
        models = db.execute(f"SELECT DISTINCT d.embedding_model FROM documents d WHERE {condition}", params).fetchall()
        if any(row["embedding_model"] != EMBEDDING_MODEL for row in models):
            raise ValueError("Some PDFs use an older embedding model. Delete and upload them again before chatting.")
        if not models:
            return []
        vector = embedding_fn([query], "query")[0]
        columns = "c.id, c.document_id, d.name AS document_name, c.page, c.text"
        if using_postgres():
            semantic = db.execute(
                f"SELECT {columns}, 1 - (c.embedding <=> ?::vector) AS semantic_score "
                f"FROM chunks c JOIN documents d ON d.id = c.document_id WHERE {condition} "
                "ORDER BY c.embedding <=> ?::vector LIMIT 40", [json.dumps(vector), *params, json.dumps(vector)],
            ).fetchall()
            # OR query preserves exact names/IDs even when a long question includes unmatched terms.
            terms = list(dict.fromkeys(tokens(query)))[:40]
            lexical = []
            if terms:
                tsquery = " | ".join(terms)
                lexical = db.execute(
                    f"SELECT {columns}, 1 - (c.embedding <=> ?::vector) AS semantic_score "
                    f"FROM chunks c JOIN documents d ON d.id = c.document_id WHERE {condition} "
                    "AND to_tsvector('simple', c.text) @@ to_tsquery('simple', ?) "
                    "ORDER BY ts_rank_cd(to_tsvector('simple', c.text), to_tsquery('simple', ?)) DESC LIMIT 40",
                    [json.dumps(vector), *params, tsquery, tsquery],
                ).fetchall()
            rows = list({row["id"]: dict(row) for row in [*semantic, *lexical]}.values())
        else:
            rows = [dict(row) for row in db.execute(
                f"SELECT {columns}, c.embedding FROM chunks c JOIN documents d ON d.id = c.document_id WHERE {condition}", params,
            )]
    return rank_sources(query, rows, vector, limit)
