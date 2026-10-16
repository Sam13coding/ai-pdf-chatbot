"""Copy the local SQLite data into the configured PostgreSQL database."""

import json
import sqlite3

from app.config import DB_PATH, EMBEDDING_DIMENSIONS
from app.db import connection, init_db, using_postgres


def rows(source, query: str):
    return [dict(row) for row in source.execute(query).fetchall()]


def main():
    if not using_postgres():
        raise SystemExit("Set DATABASE_URL to PostgreSQL before running this migration.")
    if not DB_PATH.is_file():
        print("No SQLite database exists; PostgreSQL schema initialized with no data.")
        init_db()
        return

    source = sqlite3.connect(DB_PATH)
    source.row_factory = sqlite3.Row
    try:
        tables = {row["name"] for row in source.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        users = rows(source, "SELECT * FROM users") if "users" in tables else []
        documents = rows(source, "SELECT * FROM documents")
        chunks = rows(source, "SELECT * FROM chunks")
        conversations = rows(source, "SELECT * FROM conversations")
        messages = rows(source, "SELECT * FROM messages")
        workflows = rows(source, "SELECT * FROM workflows") if "workflows" in tables else []
        reports = rows(source, "SELECT * FROM reports") if "reports" in tables else []
    finally:
        source.close()

    invalid = [chunk["id"] for chunk in chunks if len(json.loads(chunk["embedding"])) != EMBEDDING_DIMENSIONS]
    if invalid:
        raise SystemExit("SQLite contains incompatible vectors. Re-upload those PDFs before migration.")

    init_db()
    with connection() as target:
        target.executemany(
            "INSERT INTO users (id, email, password_hash, created_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (id) DO NOTHING",
            ((row["id"], row["email"], row["password_hash"], row["created_at"]) for row in users),
        )
        target.executemany(
            "INSERT INTO documents (id, name, sha256, pages, chunks, embedding_model, created_at, owner_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (id) DO NOTHING",
            ((row["id"], row["name"], row["sha256"], row["pages"], row["chunks"],
              row["embedding_model"], row["created_at"], row.get("owner_id")) for row in documents),
        )
        target.executemany(
            "INSERT INTO chunks (id, document_id, page, text, embedding) "
            "VALUES (?, ?, ?, ?, ?::vector) ON CONFLICT (id) DO NOTHING",
            ((row["id"], row["document_id"], row["page"], row["text"], row["embedding"])
             for row in chunks),
        )
        target.executemany(
            "INSERT INTO conversations (id, title, created_at, owner_id) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (id) DO NOTHING",
            ((row["id"], row["title"], row["created_at"], row.get("owner_id")) for row in conversations),
        )
        target.executemany(
            "INSERT INTO messages (id, conversation_id, role, content, citations, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (id) DO NOTHING",
            ((row["id"], row["conversation_id"], row["role"], row["content"],
              row["citations"], row["created_at"]) for row in messages),
        )
        target.executemany(
            "INSERT INTO workflows (id, owner_id, kind, goal, document_ids, status, created_at, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (id) DO NOTHING",
            ((row["id"], row["owner_id"], row["kind"], row["goal"], row["document_ids"],
              "failed" if row["status"] in {"running", "pending"} else row["status"], row["created_at"],
              "Interrupted by database migration. Retry from checkpoint." if row["status"] in {"running", "pending"} else row["error"])
             for row in workflows),
        )
        target.executemany(
            "INSERT INTO reports (id, owner_id, content, created_at) VALUES (?, ?, ?, ?) ON CONFLICT (id) DO NOTHING",
            ((row["id"], row["owner_id"], row["content"], row["created_at"]) for row in reports),
        )

    print(
        f"Migrated {len(documents)} documents, {len(chunks)} chunks, "
        f"{len(conversations)} conversations, {len(messages)} messages, "
        f"{len(workflows)} workflows, and {len(reports)} reports. Keep workflow-checkpoints.sqlite alongside the data."
    )


if __name__ == "__main__":
    main()
