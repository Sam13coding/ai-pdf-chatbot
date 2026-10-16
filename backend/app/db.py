import sqlite3
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

from .config import DATABASE_URL, DB_PATH, EMBEDDING_DIMENSIONS, UPLOAD_DIR


def using_postgres() -> bool:
    return DATABASE_URL.startswith(("postgresql://", "postgres://"))


class DatabaseConnection:
    def __init__(self, connection, postgres: bool):
        self.raw = connection
        self.postgres = postgres

    def _sql(self, statement: str) -> str:
        return statement.replace("?", "%s") if self.postgres else statement

    def execute(self, statement: str, params=()):
        return self.raw.execute(self._sql(statement), params)

    def executemany(self, statement: str, params):
        if self.postgres:
            cursor = self.raw.cursor()
            cursor.executemany(self._sql(statement), params)
            return cursor
        return self.raw.executemany(self._sql(statement), params)


def init_db():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if using_postgres():
        statements = [
            "CREATE EXTENSION IF NOT EXISTS vector",
            """CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at TIMESTAMPTZ NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE,
                pages INTEGER NOT NULL, chunks INTEGER NOT NULL,
                embedding_model TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""",
            f"""CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                page INTEGER NOT NULL, text TEXT NOT NULL,
                embedding vector({EMBEDDING_DIMENSIONS}) NOT NULL
            )""",
            "CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)",
            """CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY, title TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL, content TEXT NOT NULL,
                citations TEXT NOT NULL DEFAULT '[]',
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""",
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)",
        ]
        with connection() as db:
            for statement in statements:
                db.execute(statement)
            db.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS owner_id TEXT REFERENCES users(id)")
            db.execute("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS owner_id TEXT REFERENCES users(id)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_documents_owner ON documents(owner_id)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_conversations_owner ON conversations(owner_id)")
            init_workflow_tables(db)
        return

    with connection() as db:
        db.raw.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE,
                pages INTEGER NOT NULL, chunks INTEGER NOT NULL,
                embedding_model TEXT NOT NULL DEFAULT 'legacy',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                page INTEGER NOT NULL, text TEXT NOT NULL, embedding TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY, title TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL, content TEXT NOT NULL,
                citations TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
        """)
        columns = {row["name"] for row in db.execute("PRAGMA table_info(documents)")}
        if "embedding_model" not in columns:
            db.execute("ALTER TABLE documents ADD COLUMN embedding_model TEXT NOT NULL DEFAULT 'legacy'")
        if "owner_id" not in columns:
            db.execute("ALTER TABLE documents ADD COLUMN owner_id TEXT REFERENCES users(id)")
        columns = {row["name"] for row in db.execute("PRAGMA table_info(conversations)")}
        if "owner_id" not in columns:
            db.execute("ALTER TABLE conversations ADD COLUMN owner_id TEXT REFERENCES users(id)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_documents_owner ON documents(owner_id)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_conversations_owner ON conversations(owner_id)")
        init_workflow_tables(db)


def init_workflow_tables(db):
    db.execute("""CREATE TABLE IF NOT EXISTS workflows (
        id TEXT PRIMARY KEY, owner_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        kind TEXT NOT NULL, goal TEXT NOT NULL, document_ids TEXT NOT NULL,
        status TEXT NOT NULL, created_at TEXT NOT NULL, lease_until DOUBLE PRECISION NOT NULL DEFAULT 0,
        run_token TEXT, error TEXT
    )""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_workflows_owner ON workflows(owner_id)")
    db.execute("""CREATE TABLE IF NOT EXISTS reports (
        id TEXT PRIMARY KEY REFERENCES workflows(id) ON DELETE CASCADE,
        owner_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        content TEXT NOT NULL, created_at TEXT NOT NULL
    )""")


@contextmanager
def connection():
    if using_postgres():
        raw = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    else:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        raw = sqlite3.connect(DB_PATH)
        raw.row_factory = sqlite3.Row
        raw.execute("PRAGMA foreign_keys = ON")
    db = DatabaseConnection(raw, using_postgres())
    try:
        yield db
        raw.commit()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()
