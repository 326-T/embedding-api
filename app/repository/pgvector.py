from datetime import datetime
from functools import lru_cache
from typing import List, Optional

import numpy as np
import psycopg
from fastapi import Depends
from pgvector.psycopg import register_vector
from psycopg import sql
from pydantic import BaseModel, BeforeValidator, ConfigDict
from typing_extensions import Annotated

from app.settings import Settings, get_settings


class InsertVector(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    category: str
    title: str
    text: str
    embedding: Annotated[
        np.ndarray, BeforeValidator(lambda v: np.array(v, dtype=np.float32))
    ]


class VectorRecord(BaseModel):
    id: int
    category: str
    title: str
    text: str
    vector_score: float
    text_score: float
    hibrid_score: float
    created_at: datetime


class PgVectorRepository:
    def __init__(self, db_string: str, table_name: str = "embeddings"):
        self.table_name = table_name
        self.conn = psycopg.connect(db_string)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        register_vector(self.conn)

    def create_table(self, vector_dim: int = 384):
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id SERIAL PRIMARY KEY,
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding VECTOR({vector_dim}) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        self.conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_documents_embedding
            ON {self.table_name} USING ivfflat (embedding vector_cosine_ops);
        """)
        self.conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_documents_category_trgm
            ON {self.table_name} USING gin (category gin_trgm_ops);
        """)
        self.conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_documents_title_trgm
            ON {self.table_name} USING gin (title gin_trgm_ops);
        """)
        self.conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_documents_text_trgm
            ON {self.table_name} USING gin (text gin_trgm_ops);
        """)

    def hybrid_search(
        self,
        embedding: np.ndarray,
        query: str,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> List[VectorRecord]:
        with self.conn.cursor() as cur:
            sql_query = sql.SQL("""
                WITH vector_search AS (
                    SELECT
                        id,
                        category,
                        title,
                        text,
                        created_at,
                        1 - (embedding <=> %s::vector) as vector_score
                    FROM {}
                    WHERE embedding <=> %s::vector < 0.5
                    AND (%s::text IS NULL OR category = %s)
                    ORDER BY embedding <=> %s::vector
                    LIMIT 100
                ),
                text_search AS (
                    SELECT
                        id,
                        category,
                        title,
                        text,
                        created_at,
                        GREATEST(
                            -- 完全一致の場合は高いスコア
                            CASE
                                WHEN title ILIKE '%%' || %s || '%%' THEN 1.0
                                WHEN text ILIKE '%%' || %s || '%%' THEN 1.0
                                ELSE 0.0
                            END,
                            -- pg_trgmの類似度も使用（fallback）
                            (
                                similarity(title, %s) * 0.6 +
                                similarity(text, %s) * 0.4
                            )
                        ) as text_score
                    FROM {}
                    WHERE (%s::text IS NULL OR category = %s)
                )
                SELECT
                    COALESCE(v.id, t.id) as id,
                    COALESCE(v.category, t.category) as category,
                    COALESCE(v.title, t.title) as title,
                    COALESCE(v.text, t.text) as text,
                    COALESCE(v.vector_score, 0) as vector_score,
                    COALESCE(t.text_score, 0) as text_score,
                    (
                        COALESCE(v.vector_score, 0) * 0.7 +
                        COALESCE(t.text_score, 0) * 0.3
                    ) as hybrid_score,
                    COALESCE(v.created_at, t.created_at) as created_at
                FROM vector_search v
                FULL OUTER JOIN text_search t ON v.id = t.id
                WHERE (COALESCE(v.vector_score, 0) + COALESCE(t.text_score, 0)) > 0
                ORDER BY hybrid_score DESC
                LIMIT %s;
            """).format(
                sql.Identifier(self.table_name), sql.Identifier(self.table_name)
            )

            cur.execute(
                query=sql_query,
                params=(
                    embedding,
                    embedding,
                    category,
                    category,
                    embedding,
                    query,
                    query,
                    query,
                    query,
                    category,
                    category,
                    limit,
                ),
            )

            return [
                VectorRecord(
                    id=row[0],
                    category=row[1],
                    title=row[2],
                    text=row[3],
                    vector_score=row[4],
                    text_score=row[5],
                    hibrid_score=row[6],
                    created_at=row[7],
                )
                for row in cur.fetchall()
            ]

    def copy(self, items: List[InsertVector]):
        with self.conn.cursor() as cur:
            with cur.copy(f"""
                COPY {self.table_name} (
                    category,
                    title,
                    text,
                    embedding
                )
                FROM STDIN;
            """) as copy:
                for item in items:
                    copy.write_row(
                        (item.category, item.title, item.text, item.embedding)
                    )
            self.conn.commit()


@lru_cache(maxsize=1)
def get_pgvector_repository(
    settings: Annotated[Settings, Depends(get_settings)],
) -> PgVectorRepository:
    repo = PgVectorRepository(settings.connection_string)
    repo.create_table()
    return repo
