"""Spider data loading, prompt construction, SQL execution, and reward scoring."""

from __future__ import annotations
import json
import pandas as pd
import re
import sqlite3
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from openai import OpenAI
from pathlib import Path
from textwrap import dedent


# Custom Classes
@dataclass
class SpiderSample:
    """A Spider question paired with its reference query and SQLite database."""

    question: str
    gold_query: str
    db_id: str
    db_path: Path


@dataclass
class QueryResult:
    """The rows or error produced by executing a SQL query."""

    ok: bool
    rows: list[tuple]
    error: str | None = None


# Helper Functions
def load_spider_samples(
    spider_root: str = "Source/spider_data", split="train"
) -> list[SpiderSample]:
    """Load a Spider split and resolve each example's SQLite database path.

    Args:
        spider_root: Directory containing Spider JSON files and databases.
        split: ``"train"`` for ``train_spider.json``; any other value selects
            ``dev.json``.

    Returns:
        Samples in their source-file order.
    """
    spider_root = Path(spider_root)
    if split == "train":
        fname = "train_spider.json"
    else:
        fname = "dev.json"

    examples = json.loads((spider_root / fname).read_text())
    samples = []
    for ex in examples:
        db_id = ex["db_id"]
        samples.append(
            SpiderSample(
                question=ex["question"],
                gold_query=ex["query"],
                db_id=db_id,
                db_path=spider_root / "database" / db_id / f"{db_id}.sqlite",
            )
        )

    return samples


def spider_run_query(sample: SpiderSample) -> list[tuple]:
    """Execute a sample's reference query against its database in read-only mode."""
    db_path = sample.db_path
    query = sample.gold_query
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
        cur = conn.execute(query)
        rows = cur.fetchall()

    return rows


def spider_run_query_df(sample) -> pd.DataFrame:
    """Execute a sample's reference query and return a labeled DataFrame."""
    db_path = sample.db_path
    query = sample.gold_query
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
        cur = conn.execute(query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]

    return pd.DataFrame(rows, columns=columns)


def get_db_table_names(db_path: Path) -> list[str]:
    """Return user-defined table names from a SQLite database in sorted order."""
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
        columns = conn.execute("""
            SELECT name 
            FROM sqlite_master 
            WHERE type = 'table'
            AND name NOT LIKE 'sqlite_%'
            ORDER BY name; 
            """).fetchall()

    return [x[0] for x in columns]


def get_table_schema(db_path: Path, tname: str) -> list[tuple]:
    """Return ``(column_name, declared_type)`` pairs for a SQLite table."""
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
        columns = conn.execute(f"""
            SELECT name, type
            FROM pragma_table_info('{tname}');
            """).fetchall()

    return columns


def create_prompt(sample) -> str:
    """Build the schema-grounded text-to-SQL prompt for a Spider sample."""
    db_path = sample.db_path
    tables = {
        table: get_table_schema(db_path, table) for table in get_db_table_names(db_path)
    }

    packages = []
    for table in tables:
        package = f"Table: {table}\nColumns:\n"
        for column in tables[table]:
            package += f"  - {column[0]} {column[1]}\n"
        packages.append(package)

    db_prompt_info = "\n".join(packages)
    prompt = dedent(f"""
    You are writing SQLite SQL for a text-to-SQL task.

    Given the database schema and a natural language question, write one SQL query that answers the question.

    Rules:
    - Return only the SQL query.
    - Use SQLite syntax.
    - Do not include explanations.
    - Do not modify the database.
    - Use only the tables and columns listed in the schema.
    - Raw SQL is allowed.
    - Exactly one sql or sqlite fenced block is allowed.
    - Whitespace around the fence is allowed.
    - Other fence labels are not recognized.
    - Multiple recognized SQL fences are ambiguous and should be rejected.

    Database schema:
    {db_prompt_info}

    Question:
    {sample.question}

    SQL:
    """).strip()

    return prompt


def execute_query(db_path: Path, query: str) -> QueryResult:
    """Execute SQL against a read-only SQLite database without raising SQL errors.

    Returns:
        A successful result containing fetched rows, or a failed result containing
        the database error message.
    """
    try:
        with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
            # Preserve arbitrary SQLite TEXT bytes so malformed source encoding
            # cannot prevent otherwise valid Spider queries from being scored.
            conn.text_factory = bytes
            rows = conn.execute(query).fetchall()
        return QueryResult(ok=True, rows=rows)
    except Exception as e:
        return QueryResult(ok=False, rows=[], error=str(e))


def does_order_matters(sql: str) -> bool:
    """Return whether result-row order is semantically required by the SQL."""
    return "order by" in sql.lower()


def normalize_value(value):
    """Normalize a scalar query value before result comparison."""
    return value


def normalize_row(row: tuple):
    """Normalize every value in a query-result row."""
    return tuple(normalize_value(value) for value in row)


def rows_matched(
    gold_rows: list[tuple], candidate_rows: list[tuple], order_matters: bool
) -> bool:
    """Compare result rows while preserving duplicates and optional row order."""
    gold_rows = [normalize_row(row) for row in gold_rows]
    candidate_rows = [normalize_row(row) for row in candidate_rows]

    if order_matters:
        return gold_rows == candidate_rows

    return Counter(gold_rows) == Counter(candidate_rows)


def score_sql_candidate(
    sample: SpiderSample, candidate_sql: str, has_extra_text: bool
) -> float:
    """Score candidate SQL by execution result and output-format compliance.

    The reward is ``0.0`` for invalid SQL, ``0.2`` for executable SQL with an
    incorrect result, ``0.4`` for a correct result accompanied by extra text,
    and ``1.0`` for a clean correct result.

    Raises:
        ValueError: If the sample's reference query cannot be executed.
    """
    db_path = sample.db_path
    gold_result = execute_query(db_path, sample.gold_query)
    candidate_result = execute_query(db_path, candidate_sql)

    order_matters = does_order_matters(sample.gold_query)

    if not gold_result.ok:
        raise ValueError("Gold query failed. ")

    # four options 1) result is invalid == 0.0 2) no extra text and wrong query that executes == 0.2
    # 3) extra text but right query == 0.4 no extra text and correct query = 1.0
    if not candidate_result.ok:
        return 0.0

    query_match = rows_matched(gold_result.rows, candidate_result.rows, order_matters)
    if not query_match and candidate_result.ok:
        return 0.2

    if query_match and has_extra_text:
        return 0.4

    return 1.0


def extract_sql(query: str) -> dict:
    """Extract raw or fenced SQL and report completion-format metadata.

    Exactly one ``sql`` or ``sqlite`` fenced block is recognized. Raw text with
    no recognized fence is treated as SQL. Multiple recognized fences are
    marked ambiguous and left unextracted so that execution scoring rejects
    them.

    Returns:
        A dictionary containing ``query``, ``fence``, ``was_fenced``, and
        ``has_extra_text``.
    """
    query = query.strip()
    pat = r"```(?:sql|sqlite)[ \t]*\r?\n(?P<query>.*?)\r?\n[ \t]*```"
    m = [x for x in re.finditer(pat, query, flags=re.IGNORECASE | re.DOTALL)]

    if len(m) == 0:
        return {
            "query": query,
            "fence": "none",
            "was_fenced": False,
            "has_extra_text": False,
        }

    if len(m) > 1:
        return {
            "query": query,
            "fence": "multiple_fences",
            "was_fenced": True,
            "has_extra_text": True,
        }

    m = m[0]
    completion = m.group("query")
    before_completion = query[: m.start()]
    after_completion = query[m.end() :]

    if len(before_completion.strip()) > 0 or len(after_completion.strip()) > 0:
        has_extra_text = True
    else:
        has_extra_text = False

    return {
        "query": completion,
        "fence": "one",
        "was_fenced": True,
        "has_extra_text": has_extra_text,
    }


def _local_call_oai(
    sample: SpiderSample,
    *,
    client: OpenAI | None = None,
    model: str = "gpt-5.4-mini",
    n: int = 8,
):
    """Generate local debugging completions with the OpenAI Responses API.

    This helper is for local inspection only and is not used by GRPO training.
    """
    if client is None:
        client = OpenAI()

    responses = []
    for _ in range(n):
        response = (
            client.responses.create(
                model=model, input=create_prompt(sample), temperature=0.2
            )
            .output[0]
            .content[0]
            .text
        )
        responses.append(response)
    return responses


def create_rollout_group(sample: SpiderSample, candidate_sqls: list[str]):
    """Score candidate completions and package them with their Spider sample."""
    group = {
        "db_id": sample.db_id,
        "question": sample.question,
        "gold_query": sample.gold_query,
        "prompt": create_prompt(sample),
        "candidates": [],
    }

    for candidate_sql in candidate_sqls:
        extraction = extract_sql(candidate_sql)
        group["candidates"].append(
            {
                "candidate_sql": extraction["query"],
                "has_extra_text": extraction["has_extra_text"],
                "reward": score_sql_candidate(
                    sample, extraction["query"], extraction["has_extra_text"]
                ),
            }
        )
    return group


def summarize_rollout_groups(groups: list[dict]) -> dict:
    """Aggregate reward counts and mean reward across rollout groups."""
    rewards = [
        candidate["reward"] for group in groups for candidate in group["candidates"]
    ]

    if not rewards:
        return {
            "num_groups": len(groups),
            "num_candidates": 0,
            "mean_reward": 0,
            "num_exact": 0,
            "num_wrong": 0,
            "num_failed": 0,
            "num_right_extra_text": 0,
        }
    else:
        return {
            "num_groups": len(groups),
            "num_candidates": len(rewards),
            "mean_reward": sum(rewards) / len(rewards),
            "num_exact": sum(r == 1 for r in rewards),
            "num_wrong": sum(r == 0.2 for r in rewards),
            "num_right_extra_text": sum(r == 0.4 for r in rewards),
            "num_failed": sum(r == 0 for r in rewards),
        }


if __name__ == "__main__":
    samples = load_spider_samples()[:2]
    groups = []
    for sample in samples:
        oai_query = _local_call_oai(sample, n=5)
        groups.append(create_rollout_group(sample, oai_query))

    print(summarize_rollout_groups(groups))
