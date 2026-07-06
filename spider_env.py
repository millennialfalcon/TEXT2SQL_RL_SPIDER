import json
import pandas as pd 
import sqlite3
from collections import Counter
from contextlib import closing
from dataclasses import dataclass 
from openai import OpenAI
from pathlib import Path
from textwrap import dedent

#Custom Classes
@dataclass
class SpiderSample: 
    question: str
    gold_query:str 
    db_id:str
    db_path:Path 

@dataclass
class QueryResult: 
    ok: bool
    rows: list[tuple]
    error: str | None = None

#Helper Functions
def load_spider_samples(spider_root:str = 'Source/spider_data', split = 'train') -> list[SpiderSample]:
    spider_root = Path(spider_root)
    if split == 'train': 
        fname = 'train_spider.json'
    else: 
        fname = 'dev.json'

    examples = json.loads((spider_root/fname).read_text())
    samples = []
    for ex in examples: 
        db_id=ex['db_id']
        samples.append(
            SpiderSample(
                question=ex['question'], 
                gold_query=ex['query'], 
                db_id = db_id, 
                db_path = spider_root / "database" / db_id / f"{db_id}.sqlite" 
            )
        )

    return samples

def spider_run_query(sample:SpiderSample) -> list[tuple]:
    db_path = sample.db_path
    query = sample.gold_query 
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri = True)) as conn: 
        cur = conn.execute(query)
        rows = cur.fetchall()
    
    return rows

def spider_run_query_df(sample) -> pd.DataFrame:
    db_path = sample.db_path
    query = sample.gold_query 
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri = True)) as conn: 
        cur = conn.execute(query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
    
    return pd.DataFrame(rows, columns = columns)

def get_db_table_names(db_path:Path) -> list[str]: 
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn: 
        columns = conn.execute(
            """
            SELECT name 
            FROM sqlite_master 
            WHERE type = 'table'
            AND name NOT LIKE 'sqlite_%'
            ORDER BY name; 
            """
        ).fetchall()

    return [x[0] for x in columns]

def get_table_schema(db_path:Path, tname:str) -> list[tuple]: 
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
        columns = conn.execute(
            f"""
            SELECT name, type
            FROM pragma_table_info('{tname}');
            """ 
        ).fetchall()

    return columns

def create_prompt(sample) -> str: 
    db_path = sample.db_path
    tables = {table:get_table_schema(db_path, table) for table in get_db_table_names(db_path)}

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

    Database schema:
    {db_prompt_info}

    Question:
    {sample.question}

    SQL:
    """).strip()
    
    return prompt

def execute_query(db_path:Path, query:str) -> QueryResult: 
    try: 
        with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn: 
            rows = conn.execute(query).fetchall()
        return QueryResult(ok=True, rows=rows)
    except Exception as e: 
        return QueryResult(ok=False, rows=[], error=str(e))

def does_order_matters(sql:str) -> bool:
    return 'order by' in sql.lower()

def normalize_value(value): 
    return value

def normalize_row(row:tuple): 
    return tuple(normalize_value(value) for value in row)

def rows_matched(gold_rows:list[tuple], candidate_rows:list[tuple], order_matters: bool) -> bool: 
    gold_rows = [normalize_row(row) for row in gold_rows]
    candidate_rows = [normalize_row(row) for row in candidate_rows]

    if order_matters: 
        return gold_rows == candidate_rows 
    
    return Counter(gold_rows) == Counter(candidate_rows)

def candidate_scorer(sample:SpiderSample, candidate_sql:str) -> float: 
    db_path = sample.db_path 
    gold_result = execute_query(db_path, sample.gold_query) 
    candidate_result = execute_query(db_path, candidate_sql)

    order_matters = does_order_matters(sample.gold_query)

    if not gold_result.ok: 
        raise ValueError("Gold query failed. ")
    
    if not candidate_result.ok: 
        return 0.0 

    if rows_matched(gold_result.rows, candidate_result.rows, order_matters): 
        return 1.0 

    return 0.2

def extract_sql(query:str) -> str:
    query = query.strip()
    if query.startswith("```"): 
        lines = query.splitlines()
        lines = [line for line in lines if not line.strip().startswith("```")]
        return "\n".join(lines).strip()
    else: 
        return query

def _local_call_oai(sample:SpiderSample, client:OpenAI | None = None, model:str = 'gpt-5.4-mini', n:int = 8): 
    if client is None:
        client = OpenAI()

    responses = []
    for _ in range(n): 
        response = client.responses.create(
            model = model, 
            input = create_prompt(sample),
            temperature = 0.2
        ).output[0].content[0].text
        responses.append(extract_sql(response))
    
    return responses

if __name__ == "__main__": 
    samples = load_spider_samples()[:5]
    for sample in samples: 
        oai_query = _local_call_oai(sample, n=1)[0]
        print(f"DB ID: {sample.db_id}")
        print(f"Gold Query:  {sample.gold_query}")
        print(f"OAI Query:  {oai_query}")
        print(f"Score: {candidate_scorer(sample, oai_query)}")
        print("------------------\n")
