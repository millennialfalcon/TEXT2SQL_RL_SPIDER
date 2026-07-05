import sqlite3
import json
import pandas as pd 
from pathlib import Path
from dataclasses import dataclass 
from contextlib import closing


@dataclass
class SpiderSample: 
    question: str
    gold_query:str 
    db_id:str
    db_path:Path 

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
    prompt = f"""
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
"""
    
    return prompt

if __name__ == "__main__": 
    samples = load_spider_samples()
    prompt = create_prompt(samples[0])
    print(prompt)
