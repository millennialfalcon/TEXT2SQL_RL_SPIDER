import spider_env as env

def build_training_records(samples:list[env.SpiderSample]) -> list[dict]: 
    return [
        {"sample_id" : i, 
        "prompt" : env.create_prompt(sample)} for i, sample in enumerate(samples)
    ]

def _local_oai_completions(sample:env.SpiderSample, n:int) -> list[str]: 
    return env._local_call_oai(sample, n=n)

def reward_completions(completions:list[str], sample_ids:list[int], sample_lookup:dict[int, env.SpiderSample]) -> list[float]:
    return [env.score_sql_candidate(sample_lookup[sample_id], env.extract_sql(completion)) for completion, sample_id in zip(completions, sample_ids)]


if __name__ == "__main__":
    samples = env.load_spider_samples()[:2]
    sample_lookup = {i:sample for i,sample in enumerate(samples)}

    records = build_training_records(samples)
    for record in records: 
        sample_id = record["sample_id"]
        sample = sample_lookup[sample_id]
        
        candidates = _local_oai_completions(sample, n = 2)
        sample_ids = [sample_id] * len(candidates)
        rewards = reward_completions(candidates, sample_ids, sample_lookup)
        
        print(record["prompt"][:200])
        print(list(zip(candidates, rewards)))