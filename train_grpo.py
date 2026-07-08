from __future__ import annotations
import spider_env as env
from datasets import Dataset

def build_training_records(samples:list[env.SpiderSample]) -> list[dict]: 
    return [
        {"sample_id" : i, 
        "prompt" : env.create_prompt(sample)} for i, sample in enumerate(samples)
    ]

def build_training_dataset(samples:list[env.SpiderSample]) -> Dataset: 
    return Dataset.from_list(build_training_records(samples))

def _local_oai_completions(sample:env.SpiderSample, n:int) -> list[str]: 
    return env._local_call_oai(sample, n=n)

def reward_completions(completions:list[str], sample_ids:list[int], sample_lookup:dict[int, env.SpiderSample]) -> list[float]:
    return [env.score_sql_candidate(sample_lookup[sample_id], env.extract_sql(completion)) for completion, sample_id in zip(completions, sample_ids)]

def build_spider_reward_function(sample_lookup:dict[int, env.SpiderSample]):
    def spider_reward_function(prompts:list[str], completions:list[str], sample_id:list[int], **kwargs) -> list[float]:
        sample_ids = sample_id 
        
        if len(completions) != len(sample_ids):
            raise ValueError("Different numbers of Completions and Sample IDs. ")
        
        rewards = []
        for completion, sid in zip(completions, sample_ids): 
            sample = sample_lookup[sid]
            query = env.extract_sql(completion)
            rewards.append(env.score_sql_candidate(sample, query))

        return rewards
    return spider_reward_function

if __name__ == "__main__":
    import train_grpo as tg
    import spider_env as env

    samples = env.load_spider_samples()[:2]
    dataset = tg.build_training_dataset(samples)

    sample_lookup = {i: sample for i, sample in enumerate(samples)}
    records = tg.build_training_records(samples)
    reward_func = tg.build_spider_reward_function(sample_lookup)

    print(reward_func(
        prompts=[records[0]["prompt"]] * 3,
        completions=[samples[0].gold_query, "SELECT 1", "not valid sql"],
        sample_id=[0, 0, 0],
    ))