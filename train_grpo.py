from __future__ import annotations
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

def make_spider_reward_function(sample_lookup:dict[int, env.SpiderSample]):
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
    samples = env.load_spider_samples()[:2]
    sample_lookup = {i:sample for i,sample in enumerate(samples)}
    reward_func = make_spider_reward_function(sample_lookup)