from __future__ import annotations
import logging
import spider_env as env
from datasets import Dataset
from debug_output import JSONLWriter, build_logger


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


def build_spider_reward_function(sample_lookup:dict[int, env.SpiderSample], logger:logging.Logger|None = None, *, json_debugger:JSONLWriter|None = None):
    def spider_reward_function(prompts:list[str], completions:list[str], sample_id:list[int], **kwargs) -> list[float]:
        sample_ids = sample_id

        if len(completions) != len(sample_ids):
            raise ValueError("Different numbers of Completions and Sample IDs. ")

        rewards = []

        for completion, sid in zip(completions, sample_ids):
            sample = sample_lookup[sid]
            query = env.extract_sql(completion)
            result = env.execute_query(sample.db_path, query)
            reward = env.score_sql_candidate(sample, query)
            rewards.append(reward)

            if not result.ok:
                outcome = "execution_error"
            elif reward == 1.0:
                outcome = "exact_match"
            else:
                outcome = "wrong_result"

            record = {
                "sample_id": sid,
                "db_id": sample.db_id,
                "raw_completion": completion,
                "extracted_sql": query,
                "reward": reward,
                "outcome": outcome,
                "error": result.error,
            }
            if json_debugger:
                json_debugger.write(record)
            if logger:
                logger.info(
                    "sample_id=%s db_id=%s reward=%s outcome=%s error=%s",
                    sid,
                    sample.db_id,
                    reward,
                    outcome,
                    result.error,
                )

        return rewards

    return spider_reward_function

if __name__ == "__main__":
    import argparse
    from trl import GRPOConfig, GRPOTrainer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default = "Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--split", default = "train")
    parser.add_argument("--limit", default = 8, type = int)
    parser.add_argument("--output-dir", default = 'outputs/grpo_smoke')
    parser.add_argument("--max-steps", default = 1, type= int)
    parser.add_argument("--batch-size", default = 2, type= int)
    parser.add_argument("--num_generations", default = 2, type= int)
    parser.add_argument("--max-completion-length", default = 128, type = int)
    parser.add_argument("--learning-rate", default = 1e-6, type = float)
    parser.add_argument("--temperature", default = 1.0, type = float)
    parser.add_argument("--train", action="store_true")

    args = parser.parse_args()

    if args.batch_size % args.num_generations != 0:
        raise ValueError("batch size must be divisible by num generations")

    training_config = GRPOConfig(
        output_dir = args.output_dir,
        max_steps = args.max_steps,
        per_device_train_batch_size = args.batch_size,
        num_generations = args.num_generations,
        max_completion_length= args.max_completion_length,
        learning_rate = args.learning_rate,
        temperature = args.temperature,
        remove_unused_columns = False,
        report_to = "none"
    )

    samples = env.load_spider_samples(split=args.split)[:args.limit]
    dataset = build_training_dataset(samples)

    sample_lookup = {i: sample for i, sample in enumerate(samples)}
    records = build_training_records(samples)
    reward_func = build_spider_reward_function(sample_lookup)

    trainer = GRPOTrainer(
        model = args.model,
        args = training_config,
        reward_funcs = reward_func,
        train_dataset = dataset
    )

    if args.train:
        trainer.train()
    else:
        print("Trainer built. Pass --train to begin. ")
