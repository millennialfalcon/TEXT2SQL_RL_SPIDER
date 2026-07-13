"""Build and run the TRL GRPO training harness for the Spider environment."""

from __future__ import annotations
import logging
import spider_env as env
from datasets import Dataset
from debug_output import JSONLWriter, build_logger


def build_training_records(samples: list[env.SpiderSample]) -> list[dict]:
    """Convert Spider samples into trainer rows with conversational prompts.

    Each row contains a stable integer ``sample_id`` and one user message. The
    reference SQL remains outside the prompt and is recovered through a
    separate sample lookup during reward calculation.
    """
    return [
        {
            "sample_id": i,
            "prompt": [{"role": "user", "content": env.create_prompt(sample)}],
        }
        for i, sample in enumerate(samples)
    ]


def build_training_dataset(samples: list[env.SpiderSample]) -> Dataset:
    """Build a Hugging Face Dataset containing trainer-ready Spider records."""
    return Dataset.from_list(build_training_records(samples))


def _local_oai_completions(sample: env.SpiderSample, n: int) -> list[str]:
    """Generate OpenAI completions for local reward-path debugging."""
    return env._local_call_oai(sample, n=n)


def reward_completions(
    completions: list[str],
    sample_ids: list[int],
    sample_lookup: dict[int, env.SpiderSample],
) -> list[float]:
    """Score plain-text completions by mapping each one to its Spider sample."""
    rewards = []
    for completion, sample_id in zip(completions, sample_ids):
        extraction = env.extract_sql(completion)
        reward = env.score_sql_candidate(
            sample_lookup[sample_id], extraction["query"], extraction["has_extra_text"]
        )
        rewards.append(reward)
    return rewards


def build_spider_reward_function(
    sample_lookup: dict[int, env.SpiderSample],
    logger: logging.Logger | None = None,
    *,
    json_debugger: JSONLWriter | None = None,
):
    """Create a TRL-compatible reward callback backed by Spider execution.

    Args:
        sample_lookup: Maps dataset ``sample_id`` values to full Spider samples.
        logger: Optional logger for one-line completion outcomes.
        json_debugger: Optional writer for raw completions and scoring metadata.

    Returns:
        A reward function that accepts TRL conversational completions and
        returns one float reward per completion.
    """

    def spider_reward_function(
        prompts: list[str], completions: list[str], sample_id: list[int], **kwargs
    ) -> list[float]:
        """Score a TRL completion batch and emit optional diagnostics."""
        sample_ids = sample_id

        if len(completions) != len(sample_ids):
            raise ValueError("Different numbers of Completions and Sample IDs. ")

        rewards = []

        for completion, sid in zip(completions, sample_ids):
            completion = completion[-1]["content"]
            sample = sample_lookup[sid]
            extraction = env.extract_sql(completion)
            query = extraction["query"]
            has_extra_text = extraction["has_extra_text"]
            was_fenced = extraction["was_fenced"]

            result = env.execute_query(sample.db_path, query)
            reward = env.score_sql_candidate(sample, query, has_extra_text)
            rewards.append(reward)

            if not result.ok:
                outcome = "execution_error"
            elif reward == 1.0:
                outcome = "exact_match"
            elif reward == 0.4:
                outcome = "correct_query_extra_text"
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
                "was_fenced": was_fenced,
                "has_extra_text": has_extra_text,
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
    from datetime import datetime
    from debug_output import JSONLWriter, build_logger
    from pathlib import Path
    from trl import GRPOConfig, GRPOTrainer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", default=8, type=int)
    parser.add_argument("--output-dir", default="outputs/grpo_smoke")
    parser.add_argument("--max-steps", default=1, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--num-generations", default=2, type=int)
    parser.add_argument("--max-completion-length", default=128, type=int)
    parser.add_argument("--learning-rate", default=1e-6, type=float)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--logging-steps", default=1, type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save-steps", default=5, type=int)
    parser.add_argument("--save-total-limit", default=2, type=int)

    args = parser.parse_args()

    if args.batch_size % args.num_generations != 0:
        raise ValueError("batch size must be divisible by num generations")

    if "/" in args.model:
        model_for_run_name = args.model.split("/")[-1]
    else:
        model_for_run_name = args.model

    timestamp = datetime.strftime(datetime.utcnow(), "%Y%m%d_%H%M%S")
    run_name = f"{model_for_run_name}_{timestamp}"

    samples = env.load_spider_samples(split=args.split)[: args.limit]
    dataset = build_training_dataset(samples)

    sample_lookup = {i: sample for i, sample in enumerate(samples)}
    records = build_training_records(samples)

    run_dir = Path(args.output_dir) / run_name
    logger = build_logger(run_dir / "train.log")

    training_config = GRPOConfig(
        output_dir=run_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        remove_unused_columns=False,
        report_to=args.report_to,
        logging_steps=args.logging_steps,
        run_name=run_name,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
    )

    if args.debug:
        jsonl_writer = JSONLWriter(run_dir / "completions.jsonl")
    else:
        jsonl_writer = None

    reward_func = build_spider_reward_function(
        sample_lookup, logger, json_debugger=jsonl_writer
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=training_config,
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    if args.train:
        trainer.train()
    else:
        print("Trainer built. Pass --train to begin. ")
