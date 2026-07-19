from __future__ import annotations
import spider_env as env
import numpy as np
from debug_output import JSONLWriter
from pathlib import Path
from peft import PeftModel
from tqdm.auto import tqdm
from train_grpo import build_training_records
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, pipeline


def load_model_and_tokenizer(model_source: str, adapter: str | None = None) -> tuple:
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        dtype="auto",
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_source)

    if adapter is not None:
        model = PeftModel.from_pretrained(
            model,
            adapter,
            is_trainable=False,
        )

    model.eval()
    model.requires_grad_(False)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return (model, tokenizer)


def generate_completions(
    model,
    tokenizer,
    samples: list[env.SpiderSample],
    batch_size: int,
    num_generations: int,
    max_completion_length: int,
    temperature: float,
    seed: int,
) -> list[dict]:

    set_seed(seed)
    prompts = [record["prompt"] for record in build_training_records(samples)]

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    completions = pipe(
        (prompt for prompt in prompts),
        batch_size=batch_size,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_completion_length,
        num_return_sequences=num_generations,
        return_full_text=False,
    )
    
    completions = tqdm(
        completions, 
        total=len(samples), 
        desc="Generating:", 
        unit="Sample", 
        dynamic_ncols=True
    )

    records = []
    for sample_id, (sample, candidates) in enumerate(zip(samples, completions)):
        if len(candidates) != num_generations:
            raise ValueError("Candidate length does not match num_generations. ")
        generation_id = 0
        for candidate in candidates:
            records.append(
                {
                    "sample_id": sample_id,
                    "generation_id": generation_id,
                    "db_id": sample.db_id,
                    "raw_completion": candidate["generated_text"],
                }
            )
            generation_id += 1

    expected_length = len(samples) * num_generations
    if len(records) != expected_length:
        raise ValueError("Length of records does not match the expected length. ")

    return records


def score_completions(
    completions: list[dict], sample_lookup: dict, output: Path
) -> None:
    completion_writer = JSONLWriter(output / "completions.jsonl")
    summary_writer = JSONLWriter(output / "summary.jsonl")
    num_candidates = len(completions)
    executed = 0
    scores = []
    accuracy = []
    first_response_accuracy = []
    format_compliance = []

    for completion in tqdm(completions, desc="Scoring:", unit="Candidate", dynamic_ncols=True):
        sample = sample_lookup[completion["sample_id"]]
        generation_id = completion["generation_id"]
        raw_completion = completion["raw_completion"]
        extraction = env.extract_sql(raw_completion)
        score = env.score_sql_candidate(
            sample, extraction["query"], has_extra_text=extraction["has_extra_text"]
        )
        scores.append(score)

        if generation_id == 0:
            if score == 1 or score == 0.4:
                first_response_accuracy.append(1)
            else:
                first_response_accuracy.append(0)

        if extraction["has_extra_text"] == False:
            format_compliance.append(1)
        else:
            format_compliance.append(0)

        if score == 0:
            outcome = "execution failed. "
            accuracy.append(0)
        elif score == 0.2:
            outcome = "produced wrong query. "
            executed += 1
            accuracy.append(0)
        elif score == 0.4:
            outcome = "correct query. extra text. "
            executed += 1
            accuracy.append(1)
        else:
            outcome = "correct query."
            executed += 1
            accuracy.append(1)

        row = {
            "sample_id": completion["sample_id"],
            "generation_id": generation_id,
            "db_id": completion["db_id"],
            "question": sample.question,
            "gold_query": sample.gold_query,
            "raw_completion": completion["raw_completion"],
            "extracted_sql": extraction["query"],
            "was_fenced": extraction["was_fenced"],
            "has_extra_text": extraction["has_extra_text"],
            "execution_ok": True if score != 0 else False,
            "execution_error": True if score == 0 else False,
            "exact_match": True if score == 1 else False,
            "outcome": outcome,
            "reward": score,
        }

        completion_writer.write(row)

    mean_scores = np.array(scores).mean()
    mean_accuracy = np.array(accuracy).mean()
    mean_first_response_accuracy = np.array(first_response_accuracy).mean()
    mean_format_compliance = np.array(format_compliance).mean()

    summary = {
        "num_candidates": num_candidates,
        "num_samples": len(sample_lookup),
        "num_generations_per_sample": len(completions) / len(sample_lookup),
        "executable_rate": executed / num_candidates,
        "mean_reward_score": mean_scores,
        "mean_accuracy": mean_accuracy,
        "mean_first_response_accuracy": mean_first_response_accuracy,
        "mean_format_compliance": mean_format_compliance,
    }

    summary_writer.write(summary)


if __name__ == "__main__":
    import argparse
    from datetime import UTC, datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", default=50, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--max-completion-length", default=128, type=int)
    parser.add_argument("--num-generations", default=4, type=int)
    parser.add_argument("--seed", default=13, type=int)
    parser.add_argument("--output", default="outputs/grpo_evals")

    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(
        model_source=args.model, adapter=args.adapter if args.adapter else None
    )
    samples = env.load_spider_samples(split=args.split)[: args.limit]
    sample_lookup = {i: sample for i, sample in enumerate(samples)}

    completions = generate_completions(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        seed=args.seed,
    )

    timestamp = datetime.strftime(datetime.now(UTC), "%Y%m%d_%H%M%S")
    model_name = Path(args.model).name
    adapter_name = Path(args.adapter).name if args.adapter else "No_Adapter"
    output = Path(args.output) / f"{model_name}_{adapter_name}_{timestamp}"

    score_completions(completions, sample_lookup, output)
