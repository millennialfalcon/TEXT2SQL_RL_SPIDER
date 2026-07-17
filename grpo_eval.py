from __future__ import annotations
import spider_env as env
from peft import PeftModel
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
):

    set_seed(seed)
    prompts = [record["prompt"] for record in build_training_records(samples)]

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    completions = pipe(
        prompts,
        batch_size=batch_size,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_completion_length,
        num_return_sequences=num_generations,
        return_full_text=False,
    )

    if len(completions) != len(samples):
        raise ValueError("Completion length does not match sample length. ")

    records = []
    for sample_id, (sample, candidates) in enumerate(zip(samples, completions)):
        if len(candidates) != num_generations:
            raise ValueError("Candidate length does not match num_generations. ")

        for candidate in candidates:
            records.append(
                {
                    "sample_id": sample_id,
                    "db_id": sample.db_id,
                    "raw_completion": candidate["generated_text"],
                }
            )

    expected_length = len(samples) * num_generations
    if len(records) != expected_length:
        raise ValueError("Length of records does not match the expected length. ")

    return records


if __name__ == "__main__":
    import argparse

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

    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(
        model_source=args.model, adapter=args.adapter if args.adapter else None
    )
    samples = env.load_spider_samples(split=args.split)[: args.limit]
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

    print(completions)
