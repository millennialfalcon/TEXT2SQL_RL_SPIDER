from __future__ import annotations

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_source: str, adapter: str | None = None
) -> tuple:
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

    return model, tokenizer
