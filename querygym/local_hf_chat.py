"""Reusable local Hugging Face chat-model runtime."""

from __future__ import annotations

from typing import Mapping, Sequence


class LocalChatModel:
    """Load one causal chat model and expose batch and Nuggetizer interfaces."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str = "auto",
        local_files_only: bool = False,
        load_in_4bit: bool = False,
        *,
        context_size: int = 16384,
        max_new_tokens: int = 128,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if context_size < 1 or max_new_tokens < 1:
            raise ValueError("context_size and max_new_tokens must be positive")
        self.torch = torch
        self.model_name = model_name
        self.context_size = context_size
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        kwargs: dict[str, object] = {"local_files_only": local_files_only}
        kwargs["dtype"] = getattr(torch, dtype) if dtype != "auto" else "auto"
        if load_in_4bit:
            compute_dtype = getattr(torch, dtype) if dtype != "auto" else None
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            kwargs["device_map"] = "auto"
        elif device == "auto":
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if device != "auto" and not load_in_4bit:
            self.model.to(device)
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def generate(
        self,
        messages_batch: Sequence[list[dict[str, str]]],
        *,
        max_new_tokens: int | None = None,
        context_size: int | None = None,
        temperature: float = 0.0,
        labels: Sequence[str] | None = None,
    ) -> list[str]:
        """Generate without truncation, failing clearly if a prompt is too long."""
        max_new_tokens = max_new_tokens or self.max_new_tokens
        context_size = context_size or self.context_size
        prompts = [
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in messages_batch
        ]
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        too_long = [
            (labels[index] if labels else str(index), length + max_new_tokens)
            for index, length in enumerate(input_lengths)
            if length + max_new_tokens > context_size
        ]
        if too_long:
            required = max(length for _, length in too_long)
            raise ValueError(
                "Prompts exceed the configured context size without truncation: "
                f"{', '.join(label for label, _ in too_long)}; requires at least {required}"
            )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        generation = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation["temperature"] = temperature
        with self.torch.inference_mode():
            output = self.model.generate(**encoded, **generation)
        prompt_width = encoded["input_ids"].shape[1]
        return self.tokenizer.batch_decode(
            output[:, prompt_width:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def run(
        self, messages: list[dict[str, str]], temperature: float = 0.0
    ) -> tuple[str, int]:
        """Match the small interface used by Nuggetizer's hosted LLM handler."""
        text = self.generate([messages], temperature=temperature)[0]
        return text, len(self.tokenizer.encode(text, add_special_tokens=False))

    def choose_one_token(
        self,
        messages_batch: Sequence[list[dict[str, str]]],
        choices: Mapping[str, str],
        *,
        context_size: int | None = None,
        labels: Sequence[str] | None = None,
    ) -> list[str]:
        """Choose among single-token continuations without free-form generation."""
        context_size = context_size or self.context_size
        token_to_value: dict[int, str] = {}
        for continuation, value in choices.items():
            token_ids = self.tokenizer.encode(continuation, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(
                    f"Constrained choice {continuation!r} is not one tokenizer token"
                )
            if token_ids[0] in token_to_value:
                raise ValueError("Constrained choices must map to distinct tokens")
            token_to_value[token_ids[0]] = value

        prompts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                continue_final_message=True,
            )
            for messages in messages_batch
        ]
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True)
        lengths = encoded["attention_mask"].sum(dim=1).tolist()
        too_long = [
            (labels[index] if labels else str(index), length + 1)
            for index, length in enumerate(lengths)
            if length + 1 > context_size
        ]
        if too_long:
            required = max(length for _, length in too_long)
            raise ValueError(
                "Classification prompts exceed the configured context size: "
                f"{', '.join(label for label, _ in too_long)}; requires at least {required}"
            )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        candidate_ids = list(token_to_value)
        with self.torch.inference_mode():
            logits = self.model(**encoded).logits[:, -1, candidate_ids]
        selected = logits.argmax(dim=1).tolist()
        return [token_to_value[candidate_ids[index]] for index in selected]
