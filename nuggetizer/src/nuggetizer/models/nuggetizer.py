"""Synchronous assignment subset of the official Nuggetizer implementation."""

import ast
import logging

from ..core.llm import LLMHandler
from ..core.types import AssignedScoredNugget, NuggetAssignMode, ScoredNugget
from ..prompts.assigner_prompts import create_assign_prompt


MAX_TRIALS = 500


class Nuggetizer:
    """Run the vendored three-grade GPT assigner with official window semantics."""

    def __init__(
        self,
        model: str | None = None,
        assigner_model: str | None = "gpt-4o",
        assigner_mode: NuggetAssignMode = NuggetAssignMode.SUPPORT_GRADE_3,
        window_size: int | None = None,
        assigner_window_size: int = 10,
        log_level: int = 0,
        use_azure_openai: bool = False,
        **llm_kwargs: object,
    ):
        self.assigner_model = model or assigner_model or "gpt-4o"
        self.assigner_mode = assigner_mode
        self.assigner_window_size = window_size or assigner_window_size
        self.llm_kwargs = {"use_azure_openai": use_azure_openai, **llm_kwargs}
        self.assigner_llm = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(
            logging.DEBUG if log_level >= 2 else logging.INFO if log_level else logging.WARNING
        )

    @staticmethod
    def _parse_labels(response: str, expected: int) -> list[str]:
        cleaned = response.replace("```python", "").replace("```", "").strip()
        labels = ast.literal_eval(cleaned)
        allowed = {"support", "partial_support", "not_support"}
        if not isinstance(labels, list) or len(labels) != expected:
            raise ValueError(f"Expected {expected} assignment labels, got {labels!r}")
        normalized = [str(label).lower() for label in labels]
        if any(label not in allowed for label in normalized):
            raise ValueError(f"Unexpected assignment label(s): {normalized!r}")
        return normalized

    def _ensure_assigner(self) -> None:
        if self.assigner_llm is None:
            self.assigner_llm = LLMHandler(self.assigner_model, **self.llm_kwargs)

    def _assign_window(
        self, query: str, context: str, nuggets: list[ScoredNugget]
    ) -> list[AssignedScoredNugget]:
        self._ensure_assigner()
        prompt = create_assign_prompt(query, context, nuggets, self.assigner_mode)
        temperature = 0.0
        for _attempt in range(MAX_TRIALS):
            response, _tokens = self.assigner_llm.run(prompt, temperature=temperature)
            try:
                labels = self._parse_labels(response, len(nuggets))
                return [
                    AssignedScoredNugget(
                        text=nugget.text,
                        importance=nugget.importance,
                        assignment=label,
                    )
                    for nugget, label in zip(nuggets, labels, strict=True)
                ]
            except (SyntaxError, ValueError) as exc:
                self.logger.warning("Invalid assigner response; retrying: %s", exc)
                temperature = 0.2
        raise RuntimeError(f"Could not parse Nuggetizer assignment after {MAX_TRIALS} trials")

    def assign(
        self,
        query: str,
        context: str,
        nuggets: list[ScoredNugget],
    ) -> list[AssignedScoredNugget]:
        assigned = []
        for start in range(0, len(nuggets), self.assigner_window_size):
            assigned.extend(
                self._assign_window(
                    query, context, nuggets[start:start + self.assigner_window_size]
                )
            )
        return assigned

    def assign_batch(self, queries, contexts, nuggets_list):
        return [
            self.assign(query, context, nuggets)
            for query, context, nuggets in zip(queries, contexts, nuggets_list, strict=True)
        ]
