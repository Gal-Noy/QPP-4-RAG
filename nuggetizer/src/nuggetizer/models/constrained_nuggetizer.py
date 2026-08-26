"""Deterministic constrained classifier for small local chat-model judges."""

from __future__ import annotations

from ..core.types import AssignedScoredNugget, ScoredNugget


LABEL_CHOICES = {
    " A": "support",
    " B": "partial_support",
    " C": "not_support",
}


def create_classification_prompt(
    query: str,
    context: str,
    nugget: ScoredNugget,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You label whether one atomic information nugget is captured by "
                "an answer passage. Judge only what the passage explicitly states."
            ),
        },
        {
            "role": "user",
            "content": (
                "Choose exactly one label:\n"
                "A = support: the nugget is fully captured by the passage.\n"
                "B = partial_support: the nugget is only partially captured.\n"
                "C = not_support: the nugget is not captured at all.\n\n"
                f"Search Query: {query}\n"
                f"Passage: {context}\n"
                f"Nugget: {nugget.text}\n\n"
                "Return only A, B, or C."
            ),
        },
        {"role": "assistant", "content": "Label:"},
    ]


class ConstrainedNuggetizer:
    """Assign exactly one of three labels per nugget using next-token logits."""

    def __init__(self, chooser, *, batch_size: int = 1) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.chooser = chooser
        self.batch_size = batch_size

    def assign(
        self,
        query: str,
        context: str,
        nuggets: list[ScoredNugget],
    ) -> list[AssignedScoredNugget]:
        labels: list[str] = []
        for start in range(0, len(nuggets), self.batch_size):
            batch = nuggets[start:start + self.batch_size]
            labels.extend(
                self.chooser.choose_one_token(
                    [create_classification_prompt(query, context, nugget) for nugget in batch],
                    LABEL_CHOICES,
                    labels=[nugget.text for nugget in batch],
                )
            )
        return [
            AssignedScoredNugget(
                text=nugget.text,
                importance=nugget.importance,
                assignment=label,
            )
            for nugget, label in zip(nuggets, labels, strict=True)
        ]
