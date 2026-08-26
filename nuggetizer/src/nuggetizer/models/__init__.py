"""Nuggetizer model implementations with lazy optional-service imports."""

__all__ = ["Nuggetizer", "ConstrainedNuggetizer"]


def __getattr__(name: str):
    if name == "Nuggetizer":
        from .nuggetizer import Nuggetizer

        return Nuggetizer
    if name == "ConstrainedNuggetizer":
        from .constrained_nuggetizer import ConstrainedNuggetizer

        return ConstrainedNuggetizer
    raise AttributeError(name)
