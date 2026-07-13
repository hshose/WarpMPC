"""DAgger schedule helpers."""

from __future__ import annotations


def parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one integer value")
    return values


def parse_float_tuple(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one floating-point value")
    return values


def dagger_schedule(iterations: int, text: str | tuple[float, ...] | list[float]) -> tuple[float, ...]:
    """Return a DAgger mixing schedule, extending the last value if needed."""

    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    values = tuple(float(value) for value in text) if not isinstance(text, str) else parse_float_tuple(text)
    if not values:
        raise ValueError("expected at least one DAgger mixing value")
    if iterations <= len(values):
        return values[:iterations]
    return values + (values[-1],) * (iterations - len(values))
