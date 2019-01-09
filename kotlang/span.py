from dataclasses import dataclass


@dataclass
class Span:
    line: int
    column: int


dummy_span = Span(0, 0)
