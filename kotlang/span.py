from dataclasses import dataclass


@dataclass
class Span:
    line: int
    column: int
    filename: str


dummy_span = Span(0, 0, '')
