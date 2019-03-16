from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LineColumn:
    line: int
    column: int


@dataclass
class Location(LineColumn):
    filename: str


@dataclass
class Span:
    lo: LineColumn
    hi: LineColumn
    filename: str

    @property
    def lo_location(self) -> Location:
        return Location(self.lo.line, self.lo.column, self.filename)

    @property
    def hi_location(self) -> Location:
        return Location(self.hi.line, self.hi.column, self.filename)


dummy_location = Location(0, 0, '')
dummy_line_column = LineColumn(0, 0)
dummy_span = Span(dummy_line_column, dummy_line_column, '')
