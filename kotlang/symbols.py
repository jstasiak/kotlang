import itertools
import re
from typing import Iterable

LEGAL_SYMBOL_RE = re.compile('^[a-zA-Z_.$][a-zA-Z_.$0-9]*$')


def mangle(path: Iterable[str]) -> str:
    assert type(path) is not str
    sanitized_path = (sanitize(e) for e in path)
    prefixed_path = (f'{len(e)}{e}' for e in sanitized_path)
    return ''.join(itertools.chain(['_ZN'], prefixed_path))


def sanitize(name: str) -> str:
    assert LEGAL_SYMBOL_RE.match(name) is not None, name
    return name
