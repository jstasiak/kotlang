import itertools
import re
from typing import Iterable

LEGAL_SYMBOL_RE = re.compile('^[a-zA-Z_.$][a-zA-Z_.$0-9]*$')


def mangle(path: Iterable[str]) -> str:
    assert type(path) is not str
    mangled = '_'.join(itertools.chain(['_kot'], path))
    assert LEGAL_SYMBOL_RE.match(mangled) is not None, mangled
    return mangled
