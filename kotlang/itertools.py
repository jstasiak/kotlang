from typing import Iterable, Iterator, List, TypeVar


_T = TypeVar('_T')


class Peekable(Iterator[_T]):
    def __init__(self, iterable: Iterable[_T]) -> None:
        self._iterator = iter(iterable)
        self._peek_buffer: List[_T] = []

    def __iter__(self) -> Iterator[_T]:
        return self

    def __next__(self) -> _T:
        try:
            return self._peek_buffer.pop(0)
        except IndexError:
            return next(self._iterator)

    def peek(self) -> _T:
        return self.peek_many(1)[0]

    def peek_many(self, count: int) -> List[_T]:
        while len(self._peek_buffer) < count:
            self._peek_buffer.append(next(self._iterator))
        return self._peek_buffer[:count]

    def peek_nth(self, index: int) -> _T:
        many = self.peek_many(index + 1)
        return many[index]
