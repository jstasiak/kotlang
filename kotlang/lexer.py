import dataclasses
import enum
from typing import Iterator, Tuple


class TokenType(enum.Enum):
    integer_literal = enum.auto()
    string_literal = enum.auto()
    bool_literal = enum.auto()
    float_literal = enum.auto()
    identifier = enum.auto()
    whitespace = enum.auto()
    comment = enum.auto()
    op = enum.auto()
    eof = enum.auto()


@dataclasses.dataclass
class Token:
    type: TokenType
    text: str
    line: int = 0
    column: int = 0


IDENTIFIER_START = 'abcdefghijklmnopqrstuvwxyz_'
IDENTIFIER_START += IDENTIFIER_START.upper()
IDENTIFIER_CONTINUATION = IDENTIFIER_START + '01234567890'

OPS = set('! , . ... : ; -> * / \ + - < <= > >= ( ) [ ] { } = == !='.split())
MAX_OP_WIDTH = max(len(op) for op in OPS)


def lex(text: str) -> Iterator[Token]:
    return only_important_tokens(provide_line_and_column_numbers(text, _lex(text)))


def provide_line_and_column_numbers(text: str, tokens: Iterator[Token]) -> Iterator[Token]:
    line = 0
    column = 0

    for t in tokens:
        t = dataclasses.replace(t, line=line, column=column)
        yield t
        line += t.text.count('\n')
        has_newline = '\n' in t.text
        # FIXME: dependong in its position tab doesn't have to be displayed as 8 characters
        columns_in_this_token = len(t.text.rsplit('\n', 1)[-1].replace('\t', 8 * ' '))
        if has_newline:
            column = columns_in_this_token
        else:
            column += columns_in_this_token


def only_important_tokens(tokens: Iterator[Token]) -> Iterator[Token]:
    for t in tokens:
        if t.type not in {TokenType.whitespace, TokenType.comment}:
            yield t


def _lex(text: str) -> Iterator[Token]:
    def peek(n: int = 1) -> str:
        return text[:n]

    def expect(character: str) -> str:
        actual_character = text[0]
        assert actual_character == character
        return actual_character

    def eat(n: int = 1) -> str:
        nonlocal text
        c, text = text[:n], text[n:]
        return c

    while text:
        c = peek()
        if is_whitespace(c):
            lexeme, text = scan_whitespace(text)
            yield lexeme
        elif text.startswith('//') or text.startswith('/*'):
            lexeme, text = scan_comment(text)
            yield lexeme
        elif is_identifier_start(c):
            lexeme, text = scan_identifier(text)
            yield lexeme
        elif c == '"':
            lexeme, text = scan_string_literal(text)
            yield lexeme
        elif c.isdigit():
            lexeme, text = scan_numeric_literal(text)
            yield lexeme
        else:
            for length in range(MAX_OP_WIDTH, 0, -1):
                potential_op = text[:length]
                if potential_op in OPS:
                    yield Token(TokenType.op, eat(length))
                    break
            else:
                yield Token(TokenType.op, eat(1))
    yield Token(TokenType.eof, '')


def scan_string_literal(text: str) -> Tuple[Token, str]:
    assert text[0] == '"'
    closing_quote_index = text.index('"', 1)
    literal, text = text[:closing_quote_index + 1], text[closing_quote_index + 1:]
    return Token(type=TokenType.string_literal, text=literal), text


def scan_numeric_literal(text: str) -> Tuple[Token, str]:
    literal = ''
    has_period = False
    while text and (text[0].isdigit() or text[0] == '.' and not has_period):
        c, text = text[0], text[1:]
        if c == '.':
            has_period = True
        literal += c
    assert literal
    type_ = TokenType.float_literal if has_period else TokenType.integer_literal
    return Token(type=type_, text=literal), text


def scan_whitespace(text: str) -> Tuple[Token, str]:
    part = ''
    while text and is_whitespace(text[0]):
        element, text = text[0], text[1:]
        part += element
    assert part
    return Token(type=TokenType.whitespace, text=part), text


def scan_comment(text: str) -> Tuple[Token, str]:
    delimiters = {
        '//': '\n',
        '/*': '*/',
    }
    for begin, end in delimiters.items():
        if text.startswith(begin):
            break
    else:
        raise AssertionError('Expected a comment')

    head, tail = text.split(end, 1)
    return Token(TokenType.comment, head), tail


def is_whitespace(c: str) -> bool:
    return c in {' ', '\t', '\n', '\r'}


def is_identifier_start(c: str) -> bool:
    return c in IDENTIFIER_START


def is_identifier_continuation(c: str) -> bool:
    return c in IDENTIFIER_CONTINUATION


def scan_identifier(text: str) -> Tuple[Token, str]:
    identifier, text = text[0], text[1:]

    while text and is_identifier_continuation(text[0]):
        element, text = text[0], text[1:]
        identifier += element
    assert identifier

    type_ = TokenType.identifier
    if identifier in {'true', 'false'}:
        type_ = TokenType.bool_literal
    return Token(type=type_, text=identifier), text
