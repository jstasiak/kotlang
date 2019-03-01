from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import zip_longest
from typing import (
    Any,
    cast,
    Collection,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
)

from llvmlite import ir


class Node:
    pass


@dataclass
class StructUnion:
    name: str
    members: List[Tuple[str, TypeReference]]
    is_union: bool


@dataclass
class Function:
    name: str
    parameters: ParameterList
    return_type: TypeReference
    type_parameters: List[str]
    code_block: Optional[CodeBlock]

    @property
    def is_generic(self) -> bool:
        return bool(self.type_parameters)


@dataclass
class Module:
    types: List[StructUnion]
    functions: List[Function]
    imports: List[str]
    includes: List[str]
    variables: List[VariableDeclaration]


class Statement:
    pass


@dataclass
class CompoundStatement(Statement):
    statements: List[Statement]


@dataclass
class CodeBlock(Statement):
    statements: List[Statement]


@dataclass
class IfStatement(Statement):
    expression: Expression
    first_statement: Statement
    second_statement: Optional[Statement] = None


@dataclass
class PatternMatchArm:
    pattern: Expression
    body: Expression


@dataclass
class PatternMatch(Statement):
    match_value: Expression
    arms: List[PatternMatchArm]


@dataclass
class WhileLoop(Statement):
    condition: Expression
    body: Statement


@dataclass
class ForLoop(Statement):
    entry: Statement
    condition: Expression
    step: Statement
    body: Statement


@dataclass
class ReturnStatement(Statement):
    expression: Optional[Expression] = None


@dataclass
class VariableDeclaration(Statement):
    name: str
    expression: Optional[Expression]
    type_: Optional[TypeReference] = None

    def __post_init__(self) -> None:
        assert self.expression is not None or self.type_ is not None, (self.expression, self.type_)


class Expression(Statement):
    def get_constant_time_value(self) -> Any:
        raise NotImplementedError(f'{0} is not a compile-time constant')


@dataclass
class NegativeExpression(Expression):
    expression: Expression


@dataclass
class BoolNegationExpression(Expression):
    expression: Expression


@dataclass
class BinaryExpression(Expression):
    left_operand: Expression
    operator: str
    right_operand: Expression
    name: str = ''


@dataclass
class FunctionCall(Expression):
    name: str
    parameters: List[Expression]


"""
def whatever<T>(T a) -> void ...
def whatever<T>(int a, T b) -> void ...
"""


@dataclass
class StructInstantiation(Expression):
    name: str
    parameters: List[Expression]


@dataclass
class StringLiteral(Expression):
    text: str

    def __post_init__(self) -> None:
        self.text = evaluate_escape_sequences(self.text)


def evaluate_escape_sequences(text: str) -> str:
    return text.replace(r'\n', '\n')


@dataclass
class IntegerLiteral(Expression):
    text: str

    def get_constant_time_value(self) -> Any:
        return int(self.text)


@dataclass
class FloatLiteral(Expression):
    text: str


@dataclass
class BoolLiteral(Expression):
    value: bool


class MemoryReference(Expression):
    pass


@dataclass
class VariableReference(MemoryReference):
    name: str


@dataclass
class AddressOf(MemoryReference):
    variable: VariableReference


@dataclass
class ValueAt(MemoryReference):
    variable: VariableReference


@dataclass
class Assignment(Expression):
    target: Expression
    expression: Expression


@dataclass
class ArrayLiteral(Expression):
    initializers: List[Expression]

    def __post_init__(self) -> None:
        assert len(self.initializers) > 0


@dataclass
class DotAccess(MemoryReference):
    left_side: MemoryReference
    member: str


@dataclass
class IndexAccess(MemoryReference):
    pointer: MemoryReference
    index: Expression


class TypeReference:
    def as_pointer(self) -> PointerTypeReference:
        return PointerTypeReference(self)

    def most_basic_type(self) -> TypeReference:
        raise NotImplementedError()


@dataclass
class BaseTypeReference(TypeReference):
    name: str

    def most_basic_type(self) -> BaseTypeReference:
        return self


@dataclass
class ArrayTypeReference(TypeReference):
    base: TypeReference
    length: Expression

    def most_basic_type(self) -> TypeReference:
        return self.base.most_basic_type()


@dataclass
class FunctionTypeReference(TypeReference):
    parameter_types: List[TypeReference]
    return_type: TypeReference
    variadic: bool

    def most_basic_type(self) -> FunctionTypeReference:
        return self


@dataclass
class PointerTypeReference(TypeReference):
    base: TypeReference

    def most_basic_type(self) -> TypeReference:
        return self.base.most_basic_type()


@dataclass
class Parameter:
    name: Optional[str]
    type_: TypeReference


@dataclass
class ParameterList(Iterable[Parameter]):
    parameters: List[Parameter]
    variadic: bool = False

    def __iter__(self) -> Iterator[Parameter]:
        return iter(self.parameters)


def get_builtin_va_list_struct() -> StructUnion:
    # NOTE: this is Clang-specific and x86 64-bit ABI-specific
    # TODO: make this platform independent?
    return StructUnion(
        '__va_list_tag',
        [
            ('gp_offset', BaseTypeReference('i32')),
            ('fp_offset', BaseTypeReference('i32')),
            # Those pointer are void* originally, but LLVM IR doesn't support that, so...
            ('overflow_arg_area', BaseTypeReference('i8').as_pointer()),
            ('reg_save_area', BaseTypeReference('i8').as_pointer()),
        ],
        False,
    )
