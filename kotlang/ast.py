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
    Type as TypingType,
    TypeVar,
    Union as TypingUnion,
)

from llvmlite import ir

from kotlang import typesystem as ts
from kotlang.symbols import mangle


class Node:
    pass


@dataclass
class TypeDefinition:
    name: str

    def get_dummy_type(self) -> TypingUnion[ts.StructType, ts.UnionType]:
        # Complex types can contain references to themselves so we split creating types into two phases.
        # This is the first one. After the returned type is added to a namespace we call fill_type_members()
        raise NotImplementedError()

    def fill_type_members(self, namespace: Namespace, type_: ts.Type) -> None:
        raise NotImplementedError()


@dataclass
class Struct(TypeDefinition):
    members: List[Tuple[str, TypeReference]]

    def get_dummy_type(self) -> ts.StructType:
        return ts.StructType(self.name, [])

    def fill_type_members(self, namespace: Namespace, type_: ts.Type) -> None:
        # Note: This method mutates type_
        assert isinstance(type_, ts.StructType)
        members = [(n, t.codegen(namespace)) for n, t in self.members]
        type_.members = members


@dataclass
class Union(TypeDefinition):
    members: List[Tuple[str, TypeReference]]

    def get_dummy_type(self) -> ts.UnionType:
        return ts.UnionType(self.name, [])

    def fill_type_members(self, namespace: Namespace, type_: ts.Type) -> None:
        # Note: This method mutates type_
        assert isinstance(type_, ts.UnionType)
        members = [(n, t.codegen(namespace)) for n, t in self.members]
        type_.members = members


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

    def symbol_name(self, namespace: Namespace) -> str:
        # TODO: stop hardcoding this?
        if self.code_block is None or self.name == 'main':
            return self.name

        type_values = [namespace.get_type(t).name for t in self.type_parameters]
        return mangle([self.name] + type_values)

    def get_type(self, namespace: Namespace) -> ts.FunctionType:
        ref = FunctionTypeReference(
            [p.type_ for p in self.parameters], self.return_type, self.parameters.variadic
        )
        return ref.codegen(namespace)


@dataclass
class Module:
    types: List[TypeDefinition]
    functions: List[Function]
    imports: List[str]
    includes: List[str]
    variables: List[VariableDeclaration]


@dataclass
class Variable:
    name: str
    type_: ts.Type
    value: ir.Value


_T = TypeVar('_T')


@dataclass
class Namespace:
    parents: List[Namespace] = field(default_factory=list)
    types: Dict[str, ts.Type] = field(default_factory=dict)
    values: Dict[str, Variable] = field(default_factory=dict)
    functions: Dict[str, Function] = field(default_factory=dict)

    def add_type(self, t: ts.Type, name: Optional[str] = None) -> None:
        self._add_item(self.types, t, name or t.name)

    def add_value(self, t: Variable) -> None:
        self._add_item(self.values, t, t.name)

    def add_function(self, t: Function) -> None:
        self._add_item(self.functions, t, t.name)

    def _add_item(self, sub: Dict[str, _T], item: _T, name: str) -> None:
        # This method mutates sub
        assert name not in sub, name
        sub[name] = item

    def get_type(self, name: str) -> ts.Type:
        return cast(ts.Type, self._get_item('types', name))

    def get_value(self, name: str) -> Variable:
        return cast(Variable, self._get_item('values', name))

    def get_function(self, name: str) -> Function:
        return cast(Function, self._get_item('functions', name))

    def _get_item(self, sub_name: str, name: str) -> Any:
        sub = getattr(self, sub_name)
        try:
            result = sub[name]
            return result
        except KeyError:
            for p in self.parents:
                try:
                    return p._get_item(sub_name, name)
                except KeyError:
                    pass
            raise KeyError(name)


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

    def codegen_module_level(self, module: ir.Module, namespace: Namespace, module_name: str) -> None:
        value = self.expression.get_constant_time_value() if self.expression else None
        type_ = self.variable_type(namespace)
        constant = ir.Constant(type_.get_ir_type(), value)
        variable = ir.GlobalVariable(module, type_.get_ir_type(), mangle([module_name, self.name]))
        variable.initializer = constant
        variable.global_constant = True
        namespace.add_value(Variable(self.name, type_, variable))

    def variable_type(self, namespace: Namespace) -> ts.Type:
        return (
            self.type_.codegen(namespace)
            if self.type_ is not None
            else cast(Expression, self.expression).type(namespace)
        )


class Expression(Statement):
    def type(self, namespace: Namespace) -> ts.Type:
        raise NotImplementedError(f'type() not implemented for {type(self)}')

    def get_constant_time_value(self) -> Any:
        raise NotImplementedError(f'{0} is not a compile-time constant')


@dataclass
class NegativeExpression(Expression):
    expression: Expression

    def type(self, namespace: Namespace) -> ts.Type:
        return self.expression.type(namespace)


@dataclass
class BoolNegationExpression(Expression):
    expression: Expression

    def type(self, namespace: Namespace) -> ts.Type:
        return self.expression.type(namespace)


@dataclass
class BinaryExpression(Expression):
    left_operand: Expression
    operator: str
    right_operand: Expression
    name: str = ''

    def type(self, namespace: Namespace) -> ts.Type:
        if self.operator in {'<', '>', '<=', '>=', '==', '!='}:
            return namespace.get_type('bool')
        elif self.operator in {'+', '-', '*', '/'}:
            return self.left_operand.type(namespace)
        assert False, (self.operator, self.left_operand, self.right_operand)


@dataclass
class FunctionCall(Expression):
    name: str
    parameters: List[Expression]

    def type(self, namespace: Namespace) -> ts.Type:
        return namespace.get_type('i64')


"""
def whatever<T>(T a) -> void ...
def whatever<T>(int a, T b) -> void ...
"""


@dataclass
class StructInstantiation(Expression):
    name: str
    parameters: List[Expression]

    def type(self, namespace: Namespace) -> ts.Type:
        return namespace.get_type(self.name)


@dataclass
class StringLiteral(Expression):
    text: str

    def __post_init__(self) -> None:
        self.text = evaluate_escape_sequences(self.text)

    def type(self, namespace: Namespace) -> ts.Type:
        return namespace.get_type('i8').as_pointer()


def evaluate_escape_sequences(text: str) -> str:
    return text.replace(r'\n', '\n')


@dataclass
class IntegerLiteral(Expression):
    text: str

    def type(self, namespace: Namespace) -> ts.Type:
        return namespace.get_type('i64')

    def get_constant_time_value(self) -> Any:
        return int(self.text)


@dataclass
class FloatLiteral(Expression):
    text: str

    def type(self, namespace: Namespace) -> ts.Type:
        return namespace.get_type('f64')


@dataclass
class BoolLiteral(Expression):
    value: bool

    def type(self, namespace: Namespace) -> ts.Type:
        return namespace.get_type('bool')


class MemoryReference(Expression):
    pass


@dataclass
class VariableReference(MemoryReference):
    name: str

    def type(self, namespace: Namespace) -> ts.Type:
        value: TypingUnion[Function, Variable]
        try:
            function = namespace.get_function(self.name)
        except KeyError:
            variable = namespace.get_value(self.name)
            return variable.type_
        else:
            return function.get_type(namespace)


@dataclass
class AddressOf(MemoryReference):
    variable: VariableReference

    def type(self, namespace: Namespace) -> ts.Type:
        return namespace.get_value(self.variable.name).type_.as_pointer()


@dataclass
class ValueAt(MemoryReference):
    variable: VariableReference

    def type(self, namespace: Namespace) -> ts.Type:
        return namespace.get_value(self.variable.name).type_.as_pointee()


@dataclass
class Assignment(Expression):
    target: Expression
    expression: Expression

    def type(self, namespace: Namespace) -> ts.Type:
        # TODO and possibly quite important - type of expression can be different than the type of the target
        # (for example expression of type i8 assigned to i64 location) - which one should we use?
        # For now we take the original value but it may not be expected or desired.
        return self.expression.type(namespace)


@dataclass
class ArrayLiteral(Expression):
    initializers: List[Expression]

    def __post_init__(self) -> None:
        assert len(self.initializers) > 0

    def type(self, namespace: Namespace) -> ts.Type:
        # TODO make sure all elements are of the same type or can be coerced to one
        element_type = self.initializers[0].type(namespace)
        return ts.ArrayType(element_type, len(self.initializers))


@dataclass
class DotAccess(MemoryReference):
    left_side: MemoryReference
    member: str

    def type(self, namespace: Namespace) -> ts.Type:
        left_type = self.left_side.type(namespace)
        assert isinstance(left_type, ts.DottableType), left_type
        return left_type.get_member_type(self.member)


@dataclass
class IndexAccess(MemoryReference):
    pointer: MemoryReference
    index: Expression

    def type(self, namespace: Namespace) -> ts.Type:
        base_type = self.pointer.type(namespace)
        if isinstance(base_type, ts.PointerType):
            return base_type.pointee
        elif isinstance(base_type, ts.ArrayType):
            return base_type.element_type
        else:
            assert False, f'Bad memory reference: {self.pointer}'


class TypeReference:
    def codegen(self, namespace: Namespace) -> ts.Type:
        raise NotImplementedError()

    def as_pointer(self) -> PointerTypeReference:
        return PointerTypeReference(self)

    def most_basic_type(self) -> TypeReference:
        raise NotImplementedError()


@dataclass
class BaseTypeReference(TypeReference):
    name: str

    def codegen(self, namespace: Namespace) -> ts.Type:
        return namespace.get_type(self.name)

    def most_basic_type(self) -> BaseTypeReference:
        return self


@dataclass
class ArrayTypeReference(TypeReference):
    base: TypeReference
    length: Expression

    def codegen(self, namespace: Namespace) -> ts.Type:
        assert isinstance(self.length, IntegerLiteral)
        return ts.ArrayType(self.base.codegen(namespace), int(self.length.text))

    def most_basic_type(self) -> TypeReference:
        return self.base.most_basic_type()


@dataclass
class FunctionTypeReference(TypeReference):
    parameter_types: List[TypeReference]
    return_type: TypeReference
    variadic: bool

    def codegen(self, namespace: Namespace) -> ts.FunctionType:
        return_type = self.return_type.codegen(namespace)
        parameter_types = [t.codegen(namespace) for t in self.parameter_types]
        return ts.FunctionType(parameter_types, return_type, self.variadic)

    def most_basic_type(self) -> FunctionTypeReference:
        return self


@dataclass
class PointerTypeReference(TypeReference):
    base: TypeReference

    def codegen(self, namespace: Namespace) -> ts.PointerType:
        return self.base.codegen(namespace).as_pointer()

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


def get_builtin_va_list_struct() -> Struct:
    # NOTE: this is Clang-specific and x86 64-bit ABI-specific
    # TODO: make this platform independent?
    return Struct(
        '__va_list_tag',
        [
            ('gp_offset', BaseTypeReference('i32')),
            ('fp_offset', BaseTypeReference('i32')),
            # Those pointer are void* originally, but LLVM IR doesn't support that, so...
            ('overflow_arg_area', BaseTypeReference('i8').as_pointer()),
            ('reg_save_area', BaseTypeReference('i8').as_pointer()),
        ],
    )
