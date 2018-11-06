from __future__ import annotations

from dataclasses import dataclass, field
from itertools import zip_longest
from typing import (
    Any, cast, Collection, Dict, Iterable, Iterator, List,
    Mapping, MutableMapping, Optional,
    Tuple, Type as TypingType, TypeVar, Union,
)

from llvmlite import ir

from kotlang.symbols import mangle


constant_counter = 0


def string_constant(module: ir.Module, builder: ir.IRBuilder, s: str, namespace: Namespace) -> ir.Constant:
    global constant_counter
    name = f'constant{constant_counter}'
    constant_counter += 1

    as_bytes = s.encode()
    array_type = ir.ArrayType(namespace.get_type('i8').ir_type, len(as_bytes) + 1)
    global_value = ir.GlobalVariable(module, array_type, name)
    global_value.global_constant = True
    global_value.initializer = array_type(bytearray(as_bytes + b'\x00'))

    i64 = namespace.get_type('i64').ir_type
    return builder.gep(global_value, (i64(0), i64(0)))


class Node:
    pass


@dataclass
class Type:
    @property
    def ir_type(self) -> ir.Type:
        raise NotImplementedError(f'Not implemented for {type(self)}')

    @property
    def name(self) -> str:
        raise NotImplementedError(f'Not implemented for {type(self)}')

    def as_pointer(self) -> 'PointerType':
        return PointerType(self)

    def as_pointee(self) -> 'Type':
        assert isinstance(self, PointerType)
        return self.pointee

    def __eq__(self, other: Any) -> bool:
        return hasattr(other, 'name') and self.name == other.name

    def adapt(self, builder: ir.IRBuilder, value: ir.Value, from_type: Type) -> ir.Value:
        assert self == from_type, f'Cannot adapt {from_type.name} to {self.name}'
        return value


@dataclass
class VoidType(Type):
    @property
    def name(self) -> str:
        return 'void'

    @property
    def ir_type(self) -> ir.Type:
        return ir.VoidType()


@dataclass
class IntType(Type):
    bits: int
    signed: bool

    @property
    def name(self) -> str:
        prefix = 'i' if self.signed else 'u'
        return f'{prefix}{self.bits}'

    @property
    def ir_type(self) -> ir.Type:
        return ir.IntType(self.bits)

    def adapt(self, builder: ir.IRBuilder, value: ir.Value, from_type: Type) -> ir.Value:
        if not isinstance(from_type, IntType):
            return super().adapt(builder, value, from_type)

        if from_type.bits == self.bits:
            assert from_type.signed == self.signed
            return value

        if from_type.bits > self.bits:
            return builder.trunc(value, self.ir_type)

        if from_type.signed:
            assert self.signed
            return builder.sext(value, self.ir_type)
        else:
            assert not self.signed
            return builder.zext(value, self.ir_type)


@dataclass
class FloatType(Type):
    bits: int

    @property
    def name(self) -> str:
        return f'f{self.bits}'

    @property
    def ir_type(self) -> ir.Type:
        return ir.FloatType() if self.bits == 32 else ir.DoubleType()


@dataclass
class BoolType(Type):
    @property
    def name(self) -> str:
        return 'bool'

    @property
    def ir_type(self) -> ir.Type:
        return ir.IntType(1)


@dataclass
class PointerType(Type):
    pointee: Type

    @property
    def name(self) -> str:
        return self.pointee.name + '*'

    @property
    def ir_type(self) -> ir.Type:
        return self.pointee.ir_type.as_pointer()

    def adapt(self, builder: ir.IRBuilder, value: ir.Value, from_type: Type) -> ir.Value:
        # TODO remove this
        i64 = ir.IntType(64)

        if isinstance(from_type, ArrayType) and self.pointee == from_type.element_type:
            memory = builder.alloca(value.type)
            builder.store(value, memory)
            return builder.gep(memory, (i64(0), i64(0)))

        return super().adapt(builder, value, from_type)


@dataclass
class StructType(Type):
    struct_name: str
    members: List[Tuple[str, Type]]

    @property
    def name(self) -> str:
        return self.struct_name

    @property
    def ir_type(self) -> ir.Type:
        member_types = [t.ir_type for n, t in self.members]
        return ir.LiteralStructType(member_types)

    def get_member_index(self, name: str) -> int:
        for i, (n, t) in enumerate(self.members):
            if n == name:
                return i
        raise KeyError()

    def get_member_type(self, name: str) -> Type:
        for n, t in self.members:
            if n == name:
                return t
        raise KeyError()


@dataclass
class ArrayType(Type):
    element_type: Type
    length: int

    @property
    def name(self) -> str:
        return f'{self.element_type.name}[{self.length}]'

    @property
    def ir_type(self) -> ir.Type:
        return ir.ArrayType(self.element_type.ir_type, self.length)


@dataclass
class Struct:
    name: str
    members: List[Tuple[str, str]]

    def get_type(self, namespace: Namespace) -> Type:
        members = [(n, namespace.get_type(t)) for n, t in self.members]
        return StructType(self.name, members)


@dataclass
class Function:
    name: str
    type_parameters: List[str]
    parameters: ParameterList
    return_type: str
    code_block: Optional[CodeBlock]
    foreign_library: Optional[str] = None

    @property
    def is_generic(self) -> bool:
        return bool(self.type_parameters)

    def symbol_name(self, namespace: Namespace) -> str:
        # TODO: stop hardcoding this?
        if self.code_block is None or self.name == 'main':
            return self.name

        type_values = [namespace.get_type(t).name for t in self.type_parameters]
        return mangle([self.name] + type_values)


def get_or_create_llvm_function(
    module: ir.Module,
    namespace: Namespace,
    function: Function,
) -> ir.Function:
    symbol_name = function.symbol_name(namespace)
    try:
        llvm_function = module.globals[symbol_name]
        assert isinstance(llvm_function, ir.Function)
    except KeyError:
        return_type = namespace.get_type(function.return_type).ir_type
        function_type = ir.FunctionType(
            return_type,
            [p.type_.codegen(namespace).ir_type for p in function.parameters],
            function.parameters.variadic,
        )
        llvm_function = ir.Function(module, function_type, name=symbol_name)
        for p, arg in zip(function.parameters, llvm_function.args):
            arg.name = p.name + '_arg'

        if function.code_block is not None:
            block = llvm_function.append_basic_block(name="entry")
            builder = ir.IRBuilder(block)

            function_namespace = Namespace(parents=[namespace])
            for p, arg in zip(function.parameters, llvm_function.args):
                memory = builder.alloca(arg.type, name=p.name)
                builder.store(arg, memory)
                parameter_type = p.type_.codegen(namespace)
                function_namespace.add_item(Variable(p.name, parameter_type, memory))

            function.code_block.codegen(module, builder, function_namespace)
            if function.return_type == 'void':
                builder.ret_void()
            else:
                # FIXME: We depend on having returned already but this is not ensured
                if not builder.block.is_terminated:
                    builder.unreachable()

    return llvm_function


@dataclass
class Module:
    structs: List[Struct]
    functions: List[Function]
    imports: List[Tuple[str, Module]]

    def codegen_top_level(self, name: str) -> ir.Module:
        builtin_namespace = Namespace()
        builtin_namespace.add_item(VoidType())
        builtin_namespace.add_item(BoolType())
        for signed in {True, False}:
            for bits in {8, 16, 32, 64}:
                builtin_namespace.add_item(IntType(bits, signed=signed))
        builtin_namespace.add_item(FloatType(32))
        builtin_namespace.add_item(FloatType(64))

        module = ir.Module(name=name)
        namespace = self.codegen(module, builtin_namespace, MetaNamespace())
        functions = namespace.item_iter(Function)
        foreign_libraries = {f.foreign_library for f in functions} - {None}
        return module, foreign_libraries

    def codegen(
        self,
        module: ir.Module,
        parent_namespace: Namespace,
        meta_namespace: MetaNamespace,
    ) -> 'Namespace':
        import_namespaces: List[Namespace] = []
        for name, i in self.imports:
            if meta_namespace.has(name):
                imported_namespace = meta_namespace.get(name)
            else:
                imported_namespace = i.codegen(module, parent_namespace, meta_namespace)
                meta_namespace.set(name, imported_namespace)
            import_namespaces.append(imported_namespace)

        module_namespace = Namespace(parents=[parent_namespace, *import_namespaces])
        for s in self.structs:
            module_namespace.add_item(s.get_type(module_namespace))

        for f in self.functions:
            module_namespace.add_item(f)

        nongeneric_functions = (f for f in self.functions if not f.is_generic)
        for f in nongeneric_functions:
            get_or_create_llvm_function(module, module_namespace, f)

        return module_namespace


@dataclass
class Variable:
    name: str
    type_: Type
    value: ir.Value


@dataclass
class GeneratedFunction:
    function: Function
    llvm_function: ir.Function

    @property
    def name(self) -> str:
        return self.function.name


NamespaceItem = Union[Type, Variable, Function]
_T = TypeVar('_T')


@dataclass
class Namespace:
    parents: List[Namespace] = field(default_factory=list)
    _things: Dict[str, NamespaceItem] = field(default_factory=dict)

    def has_name(self, name: str) -> bool:
        try:
            self.get_item(name, object)
        except KeyError:
            return False
        else:
            return True

    def add_item(self, t: NamespaceItem, name: str = None) -> None:
        name = name if name is not None else t.name
        assert not self.has_name(name)
        self._things[name] = t

    def get_type(self, name: str) -> Type:
        return self.get_item(name, Type)

    def get_function(self, name: str) -> Function:
        return self.get_item(name, Function)

    def get_variable(self, name: str) -> Variable:
        return self.get_item(name, Variable)

    def item_iter(self, type_: TypingType[_T]) -> Iterator[_T]:
        for thing in self._things.values():
            if isinstance(thing, type_):
                yield thing
        for p in self.parents:
            for f in p.item_iter(type_):
                yield f

    def get_item(self, name: str, type_: TypingType[_T]) -> _T:
        try:
            result = self._things[name]
            assert isinstance(result, type_), (result, type_)
            return result  # type: ignore
        except KeyError:
            for p in self.parents:
                try:
                    return p.get_item(name, type_)
                except KeyError:
                    pass
            raise KeyError(name)


class MetaNamespace:
    def __init__(self) -> None:
        self._namespaces: Dict[str, Namespace] = {}

    def set(self, name: str, namespace: Namespace) -> None:
        assert name not in self._namespaces
        self._namespaces[name] = namespace

    def get(self, name: str) -> Namespace:
        return self._namespaces[name]

    def has(self, name: str) -> bool:
        return name in self._namespaces


class Statement:
    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace) -> None:
        raise NotImplementedError(f'Code generation not implemented for {type(self)}')


class CompoundStatement(Statement):
    def __init__(self, statements: List[Statement]) -> None:
        self.statements = statements

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace) -> None:
        for s in self.statements:
            s.codegen(module, builder, namespace)


class CodeBlock(Statement):
    def __init__(self, statements: List[Statement]) -> None:
        self.statements = statements

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace) -> None:
        block_namespace = Namespace(parents=[namespace])
        for s in self.statements:
            s.codegen(module, builder, block_namespace)


class IfStatement(Statement):
    def __init__(
        self,
        expression: Expression,
        first_statement: Statement,
        second_statement: Statement = None,
    ) -> None:
        self.expression = expression
        self.first_statement = first_statement
        self.second_statement = second_statement

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace) -> None:
        expression_value = self.expression.codegen(module, builder, namespace)
        with builder.if_else(expression_value) as (then, otherwise):
            with then:
                self.first_statement.codegen(module, builder, namespace)
            with otherwise:
                if self.second_statement is not None:
                    self.second_statement.codegen(module, builder, namespace)


def get_intrinsic(module: ir.Module, name: str) -> ir.Function:
    if name in module.globals:
        return module.globals[name]
    fntype = ir.FunctionType(ir.VoidType(), ())
    return ir.Function(module, fntype, name=name)


class PatternMatchArm:
    def __init__(self, pattern: Expression, body: Expression) -> None:
        self.pattern = pattern
        self.body = body


class PatternMatch(Statement):
    def __init__(self, match_value: Expression, arms: List[PatternMatchArm]) -> None:
        self.match_value = match_value
        self.arms = arms

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace) -> None:
        exit_block = builder.append_basic_block('match.exit')

        else_block = exit_block
        for i, arm in reversed(list(enumerate(self.arms))):
            label_prefix = f'match.arm{i}'
            condition_block = builder.append_basic_block(f'{label_prefix}.condition')
            body_block = builder.append_basic_block(f'{label_prefix}.body')
            test_expression = BinaryExpression(arm.pattern, '==', self.match_value)
            with builder.goto_block(condition_block):

                # TODO(optimization) evaluate the match_value expression only once
                test_value = test_expression.codegen(module, builder, namespace)
                builder.cbranch(test_value, body_block, else_block)

            with builder.goto_block(body_block):
                arm.body.codegen(module, builder, namespace)
                builder.branch(exit_block)

            else_block = condition_block

        builder.branch(condition_block)
        builder.position_at_end(exit_block)


class WhileLoop(Statement):
    def __init__(
        self,
        condition: Expression,
        body: Statement,
    ) -> None:
        self.condition = condition
        self.body = body

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace) -> None:
        loop_helper(module, builder, namespace, self.condition, self.body)


class ForLoop(Statement):
    def __init__(
        self,
        entry: Statement,
        condition: Expression,
        step: Statement,
        body: Statement,
    ) -> None:
        self.entry = entry
        self.condition = condition
        self.step = step
        self.body = body

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace) -> None:
        loop_namespace = Namespace(parents=[namespace])
        self.entry.codegen(module, builder, loop_namespace)
        actual_body = CompoundStatement([self.body, self.step])
        loop_helper(module, builder, loop_namespace, self.condition, actual_body)


def loop_helper(
    module: ir.Module,
    builder: ir.IRBuilder,
    namespace: Namespace,
    condition: Expression,
    body: Statement,
) -> None:
    assert isinstance(condition.type(namespace), BoolType)
    condition_block = builder.append_basic_block('loop.condition')
    body_block = builder.append_basic_block('loop.body')
    exit_block = builder.append_basic_block('loop.exit')

    builder.branch(condition_block)

    builder.position_at_end(condition_block)
    condition_value = condition.codegen(module, builder, namespace)
    builder.cbranch(condition_value, body_block, exit_block)

    builder.position_at_end(body_block)
    body.codegen(module, builder, namespace)
    builder.branch(condition_block)

    builder.position_at_end(exit_block)


class ReturnStatement(Statement):
    def __init__(self, expression: Expression = None) -> None:
        self.expression = expression

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace) -> None:
        if self.expression is not None:
            builder.ret(self.expression.codegen(module, builder, namespace, 'return_value'))
        else:
            builder.ret_void()


class VariableDeclaration(Statement):
    def __init__(
        self,
        name: str,
        expression: Optional[Expression],
        type_: str = None,
    ) -> None:
        assert expression is not None or type_ is not None, (expression, type_)
        self.name = name
        self.expression = expression
        self.type_ = type_

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace) -> None:
        type_ = namespace.get_type(self.type_) if self.type_ else cast(Expression, self.expression).type(namespace)
        memory = builder.alloca(type_.ir_type, name=self.name)
        namespace.add_item(Variable(self.name, type_, memory))
        if self.expression is not None:
            value = self.expression.codegen(module, builder, namespace)
            adapted_value = type_.adapt(builder, value, self.expression.type(namespace))
            builder.store(adapted_value, memory)


class Expression(Statement):
    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        raise NotImplementedError()

    def type(self, namespace: Namespace) -> Type:
        raise NotImplementedError(f'type() not implemented for {type(self)}')


class NegativeExpression(Expression):
    def __init__(self, expression: Expression) -> None:
        self.expression = expression

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        value = self.expression.codegen(module, builder, namespace, name)
        value.constant = -value.constant
        return value

    def type(self, namespace: Namespace) -> Type:
        return self.expression.type(namespace)


class BoolNegationExpression(Expression):
    def __init__(self, expression: Expression) -> None:
        self.expression = expression

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        assert self.expression.type(namespace).name == 'bool', self.expression

        value_to_negate = self.expression.codegen(module, builder, namespace)
        return builder.not_(value_to_negate, name=name)

    def type(self, namespace: Namespace) -> Type:
        return self.expression.type(namespace)


class BinaryExpression(Expression):
    def __init__(
        self,
        left_operand: Expression,
        operator: str,
        right_operand: Expression,
        name: str = '',
    ) -> None:
        self.left_operand = left_operand
        self.operator = operator
        self.right_operand = right_operand
        self.name = name

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        left_value = self.left_operand.codegen(module, builder, namespace)
        right_value = self.right_operand.codegen(module, builder, namespace)
        # TODO stop hardcoding those
        i8 = ir.IntType(8)
        i32 = ir.IntType(32)
        i64 = ir.IntType(64)
        f32 = ir.FloatType()
        f64 = ir.DoubleType()
        comparison_operators = {'<', '>', '<=', '>=', '==', '!='}
        if left_value.type == right_value.type:
            if left_value.type in [i8, i32, i64]:
                arithmetic_methods = {
                    '-': builder.sub,
                    '+': builder.add,
                    '*': builder.mul,
                    # TODO: what about unsigned div?
                    '/': builder.sdiv,
                }
                if self.operator in arithmetic_methods:
                    method = arithmetic_methods[self.operator]
                    return method(left_value, right_value, name=self.name)
                elif self.operator in comparison_operators:
                    return builder.icmp_signed(self.operator, left_value, right_value, name=self.name)
            elif left_value.type in {f32, f64}:
                arithmetic_methods = {
                    '-': builder.fsub,
                    '+': builder.fadd,
                    '*': builder.fmul,
                    '/': builder.fdiv,
                }
                if self.operator in arithmetic_methods:
                    method = arithmetic_methods[self.operator]
                    return method(left_value, right_value, name=self.name)
                else:
                    # TODO: decide if ordered is the right choice here
                    return builder.fcmp_ordered(self.operator, left_value, right_value, name=self.name)
        if (
            isinstance(left_value.type, ir.IntType)
            and isinstance(right_value.type, ir.IntType)
            and self.operator in comparison_operators
        ):
            extend_to = ir.IntType(max([left_value.type.width, right_value.type.width]))
            if left_value.type != extend_to:
                left_value = builder.sext(left_value, extend_to)
            else:
                right_value = builder.sext(right_value, extend_to)
            return builder.icmp_signed(self.operator, left_value, right_value, name=self.name)
        raise AssertionError(f'Invalid operand, operator, operand triple: ({left_value.type}, {right_value.type}, {self.operator})')  # noqa

    def type(self, namespace: Namespace) -> Type:
        if self.operator in {'<', '>', '<=', '>=', '==', '!='}:
            return namespace.get_type('bool')
        elif self.operator in {'+', '-', '*', '/'}:
            return self.left_operand.type(namespace)
        assert False, (self.operator, self.left_operand, self.right_operand)

    def __repr__(self) -> str:
        return f'type(self).__name__({repr(self.left_operand)}, {repr(self.operator)}, {repr(self.right_operand)})'


class FunctionCall(Expression):
    def __init__(self, name: str, parameters: List[Expression]) -> None:
        self.name = name
        self.parameters = parameters

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        function = namespace.get_function(self.name)
        if function.is_generic:
            namespace = namespace_for_specialized_function(namespace, function, self.parameters)

        parameter_names = [p.name for p in function.parameters]
        # TODO: handle not enough parameters here
        assert len(self.parameters) == len(parameter_names) or \
            function.parameters.variadic and len(self.parameters) > len(parameter_names), \
            (function.parameters, self.parameters, parameter_names)
        parameters = [
            p.codegen(module, builder, namespace, f'{self.name}.{n}')
            for (p, n) in zip_longest(self.parameters, parameter_names, fillvalue='vararg')
        ]

        assert len(self.parameters) >= len(parameter_names), (self.name, self.parameters)

        expected_parameter_types = [p.type_.codegen(namespace) for p in function.parameters]
        parameter_types = [p.type(namespace) for p in self.parameters]
        for i, (value, from_type, to_type) in enumerate(
            zip(parameters, parameter_types, expected_parameter_types),
        ):
            parameters[i] = to_type.adapt(builder, value, from_type)

        llvm_function = get_or_create_llvm_function(module, namespace, function)
        return builder.call(llvm_function, parameters, name=name)

    def type(self, namespace: Namespace) -> Type:
        return namespace.get_type('i64')


"""
def whatever<T>(T a) -> void ...
def whatever<T>(int a, T b) -> void ...
"""


def namespace_for_specialized_function(
    namespace: Namespace,
    function: Function,
    arguments: Collection[Expression],
) -> Namespace:
    mapping = {}  # type: Dict[str, Type]
    for parameter, expression in zip(function.parameters, arguments):
        assert isinstance(parameter.type_, BaseTypeReference), 'TODO support pointers etc. here'
        type_name = parameter.type_.name
        if type_name in function.type_parameters:
            deduced_type = expression.type(namespace)
            assert type_name not in mapping or mapping[type_name] == deduced_type
            mapping[type_name] = deduced_type

    new_namespace = Namespace(parents=[namespace])
    for name, type_ in mapping.items():
        new_namespace.add_item(type_, name)
    return new_namespace


class StructInstantiation(Expression):
    def __init__(self, name: str, parameters: List[Expression]) -> None:
        self.name = name
        self.parameters = parameters

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        struct = namespace.get_item(self.name, StructType)
        assert len(self.parameters) == len(struct.members)

        member_names = [m[0] for m in struct.members]
        memory = builder.alloca(struct.ir_type)
        value = builder.load(memory)
        for i, (p, n) in enumerate(zip(self.parameters, member_names)):
            member_value = p.codegen(module, builder, namespace, f'{self.name}.{n}')
            value = builder.insert_value(value, member_value, i)

        return value

    def type(self, namespace: Namespace) -> Type:
        return namespace.get_type(self.name)


class StringLiteral(Expression):
    def __init__(self, text: str) -> None:
        self.text = evaluate_escape_sequences(text)

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        return string_constant(module, builder, self.text[1:-1], namespace)

    def type(self, namespace: Namespace) -> Type:
        return namespace.get_type('i8').as_pointer()


def evaluate_escape_sequences(text: str) -> str:
    return text.replace(r'\n', '\n')


class IntegerLiteral(Expression):
    def __init__(self, text: str) -> None:
        self.text = text

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        value = int(self.text)
        return namespace.get_type('i64').ir_type(value)

    def type(self, namespace: Namespace) -> Type:
        return namespace.get_type('i64')


class FloatLiteral(Expression):
    def __init__(self, text: str) -> None:
        self.text = text

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        value = float(self.text)
        return namespace.get_type('f64').ir_type(value)

    def type(self, namespace: Namespace) -> Type:
        return namespace.get_type('f64')


class BoolLiteral(Expression):
    def __init__(self, text: str) -> None:
        assert text in {'false', 'true'}
        self.value = text == 'true'

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        return namespace.get_type('bool').ir_type(self.value)

    def type(self, namespace: Namespace) -> Type:
        return namespace.get_type('bool')


class MemoryReference(Expression):
    def get_pointer(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace) -> ir.Value:
        raise NotImplementedError()


class VariableReference(MemoryReference):
    def __init__(self, name: str) -> None:
        self.name = name

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        pointer = self.get_pointer(module, builder, namespace)
        return builder.load(pointer, name=name)

    def get_pointer(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace) -> ir.Value:
        return namespace.get_variable(self.name).value

    def type(self, namespace: Namespace) -> Type:
        return namespace.get_variable(self.name).type_


class AddressOf(MemoryReference):
    def __init__(self, variable: VariableReference) -> None:
        self.variable = variable

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        return self.variable.get_pointer(module, builder, namespace)

    def type(self, namespace: Namespace) -> Type:
        return namespace.get_variable(self.variable.name).type_.as_pointer()


class ValueAt(MemoryReference):
    def __init__(self, variable: VariableReference) -> None:
        self.variable = variable

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        pointer = self.variable.get_pointer(module, builder, namespace)
        pointer = builder.load(pointer)  # self.variable.codegen
        pointer = builder.load(pointer)
        return pointer

    def type(self, namespace: Namespace) -> Type:
        return namespace.get_variable(self.variable.name).type_.as_pointee()

    def get_pointer(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace) -> ir.Value:
        return self.variable.codegen(module, builder, namespace)


class Assignment(Statement):
    def __init__(self, target: MemoryReference, expression: Expression) -> None:
        self.target = target
        self.expression = expression

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> None:
        pointer = self.target.get_pointer(module, builder, namespace)
        value = self.expression.codegen(module, builder, namespace)
        destination_type = self.target.type(namespace)
        adapted_value = destination_type.adapt(builder, value, self.expression.type(namespace))
        builder.store(adapted_value, pointer)


class ArrayLiteral(Expression):
    def __init__(self, initializers: List[Expression]) -> None:
        assert len(initializers) > 0
        self.initializers = initializers

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        type_ = self.type(namespace)
        memory = builder.alloca(type_.ir_type, name=name)
        i64 = namespace.get_type('i64').ir_type

        for index, initializer in enumerate(self.initializers):
            indexed_memory = builder.gep(memory, (i64(0), i64(index),))
            value = initializer.codegen(module, builder, namespace)
            builder.store(value, indexed_memory)
        return builder.load(memory)

    def type(self, namespace: Namespace) -> Type:
        # TODO make sure all elements are of the same type or can be coerced to one
        element_type = self.initializers[0].type(namespace)
        return ArrayType(element_type, len(self.initializers))


class DotAccess(MemoryReference):
    def __init__(self, left_side: MemoryReference, member: str) -> None:
        self.left_side = left_side
        self.member = member

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        pointer = self.get_pointer(module, builder, namespace)
        return builder.load(pointer)

    def get_pointer(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace) -> ir.Value:
        left_type = self.left_side.type(namespace)
        assert isinstance(left_type, StructType)
        member_index = left_type.get_member_index(self.member)
        left_pointer = self.left_side.get_pointer(module, builder, namespace)
        # i32 is mandatory when indexing within a structure.
        # See http://llvm.org/docs/LangRef.html#getelementptr-instruction
        i32 = namespace.get_type('i32').ir_type
        i64 = namespace.get_type('i64').ir_type
        return builder.gep(left_pointer, (i64(0), i32(member_index),))

    def type(self, namespace: Namespace) -> Type:
        left_type = self.left_side.type(namespace)
        assert isinstance(left_type, StructType)
        return left_type.get_member_type(self.member)


class IndexAccess(MemoryReference):
    def __init__(self, pointer: MemoryReference, index: Expression) -> None:
        self.pointer = pointer
        self.index = index

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        pointer = self.get_pointer(module, builder, namespace)
        return builder.load(pointer)

    def get_pointer(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace) -> ir.Value:
        pointer_type = self.pointer.type(namespace)
        pointer = self.pointer.get_pointer(module, builder, namespace)
        index = self.index.codegen(module, builder, namespace)
        i64 = namespace.get_type('i64').ir_type
        # TODO remove conditional logic from here if possible
        if isinstance(pointer_type, PointerType):
            pointer = builder.load(pointer)
            return builder.gep(pointer, (index,))
        else:
            return builder.gep(pointer, (i64(0), index))

    def type(self, namespace: Namespace) -> Type:
        base_type = self.pointer.type(namespace)
        if isinstance(base_type, PointerType):
            return base_type.pointee
        elif isinstance(base_type, ArrayType):
            return base_type.element_type
        else:
            assert False, f'Bad memory reference: {self.pointer}'


class TypeReference:
    def codegen(self, namespace: Namespace) -> Type:
        raise NotImplementedError()

    def as_pointer(self) -> 'PointerTypeReference':
        return PointerTypeReference(self)

    def most_basic_type(self) -> 'BaseTypeReference':
        raise NotImplementedError()


@dataclass
class BaseTypeReference(TypeReference):
    name: str

    def codegen(self, namespace: Namespace) -> Type:
        return namespace.get_type(self.name)

    def most_basic_type(self) -> 'BaseTypeReference':
        return self


@dataclass
class PointerTypeReference(TypeReference):
    base: TypeReference

    def codegen(self, namespace: Namespace) -> ir.Type:
        return self.base.codegen(namespace).as_pointer()

    def most_basic_type(self) -> 'BaseTypeReference':
        return self.base.most_basic_type()


@dataclass
class Parameter:
    name: str
    type_: TypeReference


@dataclass
class ParameterList(Iterable[Parameter]):
    parameters: List[Parameter]
    variadic: bool = False

    def __iter__(self) -> Iterator[Parameter]:
        return iter(self.parameters)
