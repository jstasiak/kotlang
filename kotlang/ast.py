from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import zip_longest
from typing import (
    Any, cast, Collection, Dict, Iterable, Iterator, List,
    Mapping, MutableMapping, Optional,
    Tuple, Type as TypingType, TypeVar, Union as TypingUnion,
)

from llvmlite import ir

from kotlang import typesystem as ts
from kotlang.symbols import mangle


constant_counter = 0


def string_constant(module: ir.Module, builder: ir.IRBuilder, s: str, namespace: Namespace) -> ir.Constant:
    global constant_counter
    name = f'constant{constant_counter}'
    constant_counter += 1

    as_bytes = s.encode()
    array_type = ir.ArrayType(namespace.get_type('i8').get_ir_type(), len(as_bytes) + 1)
    global_value = ir.GlobalVariable(module, array_type, name)
    global_value.global_constant = True
    global_value.initializer = array_type(bytearray(as_bytes + b'\x00'))

    i64 = namespace.get_type('i64').get_ir_type()
    return builder.gep(global_value, (i64(0), i64(0)))


class Node:
    pass


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
    name: str
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
    name: str
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
        ref = FunctionTypeReference([p.type_ for p in self.parameters], self.return_type, self.parameters.variadic)
        return ref.codegen(namespace)


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
        ft = function.get_type(namespace)
        ir_ft = ft.get_ir_type()

        llvm_function = ir.Function(module, ir_ft, name=symbol_name)
        for i, (p, arg) in enumerate(zip(function.parameters, llvm_function.args)):
            arg.name = (p.name or f'param{i + 1}') + '_arg'

        if function.code_block is not None:
            block = llvm_function.append_basic_block(name="entry")
            builder = ir.IRBuilder(block)

            function_namespace = Namespace(parents=[namespace])
            parameter_types = zip(function.parameters, ft.parameter_types)
            for i, (pt, arg) in enumerate(zip(parameter_types, llvm_function.args)):
                (parameter, parameter_type) = pt
                memory = builder.alloca(arg.type, name=parameter.name)
                builder.store(arg, memory)
                function_namespace.add_value(Variable(parameter.name or f'param{i + 1}', parameter_type, memory))

            function.code_block.codegen(module, builder, function_namespace)
            if ft.return_type == ts.void:
                builder.ret_void()
            else:
                # FIXME: We depend on having returned already but this is not ensured
                if not builder.block.is_terminated:
                    builder.unreachable()

    return llvm_function


@dataclass
class Module:
    types: List[TypeDefinition]
    functions: List[Function]
    imports: List[Tuple[str, Module]]
    variables: List[VariableDeclaration]

    def codegen_top_level(self, name: str) -> ir.Module:
        builtin_namespace = Namespace()
        builtin_namespace.add_type(ts.VoidType())
        builtin_namespace.add_type(ts.BoolType())
        for signed in {True, False}:
            for bits in {8, 16, 32, 64}:
                builtin_namespace.add_type(ts.IntType(bits, signed=signed))
        for bits in [32, 64, 80]:
            builtin_namespace.add_type(ts.FloatType(bits))

        module = ir.Module(name=name)
        self.codegen(module, builtin_namespace, MetaNamespace(), name.split('/')[1].replace('.kot', ''))
        return module

    def codegen(
        self,
        module: ir.Module,
        parent_namespace: Namespace,
        meta_namespace: MetaNamespace,
        module_name: str,
    ) -> Namespace:
        import_namespaces: List[Namespace] = []
        for name, i in self.imports:
            # FIXME we currently need to special-case the c module where we put bits
            # imported from C headers. If we remove this special handling one set of headers imported by
            # one module would overshadow another set of headers in another module processed later.
            # This whole MetaNamespace hack needs to go most likely.
            if meta_namespace.has(name) and name != 'c':
                imported_namespace = meta_namespace.get(name)
            else:
                # HACK c imports are "kinda" kot modules but not really
                imported_module_name = name if name != 'c' else module_name + '_c'

                imported_namespace = i.codegen(module, parent_namespace, meta_namespace, imported_module_name)
                if name != 'c':
                    meta_namespace.set(name, imported_namespace)
            import_namespaces.append(imported_namespace)

        module_namespace = Namespace(parents=[parent_namespace, *import_namespaces])

        definitions_types = [(td, td.get_dummy_type()) for td in self.types]
        for _, t in definitions_types:
            module_namespace.add_type(t)
        for td, t in definitions_types:
            td.fill_type_members(module_namespace, t)

        for f in self.functions:
            module_namespace.add_function(f)

        for variable_declaration in self.variables:
            variable_declaration.codegen_module_level(module, module_namespace, module_name)

        nongeneric_functions = (f for f in self.functions if not f.is_generic)
        for f in nongeneric_functions:
            get_or_create_llvm_function(module, module_namespace, f)

        return module_namespace


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
        second_statement: Optional[Statement] = None,
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
    assert isinstance(condition.type(namespace), ts.BoolType)
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
    def __init__(self, expression: Optional[Expression] = None) -> None:
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
        type_: Optional[TypeReference] = None,
    ) -> None:
        assert expression is not None or type_ is not None, (expression, type_)
        self.name = name
        self.expression = expression
        self.type_ = type_

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace) -> None:
        type_ = self.variable_type(namespace)
        ir_type = type_.get_ir_type()
        if isinstance(type_, ts.FunctionType):
            # TODO: now our typesystem things we're dealing with functions while actually we're
            # dealing with function pointers. See if this can be ironed out. If it can't then see
            # if the abstraction is right.
            ir_type = ir_type.as_pointer()
        memory = builder.alloca(ir_type, name=self.name)
        namespace.add_value(Variable(self.name, type_, memory))
        if self.expression is not None:
            value = self.expression.codegen(module, builder, namespace)
            adapted_value = type_.adapt(builder, value, self.expression.type(namespace))
            builder.store(adapted_value, memory)

    def codegen_module_level(self, module: ir.Module, namespace: Namespace, module_name: str) -> None:
        value = self.expression.get_constant_time_value() if self.expression else None
        type_ = self.variable_type(namespace)
        constant = ir.Constant(type_.get_ir_type(), value)
        variable = ir.GlobalVariable(module, type_.get_ir_type(), mangle([module_name, self.name]))
        variable.initializer = constant
        namespace.add_value(Variable(self.name, type_, variable))

    def variable_type(self, namespace: Namespace) -> ts.Type:
        return (
            self.type_.codegen(namespace) if self.type_ is not None
            else cast(Expression, self.expression).type(namespace)
        )


class Expression(Statement):
    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        raise NotImplementedError()

    def type(self, namespace: Namespace) -> ts.Type:
        raise NotImplementedError(f'type() not implemented for {type(self)}')

    def get_pointer(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace) -> ir.Value:
        raise AssertionError(f'{type(self).__name__} cannot be used as a l-value nor can you grab its address')

    def get_constant_time_value(self) -> Any:
        raise NotImplementedError(f'{0} is not a compile-time constant')


class NegativeExpression(Expression):
    def __init__(self, expression: Expression) -> None:
        self.expression = expression

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        value = self.expression.codegen(module, builder, namespace, name)
        value.constant = -value.constant
        return value

    def type(self, namespace: Namespace) -> ts.Type:
        return self.expression.type(namespace)


class BoolNegationExpression(Expression):
    def __init__(self, expression: Expression) -> None:
        self.expression = expression

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        assert self.expression.type(namespace).name == 'bool', self.expression

        value_to_negate = self.expression.codegen(module, builder, namespace)
        return builder.not_(value_to_negate, name=name)

    def type(self, namespace: Namespace) -> ts.Type:
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

    def type(self, namespace: Namespace) -> ts.Type:
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
        function: TypingUnion[Function, Variable]
        parameter_names: List[str]
        try:
            function = namespace.get_function(self.name)
        except KeyError:
            function = namespace.get_value(self.name)
            assert isinstance(function, Variable)
            assert isinstance(function.type_, ts.FunctionType)
            parameter_types = function.type_.parameter_types
            # TODO provide parameter names here somehow? We don't have them right now.
            parameter_names = []
            llvm_function = builder.load(function.value)
            ft = function.type_
        else:
            if function.is_generic:
                namespace = namespace_for_specialized_function(namespace, function, self.parameters)

            ft = function.get_type(namespace)
            parameter_types = ft.parameter_types
            # TODO: eliminate this "or ''" below
            parameter_names = [p.name or '' for p in function.parameters]
            llvm_function = get_or_create_llvm_function(module, namespace, function)

        # TODO: handle not enough parameters here
        assert len(self.parameters) == len(parameter_types) or \
            ft.variadic and len(self.parameters) > len(parameter_types), \
            (ft, self.parameters)
        parameter_values = [
            p.codegen(module, builder, namespace, f'{self.name}.{n}')
            for (p, n) in zip_longest(self.parameters, parameter_names, fillvalue='arg')
        ]

        assert len(self.parameters) >= len(parameter_names), (self.name, self.parameters)

        provided_parameter_types = [p.type(namespace) for p in self.parameters]
        for i, (value, from_type, to_type) in enumerate(
            zip(parameter_values, provided_parameter_types, parameter_types),
        ):
            parameter_values[i] = to_type.adapt(builder, value, from_type)

        return builder.call(llvm_function, parameter_values, name=name)

    def type(self, namespace: Namespace) -> ts.Type:
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
    mapping: Dict[str, ts.Type] = {}
    for parameter, expression in zip(function.parameters, arguments):
        assert isinstance(parameter.type_, BaseTypeReference), 'TODO support pointers etc. here'
        type_name = parameter.type_.name
        if type_name in function.type_parameters:
            deduced_type = expression.type(namespace)
            assert type_name not in mapping or mapping[type_name] == deduced_type
            mapping[type_name] = deduced_type

    new_namespace = Namespace(parents=[namespace])
    for name, type_ in mapping.items():
        new_namespace.add_type(type_, name)
    return new_namespace


class StructInstantiation(Expression):
    def __init__(self, name: str, parameters: List[Expression]) -> None:
        self.name = name
        self.parameters = parameters

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        struct = namespace.get_type(self.name)
        assert isinstance(struct, ts.StructType)
        assert len(self.parameters) == len(struct.members)

        member_names = [m[0] for m in struct.members]
        memory = builder.alloca(struct.get_ir_type())
        value = builder.load(memory)
        for i, (p, n) in enumerate(zip(self.parameters, member_names)):
            member_value = p.codegen(module, builder, namespace, f'{self.name}.{n}')
            value = builder.insert_value(value, member_value, i)

        return value

    def type(self, namespace: Namespace) -> ts.Type:
        return namespace.get_type(self.name)


class StringLiteral(Expression):
    def __init__(self, text: str) -> None:
        self.text = evaluate_escape_sequences(text)

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        return string_constant(module, builder, self.text[1:-1], namespace)

    def type(self, namespace: Namespace) -> ts.Type:
        return namespace.get_type('i8').as_pointer()


def evaluate_escape_sequences(text: str) -> str:
    return text.replace(r'\n', '\n')


class IntegerLiteral(Expression):
    def __init__(self, text: str) -> None:
        self.text = text

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        value = int(self.text)
        return namespace.get_type('i64').get_ir_type()(value)

    def type(self, namespace: Namespace) -> ts.Type:
        return namespace.get_type('i64')

    def get_constant_time_value(self) -> Any:
        return int(self.text)


class FloatLiteral(Expression):
    def __init__(self, text: str) -> None:
        self.text = text

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        value = float(self.text)
        return namespace.get_type('f64').get_ir_type()(value)

    def type(self, namespace: Namespace) -> ts.Type:
        return namespace.get_type('f64')


class BoolLiteral(Expression):
    def __init__(self, text: str) -> None:
        assert text in {'false', 'true'}
        self.value = text == 'true'

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        return namespace.get_type('bool').get_ir_type()(self.value)

    def type(self, namespace: Namespace) -> ts.Type:
        return namespace.get_type('bool')


class MemoryReference(Expression):
    def get_pointer(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace) -> ir.Value:
        raise NotImplementedError()


class VariableReference(MemoryReference):
    def __init__(self, name: str) -> None:
        self.name = name

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        type_ = self.type(namespace)
        pointer = self.get_pointer(module, builder, namespace)
        # The first part of this condition makes sure we keep referring to functions by their pointers.
        # The second makes it so that if we're referring to a variable already (pointer here is a pointer to
        # a pointer to a function) we actually dereference it once.
        if isinstance(type_, ts.FunctionType) and not isinstance(pointer.type.pointee, ir.PointerType):
            return pointer
        return builder.load(pointer, name=name)

    def get_pointer(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace) -> ir.Value:
        value: TypingUnion[Function, Variable]
        try:
            value = namespace.get_function(self.name)
        except KeyError:
            value = namespace.get_value(self.name)
            return value.value
        else:
            return get_or_create_llvm_function(module, namespace, value)

    def type(self, namespace: Namespace) -> ts.Type:
        value: TypingUnion[Function, Variable]
        try:
            function = namespace.get_function(self.name)
        except KeyError:
            variable = namespace.get_value(self.name)
            return variable.type_
        else:
            return function.get_type(namespace)


class AddressOf(MemoryReference):
    def __init__(self, variable: VariableReference) -> None:
        self.variable = variable

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        return self.variable.get_pointer(module, builder, namespace)

    def type(self, namespace: Namespace) -> ts.Type:
        return namespace.get_value(self.variable.name).type_.as_pointer()


class ValueAt(MemoryReference):
    def __init__(self, variable: VariableReference) -> None:
        self.variable = variable

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        pointer = self.variable.get_pointer(module, builder, namespace)
        pointer = builder.load(pointer)  # self.variable.codegen
        pointer = builder.load(pointer)
        return pointer

    def type(self, namespace: Namespace) -> ts.Type:
        return namespace.get_value(self.variable.name).type_.as_pointee()

    def get_pointer(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace) -> ir.Value:
        return self.variable.codegen(module, builder, namespace)


class Assignment(Expression):
    def __init__(self, target: Expression, expression: Expression) -> None:
        self.target = target
        self.expression = expression

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        pointer = self.target.get_pointer(module, builder, namespace)
        value = self.expression.codegen(module, builder, namespace)
        destination_type = self.target.type(namespace)
        adapted_value = destination_type.adapt(builder, value, self.expression.type(namespace))
        builder.store(adapted_value, pointer)
        return value

    def type(self, namespace: Namespace) -> ts.Type:
        # TODO and possibly quite important - type of expression can be different than the type of the target
        # (for example expression of type i8 assigned to i64 location) - which one should we use?
        # For now we take the original value but it may not be expected or desired.
        return self.expression.type(namespace)


class ArrayLiteral(Expression):
    def __init__(self, initializers: List[Expression]) -> None:
        assert len(initializers) > 0
        self.initializers = initializers

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        type_ = self.type(namespace)
        memory = builder.alloca(type_.get_ir_type(), name=name)
        i64 = namespace.get_type('i64').get_ir_type()

        for index, initializer in enumerate(self.initializers):
            indexed_memory = builder.gep(memory, (i64(0), i64(index),))
            value = initializer.codegen(module, builder, namespace)
            builder.store(value, indexed_memory)
        return builder.load(memory)

    def type(self, namespace: Namespace) -> ts.Type:
        # TODO make sure all elements are of the same type or can be coerced to one
        element_type = self.initializers[0].type(namespace)
        return ts.ArrayType(element_type, len(self.initializers))


class DotAccess(MemoryReference):
    def __init__(self, left_side: MemoryReference, member: str) -> None:
        self.left_side = left_side
        self.member = member

    def codegen(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = '') -> ir.Value:
        pointer = self.get_pointer(module, builder, namespace)
        return builder.load(pointer)

    def get_pointer(self, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace) -> ir.Value:
        left_type = self.left_side.type(namespace)
        assert isinstance(left_type, ts.DottableType), left_type
        left_pointer = self.left_side.get_pointer(module, builder, namespace)
        return left_type.get_member_pointer(builder, left_pointer, self.member)

    def type(self, namespace: Namespace) -> ts.Type:
        left_type = self.left_side.type(namespace)
        assert isinstance(left_type, ts.DottableType), left_type
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
        i64 = namespace.get_type('i64').get_ir_type()
        # TODO remove conditional logic from here if possible
        if isinstance(pointer_type, ts.PointerType):
            pointer = builder.load(pointer)
            return builder.gep(pointer, (index,))
        else:
            return builder.gep(pointer, (i64(0), index))

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
