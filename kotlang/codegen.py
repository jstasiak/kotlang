from __future__ import annotations

from dataclasses import dataclass, field
from itertools import zip_longest
from typing import Any, cast, Collection, Dict, List, Optional, TypeVar, Union

from llvmlite import ir

from kotlang import ast, typesystem as ts
from kotlang.symbols import mangle


def codegen_module(
    node: ast.Module, module: ir.Module, parent_namespaces: List[Namespace], module_name: str
) -> Namespace:
    module_namespace = Namespace(parents=parent_namespaces)

    definitions_types = [(td, ts.StructUnionType(td.name.text, [], td.is_union)) for td in node.types]
    for _, t in definitions_types:
        module_namespace.add_type(t)
    for td, t in definitions_types:
        fill_structunion_members(td, module_namespace, t)

    for f in node.functions:
        module_namespace.add_function(f)

    for variable_declaration in node.variables:
        codegen_variable_module_level(variable_declaration, module, module_namespace, module_name)

    nongeneric_functions = (f for f in node.functions if not f.is_generic)
    for f in nongeneric_functions:
        get_or_create_llvm_function(module, module_namespace, f)

    return module_namespace


def fill_structunion_members(node: ast.StructUnion, namespace: Namespace, type_: ts.StructUnionType) -> None:
    # Note: This method mutates type_
    assert isinstance(type_, ts.StructUnionType)
    members = [(m.name.text, resolve_type(cast(ast.TypeReference, m.type_), namespace)) for m in node.members]
    type_.members = members


def codegen_statement(  # noqa: C901
    node: ast.Statement, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace
) -> None:
    if isinstance(node, ast.CompoundStatement):
        for s in node.statements:
            codegen_statement(s, module, builder, namespace)
    elif isinstance(node, ast.CodeBlock):
        block_namespace = Namespace(parents=[namespace])
        for s in node.statements:
            codegen_statement(s, module, builder, block_namespace)
    elif isinstance(node, ast.IfStatement):
        expression_value = codegen_expression(node.expression, module, builder, namespace)
        with builder.if_else(expression_value) as (then, otherwise):
            with then:
                codegen_statement(node.first_statement, module, builder, namespace)
            with otherwise:
                if node.second_statement is not None:
                    codegen_statement(node.second_statement, module, builder, namespace)
    elif isinstance(node, ast.PatternMatch):
        exit_block = builder.append_basic_block('match.exit')

        else_block = exit_block
        for i, arm in reversed(list(enumerate(node.arms))):
            label_prefix = f'match.arm{i}'
            condition_block = builder.append_basic_block(f'{label_prefix}.condition')
            body_block = builder.append_basic_block(f'{label_prefix}.body')
            test_expression = ast.BinaryExpression(arm.pattern, '==', node.match_value)
            with builder.goto_block(condition_block):

                # TODO(optimization) evaluate the match_value expression only once
                test_value = codegen_expression(test_expression, module, builder, namespace)
                builder.cbranch(test_value, body_block, else_block)

            with builder.goto_block(body_block):
                codegen_statement(arm.body, module, builder, namespace)
                builder.branch(exit_block)

            else_block = condition_block

        builder.branch(condition_block)
        builder.position_at_end(exit_block)
    elif isinstance(node, ast.WhileLoop):
        loop_helper(module, builder, namespace, node.condition, node.body)
    elif isinstance(node, ast.ForLoop):
        loop_namespace = Namespace(parents=[namespace])
        codegen_statement(node.entry, module, builder, loop_namespace)
        actual_body = ast.CompoundStatement([node.body, node.step])
        loop_helper(module, builder, loop_namespace, node.condition, actual_body)
    elif isinstance(node, ast.ReturnStatement):
        if node.expression is not None:
            builder.ret(codegen_expression(node.expression, module, builder, namespace, 'return_value'))
        else:
            builder.ret_void()
    elif isinstance(node, ast.VariableDeclaration):
        type_ = variable_type(node, namespace)
        ir_type = type_.get_ir_type()
        if isinstance(type_, ts.FunctionType):
            # TODO: now our typesystem things we're dealing with functions while actually we're
            # dealing with function pointers. See if this can be ironed out. If it can't then see
            # if the abstraction is right.
            ir_type = ir_type.as_pointer()
        memory = builder.alloca(ir_type, name=node.name.text)
        namespace.add_value(Variable(node.name.text, type_, memory))
        if node.expression is not None:
            value = codegen_expression(node.expression, module, builder, namespace)
            adapted_value = type_.adapt(builder, value, expression_type(node.expression, namespace))
            builder.store(adapted_value, memory)
    elif isinstance(node, ast.Expression):
        codegen_expression(node, module, builder, namespace)
    else:
        raise NotImplementedError(f'Code generation not implemented for {type(node)}')


def codegen_expression(  # noqa: C901
    node: ast.Expression, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace, name: str = ''
) -> ir.Value:
    if isinstance(node, ast.NegativeExpression):
        value = codegen_expression(node.expression, module, builder, namespace, name)
        value.constant = -value.constant
        return value
    elif isinstance(node, ast.BoolNegationExpression):
        assert expression_type(node.expression, namespace).name == 'bool', node.expression

        value_to_negate = codegen_expression(node.expression, module, builder, namespace)
        return builder.not_(value_to_negate, name=name)
    elif isinstance(node, ast.BinaryExpression):
        left_value = codegen_expression(node.left_operand, module, builder, namespace)
        right_value = codegen_expression(node.right_operand, module, builder, namespace)
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
                if node.operator in arithmetic_methods:
                    method = arithmetic_methods[node.operator]
                    return method(left_value, right_value, name=node.name)
                elif node.operator in comparison_operators:
                    return builder.icmp_signed(node.operator, left_value, right_value, name=node.name)
            elif left_value.type in {f32, f64}:
                arithmetic_methods = {
                    '-': builder.fsub,
                    '+': builder.fadd,
                    '*': builder.fmul,
                    '/': builder.fdiv,
                }
                if node.operator in arithmetic_methods:
                    method = arithmetic_methods[node.operator]
                    return method(left_value, right_value, name=node.name)
                else:
                    # TODO: decide if ordered is the right choice here
                    return builder.fcmp_ordered(node.operator, left_value, right_value, name=node.name)
        if (
            isinstance(left_value.type, ir.IntType)
            and isinstance(right_value.type, ir.IntType)
            and node.operator in comparison_operators
        ):
            extend_to = ir.IntType(max([left_value.type.width, right_value.type.width]))
            if left_value.type != extend_to:
                left_value = builder.sext(left_value, extend_to)
            else:
                right_value = builder.sext(right_value, extend_to)
            return builder.icmp_signed(node.operator, left_value, right_value, name=node.name)
        raise AssertionError(
            f'Invalid operand, operator, operand triple: ({left_value.type}, {right_value.type}, {node.operator})'
        )  # noqa
    elif isinstance(node, ast.FunctionCall):
        function: Union[ast.Function, Variable]
        parameter_names: List[str]
        try:
            function = namespace.get_function(node.name)
        except KeyError:
            function = namespace.get_value(node.name)
            assert isinstance(function, Variable)
            assert isinstance(function.type_, ts.FunctionType)
            parameter_types = function.type_.parameter_types
            # TODO provide parameter names here somehow? We don't have them right now.
            parameter_names = []
            llvm_function = builder.load(function.value)
            ft = function.type_
        else:
            if function.is_generic:
                namespace = namespace_for_specialized_function(namespace, function, node.parameters)

            ft = get_function_type(function, namespace)
            parameter_types = ft.parameter_types
            # TODO: eliminate this "or ''" below
            parameter_names = [p.name or '' for p in function.parameters]
            llvm_function = get_or_create_llvm_function(module, namespace, function)

        # TODO: handle not enough parameters here
        assert (
            len(node.parameters) == len(parameter_types)
            or ft.variadic
            and len(node.parameters) > len(parameter_types)
        ), (ft, node.parameters)
        parameter_values = [
            codegen_expression(p, module, builder, namespace, f'{node.name}.{n}')
            for (p, n) in zip_longest(node.parameters, parameter_names, fillvalue='arg')
        ]

        assert len(node.parameters) >= len(parameter_names), (node.name, node.parameters)

        provided_parameter_types = [expression_type(p, namespace) for p in node.parameters]
        for i, (value, from_type, to_type) in enumerate(
            zip(parameter_values, provided_parameter_types, parameter_types)
        ):
            parameter_values[i] = to_type.adapt(builder, value, from_type)

        return builder.call(llvm_function, parameter_values, name=name)
    elif isinstance(node, ast.StructInstantiation):
        struct = namespace.get_type(node.name)
        assert isinstance(struct, ts.StructUnionType)
        assert len(node.parameters) == len(struct.members)
        assert not struct.is_union

        member_names = [m[0] for m in struct.members]
        memory = builder.alloca(struct.get_ir_type())
        value = builder.load(memory)
        for i, (p, n) in enumerate(zip(node.parameters, member_names)):
            member_value = codegen_expression(p, module, builder, namespace, f'{node.name}.{n}')
            value = builder.insert_value(value, member_value, i)

        return value
    elif isinstance(node, ast.StringLiteral):
        return string_constant(module, builder, node.text, namespace)
    elif isinstance(node, ast.IntegerLiteral):
        value = int(node.text)
        return namespace.get_type('i64').get_ir_type()(value)
    elif isinstance(node, ast.FloatLiteral):
        value = float(node.text)
        return namespace.get_type('f64').get_ir_type()(value)
    elif isinstance(node, ast.BoolLiteral):
        return namespace.get_type('bool').get_ir_type()(node.value)
    elif isinstance(node, ast.VariableReference):
        type_ = expression_type(node, namespace)
        pointer = get_pointer(node, module, builder, namespace)
        # The first part of this condition makes sure we keep referring to functions by their pointers.
        # The second makes it so that if we're referring to a variable already (pointer here is a pointer to
        # a pointer to a function) we actually dereference it once.
        if isinstance(type_, ts.FunctionType) and not isinstance(pointer.type.pointee, ir.PointerType):
            return pointer
        return builder.load(pointer, name=name)
    elif isinstance(node, ast.AddressOf):
        return get_pointer(node.variable, module, builder, namespace)
    elif isinstance(node, ast.ValueAt):
        pointer = get_pointer(node.variable, module, builder, namespace)
        pointer = builder.load(pointer)  # codegen of node.variable
        pointer = builder.load(pointer)
        return pointer
    elif isinstance(node, ast.Assignment):
        pointer = get_pointer(node.target, module, builder, namespace)
        value = codegen_expression(node.expression, module, builder, namespace)
        destination_type = expression_type(node.target, namespace)
        adapted_value = destination_type.adapt(builder, value, expression_type(node.expression, namespace))
        builder.store(adapted_value, pointer)
        return value
    elif isinstance(node, ast.ArrayLiteral):
        type_ = expression_type(node, namespace)
        memory = builder.alloca(type_.get_ir_type(), name=name)
        i64 = namespace.get_type('i64').get_ir_type()

        for index, initializer in enumerate(node.initializers):
            indexed_memory = builder.gep(memory, (i64(0), i64(index)))
            value = codegen_expression(initializer, module, builder, namespace)
            builder.store(value, indexed_memory)
        return builder.load(memory)
    elif isinstance(node, ast.DotAccess):
        pointer = get_pointer(node, module, builder, namespace)
        return builder.load(pointer)
    elif isinstance(node, ast.IndexAccess):
        pointer = get_pointer(node, module, builder, namespace)
        return builder.load(pointer)
    else:
        raise NotImplementedError()


def loop_helper(
    module: ir.Module,
    builder: ir.IRBuilder,
    namespace: Namespace,
    condition: ast.Expression,
    body: ast.Statement,
) -> None:
    assert isinstance(expression_type(condition, namespace), ts.BoolType)
    condition_block = builder.append_basic_block('loop.condition')
    body_block = builder.append_basic_block('loop.body')
    exit_block = builder.append_basic_block('loop.exit')

    builder.branch(condition_block)

    builder.position_at_end(condition_block)
    condition_value = codegen_expression(condition, module, builder, namespace)
    builder.cbranch(condition_value, body_block, exit_block)

    builder.position_at_end(body_block)
    codegen_statement(body, module, builder, namespace)
    builder.branch(condition_block)

    builder.position_at_end(exit_block)


def namespace_for_specialized_function(
    namespace: Namespace, function: ast.Function, arguments: Collection[ast.Expression]
) -> Namespace:
    mapping: Dict[str, ts.Type] = {}
    function_type_parameters_by_text = {n.text: n for n in function.type_parameters}
    for parameter, expression in zip(function.parameters, arguments):
        assert isinstance(parameter.type_, ast.BaseTypeReference), 'TODO support pointers etc. here'
        type_name = parameter.type_.name
        if type_name in function_type_parameters_by_text:
            deduced_type = expression_type(expression, namespace)
            assert type_name not in mapping or mapping[type_name] == deduced_type
            mapping[type_name] = deduced_type

    new_namespace = Namespace(parents=[namespace])
    for name, type_ in mapping.items():
        new_namespace.add_type(type_, name)
    return new_namespace


def get_or_create_llvm_function(
    module: ir.Module, namespace: Namespace, function: ast.Function
) -> ir.Function:
    symbol_name = function_symbol_name(function, namespace)
    try:
        llvm_function = module.globals[symbol_name]
        assert isinstance(llvm_function, ir.Function)
    except KeyError:
        ft = get_function_type(function, namespace)
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
                function_namespace.add_value(
                    Variable(parameter.name or f'param{i + 1}', parameter_type, memory)
                )

            codegen_statement(function.code_block, module, builder, function_namespace)
            if ft.return_type == ts.void:
                builder.ret_void()
            else:
                # FIXME: We depend on having returned already but this is not ensured
                if not builder.block.is_terminated:
                    builder.unreachable()

    return llvm_function


def function_symbol_name(node: ast.Function, namespace: Namespace) -> str:
    # TODO: stop hardcoding this?
    if node.code_block is None or node.name.text == 'main':
        return node.name.text

    type_values = [namespace.get_type(t.text).name for t in node.type_parameters]
    return mangle([node.name.text] + type_values)


def get_function_type(node: ast.Function, namespace: Namespace) -> ts.FunctionType:
    ref = ast.FunctionTypeReference(
        [p.type_ for p in node.parameters], node.return_type, node.parameters.variadic
    )
    return cast(ts.FunctionType, resolve_type(ref, namespace))


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


def get_pointer(
    node: ast.Expression, module: ir.Module, builder: ir.IRBuilder, namespace: Namespace
) -> ir.Value:
    if isinstance(node, ast.VariableReference):
        value: Union[ast.Function, Variable]
        try:
            value = namespace.get_function(node.name)
        except KeyError:
            value = namespace.get_value(node.name)
            return value.value
        else:
            return get_or_create_llvm_function(module, namespace, value)
    elif isinstance(node, ast.ValueAt):
        return codegen_expression(node.variable, module, builder, namespace)
    elif isinstance(node, ast.DotAccess):
        left_type = expression_type(node.left_side, namespace)
        assert isinstance(left_type, ts.StructUnionType), left_type
        left_pointer = get_pointer(node.left_side, module, builder, namespace)
        return left_type.get_member_pointer(builder, left_pointer, node.member)
    elif isinstance(node, ast.IndexAccess):
        pointer_type = expression_type(node.pointer, namespace)
        pointer = get_pointer(node.pointer, module, builder, namespace)
        index = codegen_expression(node.index, module, builder, namespace)
        i64 = namespace.get_type('i64').get_ir_type()
        # TODO remove conditional logic from here if possible
        if isinstance(pointer_type, ts.PointerType):
            pointer = builder.load(pointer)
            return builder.gep(pointer, (index,))
        else:
            return builder.gep(pointer, (i64(0), index))
    else:
        raise AssertionError(
            f'{type(node).__name__} cannot be used as a l-value nor can you grab its address'
        )


def expression_type(node: ast.Expression, namespace: Namespace) -> ts.Type:  # noqa: C901
    if isinstance(node, (ast.NegativeExpression, ast.BoolNegationExpression)):
        return expression_type(node.expression, namespace)
    elif isinstance(node, ast.BinaryExpression):
        if node.operator in {'<', '>', '<=', '>=', '==', '!='}:
            return namespace.get_type('bool')
        elif node.operator in {'+', '-', '*', '/'}:
            return expression_type(node.left_operand, namespace)
        assert False, (node.operator, node.left_operand, node.right_operand)
    elif isinstance(node, ast.FunctionCall):
        return namespace.get_type('i64')
    elif isinstance(node, ast.StructInstantiation):
        return namespace.get_type(node.name)
    elif isinstance(node, ast.StringLiteral):
        return namespace.get_type('i8').as_pointer()
    elif isinstance(node, ast.IntegerLiteral):
        return namespace.get_type('i64')
    elif isinstance(node, ast.FloatLiteral):
        return namespace.get_type('f64')
    elif isinstance(node, ast.BoolLiteral):
        return namespace.get_type('bool')
    elif isinstance(node, ast.VariableReference):
        value: Union[ast.Function, Variable]
        try:
            function = namespace.get_function(node.name)
        except KeyError:
            variable = namespace.get_value(node.name)
            return variable.type_
        else:
            return get_function_type(function, namespace)
    elif isinstance(node, ast.AddressOf):
        return namespace.get_value(node.variable.name).type_.as_pointer()
    elif isinstance(node, ast.ValueAt):
        return namespace.get_value(node.variable.name).type_.as_pointee()
    elif isinstance(node, ast.Assignment):
        # TODO and possibly quite important - type of expression can be different than the type of the target
        # (for example expression of type i8 assigned to i64 location) - which one should we use?
        # For now we take the original value but it may not be expected or desired.
        return expression_type(node.expression, namespace)
    elif isinstance(node, ast.ArrayLiteral):
        # TODO make sure all elements are of the same type or can be coerced to one
        element_type = expression_type(node.initializers[0], namespace)
        return ts.ArrayType(element_type, len(node.initializers))
    elif isinstance(node, ast.DotAccess):
        left_type = expression_type(node.left_side, namespace)
        assert isinstance(left_type, ts.StructUnionType), left_type
        return left_type.get_member_type(node.member)
    elif isinstance(node, ast.IndexAccess):
        base_type = expression_type(node.pointer, namespace)
        if isinstance(base_type, ts.PointerType):
            return base_type.pointee
        elif isinstance(base_type, ts.ArrayType):
            return base_type.element_type
        else:
            assert False, f'Bad memory reference: {node.pointer}'
    else:
        raise AssertionError(f'{type(node).__name__} cannot be used here')


def codegen_variable_module_level(
    node: ast.VariableDeclaration, module: ir.Module, namespace: Namespace, module_name: str
) -> None:
    value = node.expression.get_constant_time_value() if node.expression else None
    type_ = variable_type(node, namespace)
    constant = ir.Constant(type_.get_ir_type(), value)
    variable = ir.GlobalVariable(module, type_.get_ir_type(), mangle([module_name, node.name.text]))
    variable.initializer = constant
    variable.global_constant = True
    namespace.add_value(Variable(node.name.text, type_, variable))


def variable_type(node: ast.VariableDeclaration, namespace: Namespace) -> ts.Type:
    return (
        resolve_type(node.type_, namespace)
        if node.type_ is not None
        else expression_type(cast(ast.Expression, node.expression), namespace)
    )


def resolve_type(node: ast.TypeReference, namespace: Namespace) -> ts.Type:
    if isinstance(node, ast.BaseTypeReference):
        return namespace.get_type(node.name)
    elif isinstance(node, ast.ArrayTypeReference):
        assert isinstance(node.length, ast.IntegerLiteral)
        base_type = resolve_type(node.base, namespace)
        return ts.ArrayType(base_type, int(node.length.text))
    elif isinstance(node, ast.FunctionTypeReference):
        return_type = resolve_type(node.return_type, namespace)
        parameter_types = [resolve_type(t, namespace) for t in node.parameter_types]
        return ts.FunctionType(parameter_types, return_type, node.variadic)
    elif isinstance(node, ast.PointerTypeReference):
        return resolve_type(node.base, namespace).as_pointer()
    else:
        raise AssertionError(f'{type(node).__name__} cannot be used here')


_T = TypeVar('_T')


@dataclass
class Namespace:
    parents: List[Namespace] = field(default_factory=list)
    types: Dict[str, ts.Type] = field(default_factory=dict)
    values: Dict[str, Variable] = field(default_factory=dict)
    functions: Dict[str, ast.Function] = field(default_factory=dict)

    def add_type(self, t: ts.Type, name: Optional[str] = None) -> None:
        self._add_item(self.types, t, name or t.name)

    def add_value(self, t: Variable) -> None:
        self._add_item(self.values, t, t.name)

    def add_function(self, t: ast.Function) -> None:
        self._add_item(self.functions, t, t.name.text)

    def _add_item(self, sub: Dict[str, _T], item: _T, name: str) -> None:
        # This method mutates sub
        assert name not in sub, name
        sub[name] = item

    def get_type(self, name: str) -> ts.Type:
        return cast(ts.Type, self._get_item('types', name))

    def get_value(self, name: str) -> Variable:
        return cast(Variable, self._get_item('values', name))

    def get_function(self, name: str) -> ast.Function:
        return cast(ast.Function, self._get_item('functions', name))

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


@dataclass
class Variable:
    name: str
    type_: ts.Type
    value: ir.Value
