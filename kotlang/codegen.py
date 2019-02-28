from __future__ import annotations

from itertools import zip_longest
from typing import Collection, Dict, List, Union

from llvmlite import ir

from kotlang import ast, typesystem as ts


def codegen_module(
    node: ast.Module, module: ir.Module, parent_namespaces: List[ast.Namespace], module_name: str
) -> ast.Namespace:
    module_namespace = ast.Namespace(parents=parent_namespaces)

    definitions_types = [(td, td.get_dummy_type()) for td in node.types]
    for _, t in definitions_types:
        module_namespace.add_type(t)
    for td, t in definitions_types:
        td.fill_type_members(module_namespace, t)

    for f in node.functions:
        module_namespace.add_function(f)

    for variable_declaration in node.variables:
        variable_declaration.codegen_module_level(module, module_namespace, module_name)

    nongeneric_functions = (f for f in node.functions if not f.is_generic)
    for f in nongeneric_functions:
        get_or_create_llvm_function(module, module_namespace, f)

    return module_namespace


def codegen_statement(  # noqa: C901
    node: ast.Statement, module: ir.Module, builder: ir.IRBuilder, namespace: ast.Namespace
) -> None:
    if isinstance(node, ast.CompoundStatement):
        for s in node.statements:
            codegen_statement(s, module, builder, namespace)
    elif isinstance(node, ast.CodeBlock):
        block_namespace = ast.Namespace(parents=[namespace])
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
        loop_namespace = ast.Namespace(parents=[namespace])
        codegen_statement(node.entry, module, builder, loop_namespace)
        actual_body = ast.CompoundStatement([node.body, node.step])
        loop_helper(module, builder, loop_namespace, node.condition, actual_body)
    elif isinstance(node, ast.ReturnStatement):
        if node.expression is not None:
            builder.ret(codegen_expression(node.expression, module, builder, namespace, 'return_value'))
        else:
            builder.ret_void()
    elif isinstance(node, ast.VariableDeclaration):
        type_ = node.variable_type(namespace)
        ir_type = type_.get_ir_type()
        if isinstance(type_, ts.FunctionType):
            # TODO: now our typesystem things we're dealing with functions while actually we're
            # dealing with function pointers. See if this can be ironed out. If it can't then see
            # if the abstraction is right.
            ir_type = ir_type.as_pointer()
        memory = builder.alloca(ir_type, name=node.name)
        namespace.add_value(ast.Variable(node.name, type_, memory))
        if node.expression is not None:
            value = codegen_expression(node.expression, module, builder, namespace)
            adapted_value = type_.adapt(builder, value, node.expression.type(namespace))
            builder.store(adapted_value, memory)
    elif isinstance(node, ast.Expression):
        codegen_expression(node, module, builder, namespace)
    else:
        raise NotImplementedError(f'Code generation not implemented for {type(node)}')


def codegen_expression(  # noqa: C901
    node: ast.Expression, module: ir.Module, builder: ir.IRBuilder, namespace: ast.Namespace, name: str = ''
) -> ir.Value:
    if isinstance(node, ast.NegativeExpression):
        value = codegen_expression(node.expression, module, builder, namespace, name)
        value.constant = -value.constant
        return value
    elif isinstance(node, ast.BoolNegationExpression):
        assert node.expression.type(namespace).name == 'bool', node.expression

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
        function: Union[ast.Function, ast.Variable]
        parameter_names: List[str]
        try:
            function = namespace.get_function(node.name)
        except KeyError:
            function = namespace.get_value(node.name)
            assert isinstance(function, ast.Variable)
            assert isinstance(function.type_, ts.FunctionType)
            parameter_types = function.type_.parameter_types
            # TODO provide parameter names here somehow? We don't have them right now.
            parameter_names = []
            llvm_function = builder.load(function.value)
            ft = function.type_
        else:
            if function.is_generic:
                namespace = namespace_for_specialized_function(namespace, function, node.parameters)

            ft = function.get_type(namespace)
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

        provided_parameter_types = [p.type(namespace) for p in node.parameters]
        for i, (value, from_type, to_type) in enumerate(
            zip(parameter_values, provided_parameter_types, parameter_types)
        ):
            parameter_values[i] = to_type.adapt(builder, value, from_type)

        return builder.call(llvm_function, parameter_values, name=name)
    elif isinstance(node, ast.StructInstantiation):
        struct = namespace.get_type(node.name)
        assert isinstance(struct, ts.StructType)
        assert len(node.parameters) == len(struct.members)

        member_names = [m[0] for m in struct.members]
        memory = builder.alloca(struct.get_ir_type())
        value = builder.load(memory)
        for i, (p, n) in enumerate(zip(node.parameters, member_names)):
            member_value = codegen_expression(p, module, builder, namespace, f'{node.name}.{n}')
            value = builder.insert_value(value, member_value, i)

        return value
    elif isinstance(node, ast.StringLiteral):
        return string_constant(module, builder, node.text[1:-1], namespace)
    elif isinstance(node, ast.IntegerLiteral):
        value = int(node.text)
        return namespace.get_type('i64').get_ir_type()(value)
    elif isinstance(node, ast.FloatLiteral):
        value = float(node.text)
        return namespace.get_type('f64').get_ir_type()(value)
    elif isinstance(node, ast.BoolLiteral):
        return namespace.get_type('bool').get_ir_type()(node.value)
    elif isinstance(node, ast.VariableReference):
        type_ = node.type(namespace)
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
        destination_type = node.target.type(namespace)
        adapted_value = destination_type.adapt(builder, value, node.expression.type(namespace))
        builder.store(adapted_value, pointer)
        return value
    elif isinstance(node, ast.ArrayLiteral):
        type_ = node.type(namespace)
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
    namespace: ast.Namespace,
    condition: ast.Expression,
    body: ast.Statement,
) -> None:
    assert isinstance(condition.type(namespace), ts.BoolType)
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
    namespace: ast.Namespace, function: ast.Function, arguments: Collection[ast.Expression]
) -> ast.Namespace:
    mapping: Dict[str, ts.Type] = {}
    for parameter, expression in zip(function.parameters, arguments):
        assert isinstance(parameter.type_, ast.BaseTypeReference), 'TODO support pointers etc. here'
        type_name = parameter.type_.name
        if type_name in function.type_parameters:
            deduced_type = expression.type(namespace)
            assert type_name not in mapping or mapping[type_name] == deduced_type
            mapping[type_name] = deduced_type

    new_namespace = ast.Namespace(parents=[namespace])
    for name, type_ in mapping.items():
        new_namespace.add_type(type_, name)
    return new_namespace


def get_or_create_llvm_function(
    module: ir.Module, namespace: ast.Namespace, function: ast.Function
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

            function_namespace = ast.Namespace(parents=[namespace])
            parameter_types = zip(function.parameters, ft.parameter_types)
            for i, (pt, arg) in enumerate(zip(parameter_types, llvm_function.args)):
                (parameter, parameter_type) = pt
                memory = builder.alloca(arg.type, name=parameter.name)
                builder.store(arg, memory)
                function_namespace.add_value(
                    ast.Variable(parameter.name or f'param{i + 1}', parameter_type, memory)
                )

            codegen_statement(function.code_block, module, builder, function_namespace)
            if ft.return_type == ts.void:
                builder.ret_void()
            else:
                # FIXME: We depend on having returned already but this is not ensured
                if not builder.block.is_terminated:
                    builder.unreachable()

    return llvm_function


constant_counter = 0


def string_constant(
    module: ir.Module, builder: ir.IRBuilder, s: str, namespace: ast.Namespace
) -> ir.Constant:
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
    node: ast.Expression, module: ir.Module, builder: ir.IRBuilder, namespace: ast.Namespace
) -> ir.Value:
    if isinstance(node, ast.VariableReference):
        value: Union[ast.Function, ast.Variable]
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
        left_type = node.left_side.type(namespace)
        assert isinstance(left_type, ts.DottableType), left_type
        left_pointer = get_pointer(node.left_side, module, builder, namespace)
        return left_type.get_member_pointer(builder, left_pointer, node.member)
    elif isinstance(node, ast.IndexAccess):
        pointer_type = node.pointer.type(namespace)
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
