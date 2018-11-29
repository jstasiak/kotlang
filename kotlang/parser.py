from collections import OrderedDict
import os
from pathlib import Path
import platform
import subprocess
from typing import cast, Iterator, List, MutableMapping, Optional, Tuple, Union

from kotlang import ast
from kotlang.itertools import Peekable
from kotlang.lexer import lex, Token, TokenType

# Paths are hardcoded, can't be bothered to detect them at the moment
if platform.system() == 'Linux':
    # Ubuntu
    libclang_file = '/usr/lib/llvm-7/lib/libclang.so'
    include_paths = [
        '/usr/local/include',
        '/usr/local/clang-7.0.0/lib/clang/7.0.0/include',
        '/usr/include/x86_64-linux-gnu',
        '/usr/include',
    ]
else:
    # Mac OS 10.13 with HomeBrew
    libclang_file = '/usr/local/opt/llvm/lib/libclang.dylib'
    include_paths = [
        '/usr/local/include',
        '/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/10.0.0/include',  # noqa
        '/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include',
        '/usr/include',
        '/System/Library/Frameworks',
        '/Library/Frameworks',
    ]

from kotlang.clang import cindex  # noqa
cindex.Config.set_library_file(libclang_file)
clang_index = cindex.Index.create()


def find_header(header: str) -> str:
    paths = [Path(ip) / header for ip in include_paths]
    for p in paths:
        if p.exists():
            return str(p)
    assert False, f'Header {header} not found'


def parse(text: str, name: str) -> ast.Module:
    tokens = Peekable(lex(text))
    try:
        module = read_module(tokens)
    except UnexpectedToken as e:
        context = context_with_pointer(text, e.token.line, e.token.column)
        raise ParseError(context, str(e))
    return module


def context_with_pointer(text: str, line: int, column: int) -> str:
    lines = text.split('\n')
    return '\n'.join([
        '// ...',
        '\n'.join(lines[max(line - 5, 0):line + 1]),
        '-' * column + '^',
        '\n'.join(lines[line + 1:line + 5 + 1]),
        '// ...',
    ])


class ParseError(Exception):
    def __init__(self, context: str, message: str) -> None:
        self.context = context
        self.message = message
        super().__init__(context, message)


ExpectedTokenT = Union[TokenType, Token, str]


class UnexpectedToken(Exception):
    def __init__(self, token: Token, expected: List[ExpectedTokenT]) -> None:
        self.token = token
        message = f'Unexpected token: {repr(token.text)}'
        if expected:
            message += f', expected one of: {list(expected)}'
        super().__init__(message)


def expect(tokens: Peekable[Token], *what: ExpectedTokenT) -> Token:
    token = expect_no_eat(tokens, *what)
    next(tokens)
    return token


def expect_no_eat(tokens: Peekable[Token], *what: ExpectedTokenT) -> Token:
    next_token = tokens.peek()

    for w in what:
        if (
            isinstance(w, TokenType) and next_token.type == w
            or isinstance(w, str) and next_token.text == w
            or next_token == w
        ):
            return next_token
    raise UnexpectedToken(next_token, list(what))


def read_module(tokens: Peekable[Token]) -> ast.Module:
    # Module = (FunctionDefinition | FunctionDeclaration | StructDefinition)*;
    functions = []
    structs = []
    imports = []
    cfunctions = {}
    while tokens.peek().type is not TokenType.eof:
        next_text = expect_no_eat(tokens, 'def', 'extern', 'struct', 'import', 'cimport').text
        if next_text == 'extern':
            functions.append(read_function_declaration(tokens))
        elif next_text == 'def':
            functions.append(read_function_definition(tokens))
        elif next_text == 'import':
            imports.append(read_import(tokens))
        elif next_text == 'cimport':
            cfunctions.update({f.name: f for f in read_cimport(tokens)})
        else:
            structs.append(read_struct_definition(tokens))

    if cfunctions:
        imports.append(('c', ast.Module([], list(cfunctions.values()), [])))

    return ast.Module(structs, functions, imports)


def read_struct_definition(tokens: Peekable[Token]) -> ast.Struct:
    # StructDefinition = "struct" name "{" StructMembers "}";
    # StructMembers = StructMember ";" [StructMembers] | empty;
    # StructMember = name ":" name;
    expect(tokens, 'struct')
    name = expect(tokens, TokenType.identifier).text
    expect(tokens, '{')

    members: List[Tuple[str, ast.TypeReference]] = []

    while tokens.peek().text != '}':
        member_name = expect(tokens, TokenType.identifier)
        expect(tokens, ':')
        member_type = read_type_reference(tokens)
        expect(tokens, ';')
        assert member_name.text not in members
        members.append((member_name.text, member_type))
    expect(tokens, '}')
    return ast.Struct(name, members)


def read_function_declaration(tokens: Peekable[Token]) -> ast.Function:
    expect(tokens, 'extern')
    name, type_parameters, parameters, return_type = read_function_header(tokens)
    assert not type_parameters
    expect(tokens, ';')
    return ast.Function(name, parameters, return_type, [], None)


def read_function_definition(tokens: Peekable[Token]) -> ast.Function:
    name, type_parameters, parameters, return_type = read_function_header(tokens)
    code_block = read_code_block(tokens)
    return ast.Function(name, parameters, return_type, type_parameters, code_block)


def read_import(tokens: Peekable[Token]) -> Tuple[str, ast.Module]:
    expect(tokens, 'import')
    module_name = expect(tokens, TokenType.identifier).text
    expect(tokens, ';')
    text = read_module_text(module_name)
    return (module_name, parse(text, module_name))


def read_cimport(tokens: Peekable[Token]) -> List[ast.Function]:
    # CImport = "cimport" StringLiteral ";";
    expect(tokens, 'cimport')
    header = expect(tokens, TokenType.string_literal).text
    expect(tokens, ';')

    # Stripping first and last quotation mark characters because that's what string literals have
    header = header[1:-1]

    tu = clang_index.parse(find_header(header))
    functions: List[ast.Function] = []
    cursors = [tu.cursor]
    while cursors:
        c = cursors.pop()
        if (
            c.kind is cindex.CursorKind.FUNCTION_DECL  # type: ignore
            and not c.is_definition()
            # TODO: This restriction is in place until we follow typedefs and import struct declarations.
            # For now let's import few whitelisted functions.
            and c.spelling in {'printf', 'abort', 'strlen'}
        ):
            functions.append(convert_c_function_declaration(c))
        cursors.extend(c.get_children())

    return functions


def convert_c_function_declaration(declaration: cindex.Cursor) -> ast.Function:
    return_type = convert_c_type_reference(declaration.type.get_result())

    parameter_names_types = [
        (p.spelling or None, convert_c_type_reference(p.type))
        for p in declaration.get_arguments()
    ]
    parameters = ast.ParameterList(
        [ast.Parameter(n, t) for (n, t) in parameter_names_types],
        declaration.type.is_function_variadic(),
    )
    return ast.Function(declaration.spelling, parameters, return_type, [], None)


def convert_c_type_reference(ref: cindex.Type) -> ast.TypeReference:
    pointer_level = 0
    while ref.kind is cindex.TypeKind.POINTER:  # type: ignore
        pointer_level += 1
        ref = ref.get_pointee()
    t: ast.TypeReference = ast.BaseTypeReference(c_types_mapping[ref.kind])
    for i in range(0, pointer_level):
        t = t.as_pointer()
    return t


# The compiler targets 64-bit Mac and Linux environments, mappings based on
# https://en.wikipedia.org/wiki/64-bit_computing#64-bit_data_models
c_types_mapping = {
    cindex.TypeKind.INT: 'i32',  # type: ignore
    cindex.TypeKind.UINT: 'u32',  # type: ignore
    cindex.TypeKind.CHAR_S: 'i8',  # type: ignore
    cindex.TypeKind.UCHAR: 'u8',  # type: ignore
    cindex.TypeKind.SHORT: 'i16',  # type: ignore
    cindex.TypeKind.USHORT: 'u16',  # type: ignore
    cindex.TypeKind.LONG: 'i64',  # type: ignore
    cindex.TypeKind.ULONG: 'u64',  # type: ignore
    cindex.TypeKind.LONGLONG: 'i64',  # type: ignore
    cindex.TypeKind.ULONGLONG: 'i64',  # type: ignore
    cindex.TypeKind.VOID: 'void',  # type: ignore
    cindex.TypeKind.FLOAT: 'f32',  # type: ignore
    cindex.TypeKind.DOUBLE: 'f64',  # type: ignore
    cindex.TypeKind.LONGDOUBLE: 'f80',  # type: ignore
}


def read_module_text(name: str) -> str:
    with open(os.path.join('modules', f'{name}.kot')) as f:
        return f.read()


def read_function_header(tokens: Peekable[Token]) -> Tuple[str, List[str], ast.ParameterList, ast.TypeReference]:
    expect(tokens, 'def')
    name = expect(tokens, TokenType.identifier)
    type_parameters = []
    if tokens.peek().text == '<':
        next(tokens)
        while tokens.peek().text != '>':
            type_ = expect(tokens, TokenType.identifier).text
            type_parameters.append(type_)
            if tokens.peek().text != ',':
                assert tokens.peek().text == '>'
        expect(tokens, '>')

    expect(tokens, '(')
    parameters = read_function_parameters(tokens)
    expect(tokens, ')')
    if tokens.peek().text == '->':
        expect(tokens, '->')
        return_type = read_type_reference(tokens)
    else:
        return_type = ast.BaseTypeReference('void')
    return name.text, type_parameters, parameters, return_type


def read_function_parameters(tokens: Peekable[Token]) -> ast.ParameterList:
    parameters = []
    variadic = False
    while tokens.peek().text != ')':
        if tokens.peek().text == '...':
            next(tokens)
            variadic = True
            break

        name = read_variable_name(tokens)
        expect(tokens, ':')
        type_ = read_type_reference(tokens)
        parameters.append(ast.Parameter(name, type_))
        if tokens.peek().text == ',':
            next(tokens)
    return ast.ParameterList(parameters, variadic=variadic)


def read_variable_name(tokens: Peekable[Token]) -> str:
    return expect(tokens, TokenType.identifier).text


def read_type_reference(tokens: Peekable[Token]) -> ast.TypeReference:
    base_name = expect(tokens, TokenType.identifier)
    reference: ast.TypeReference = ast.BaseTypeReference(base_name.text)  # noqa: E701
    while tokens.peek().text in ['*', '[']:
        token = next(tokens)
        if token.text == '*':
            reference = reference.as_pointer()
        else:
            length = read_primary_expression(tokens)
            expect(tokens, ']')
            reference = ast.ArrayTypeReference(reference, length)
    return reference


def read_code_block(tokens: Peekable[Token]) -> ast.CodeBlock:
    # CodeBlock = "{" Statement* "}"
    expect(tokens, '{')
    statements = []
    while tokens.peek().text != '}':
        statements.append(read_statement(tokens))
    expect(tokens, '}')
    return ast.CodeBlock(statements)


def read_statement(tokens: Peekable[Token], *, eat_semicolon: bool = True) -> ast.Statement:
    # Statement = CodeBlock | IfStatement | PatternMatch
    #           | (ReturnStatement | Expression | VariableDeclaration) ";";
    next_token = tokens.peek()
    needs_semicolon = True
    statement: ast.Statement

    if next_token.text == 'return':
        next(tokens)
        expression = None
        if tokens.peek().text != ';':
            expression = read_expression(tokens)
        statement = ast.ReturnStatement(expression)
    elif next_token.text == 'let':
        statement = read_variable_declaration(tokens)
    elif next_token.text == 'if':
        statement = read_if_statement(tokens)
        needs_semicolon = False
    elif next_token.text == 'while':
        statement = read_while_loop(tokens)
        needs_semicolon = False
    elif next_token.text == 'for':
        statement = read_for_loop(tokens)
        needs_semicolon = False
    elif next_token.text == '{':
        statement = read_code_block(tokens)
        needs_semicolon = False
    elif next_token.text == 'match':
        statement = read_pattern_match(tokens)
        needs_semicolon = False
    else:
        statement = read_assignment_expression(tokens)
    if needs_semicolon and eat_semicolon:
        expect(tokens, ';')
    return statement


def read_assignment_expression(tokens: Peekable[Token]) -> ast.Expression:
    # TODO: implement robust lvalue concept
    operands = [read_expression(tokens)]

    while tokens.peek().text == '=':
        next(tokens)
        operands.append(read_expression(tokens))

    result: ast.Expression = operands[-1]
    for left in reversed(operands[0:-1]):
        result = ast.Assignment(left, result)
    return result


def read_if_statement(tokens: Peekable[Token]) -> ast.Statement:
    # IfStatement = "if" "(" Expression ")" CodeBlock ["else" CodeBlock]
    expect(tokens, 'if')
    expect(tokens, '(')
    expression = read_expression(tokens)
    expect(tokens, ')')
    first_statement = read_code_block(tokens)
    second_statement = None
    if tokens.peek().text == 'else':
        next(tokens)
        second_statement = read_code_block(tokens)
    return ast.IfStatement(expression, first_statement, second_statement)


def read_while_loop(tokens: Peekable[Token]) -> ast.Statement:
    # WhileLoop = "while" "(" Expression ")" CodeBlock
    expect(tokens, 'while')
    expect(tokens, '(')
    condition = read_expression(tokens)
    expect(tokens, ')')
    body = read_code_block(tokens)
    return ast.WhileLoop(condition, body)


def read_for_loop(tokens: Peekable[Token]) -> ast.Statement:
    # WhileLoop = "for" "(" Statement ";" Expression ";" Statement ")" CodeBlock
    expect(tokens, 'for')
    expect(tokens, '(')
    entry = read_statement(tokens, eat_semicolon=False)
    expect(tokens, ';')
    condition = read_expression(tokens)
    expect(tokens, ';')
    step = read_statement(tokens, eat_semicolon=False)
    expect(tokens, ')')
    body = read_code_block(tokens)
    return ast.ForLoop(entry, condition, step, body)


def read_pattern_match(tokens: Peekable[Token]) -> ast.Statement:
    # PatternMatch = "match" "(" Expression ")" "{" Patterns "}";
    # PatternMatchArms = PatternMatchArm ["," PatternMatchArm]+ [","];
    # PatternMatchArm = Pattern "->" Statement
    # Pattern = Expression;

    expect(tokens, 'match')
    expect(tokens, '(')
    match_value = read_expression(tokens)
    expect(tokens, ')')
    expect(tokens, '{')

    arms = []

    while tokens.peek().text != '}':
        arm_pattern = read_expression(tokens)
        expect(tokens, '->')
        arm_body = read_expression(tokens)
        arms.append(ast.PatternMatchArm(arm_pattern, arm_body))
        if tokens.peek().text != ',':
            break
        expect(tokens, ',')

    expect(tokens, '}')
    return ast.PatternMatch(match_value, arms)


def read_comparison_expression(tokens: Peekable[Token]) -> ast.Expression:
    # ComparisonExpression = AddExpression [("<" | ">" | "<=" | ">=" | "==") AddExpression]
    expression = read_add_expression(tokens)
    if tokens.peek().text in {'<', '>', '<=', '>=', '==', '!='}:
        operator = next(tokens).text
        right_expression = read_add_expression(tokens)
        expression = ast.BinaryExpression(expression, operator, right_expression)
    return expression


def read_add_expression(tokens: Peekable[Token]) -> ast.Expression:
    # AddExpression = MulExpression (("+" | "-") MulExpression)*;
    expression = read_mul_expression(tokens)
    while tokens.peek().text in {'+', '-'}:
        operator = next(tokens).text
        right_expression = read_mul_expression(tokens)
        expression = ast.BinaryExpression(expression, operator, right_expression)
    return expression


# Expression = ComparisonExpression;
read_expression = read_comparison_expression


def read_mul_expression(tokens: Peekable[Token]) -> ast.Expression:
    # MulExpression = PrimaryExpression (("*" | "/") PrimaryExpression)*;
    expression = read_primary_expression(tokens)
    while tokens.peek().text in {'*', '/'}:
        operator = next(tokens).text
        right_expression = read_primary_expression(tokens)
        expression = ast.BinaryExpression(expression, operator, right_expression)
    return expression


def read_primary_expression(tokens: Peekable[Token]) -> ast.Expression:  # noqa: C901
    next_token = next(tokens)
    negative = False
    bool_negation = False
    if next_token.text == '-':
        negative = True
        next_token = next(tokens)
    elif next_token.text == 'not':
        bool_negation = True
        next_token = next(tokens)

    to_return: ast.Expression

    if next_token.type is TokenType.identifier:
        name = next_token.text
        next_token = tokens.peek()
        if next_token.text in {'(', '{'}:
            if next_token.text == '(':
                delimiters = ('(', ')')
                struct = False
            else:
                delimiters = ('{', '}')
                struct = True
            parameters = read_parameters(tokens, delimiters)
            if struct:
                to_return = ast.StructInstantiation(name, parameters)
            else:
                to_return = ast.FunctionCall(name, parameters)
        elif next_token.text in {'.', '['}:
            reference: ast.MemoryReference = ast.VariableReference(name)
            while next_token.text in {'.', '['}:
                character = next(tokens).text
                if character == '.':
                    right_name = expect(tokens, TokenType.identifier).text
                    reference = ast.DotAccess(reference, right_name)
                else:
                    index = read_expression(tokens)
                    expect(tokens, ']')
                    reference = ast.IndexAccess(reference, index)
                next_token = tokens.peek()
            to_return = reference
        else:
            to_return = ast.VariableReference(name)
    elif next_token.text == '[':
        values = read_array_initializers(tokens)
        expect(tokens, ']')
        to_return = ast.ArrayLiteral(values)
    elif next_token.type is TokenType.string_literal:
        to_return = ast.StringLiteral(next_token.text)
    elif next_token.type is TokenType.integer_literal:
        to_return = ast.IntegerLiteral(next_token.text)
    elif next_token.type is TokenType.float_literal:
        to_return = ast.FloatLiteral(next_token.text)
    elif next_token.type is TokenType.bool_literal:
        to_return = ast.BoolLiteral(next_token.text)
    elif next_token.text == '&':
        subexpression = read_primary_expression(tokens)
        # TODO: make this nicer
        assert isinstance(subexpression, ast.VariableReference), \
            'Can only get address of simple variable at the moment'
        to_return = ast.AddressOf(subexpression)
    elif next_token.text == '*':
        subexpression = read_primary_expression(tokens)
        # TODO: make this nicer
        assert isinstance(subexpression, ast.VariableReference), \
            'Can only get value of simple variable pointer at the moment'
        to_return = ast.ValueAt(subexpression)

    else:
        raise UnexpectedToken(next_token, [])

    if negative:
        to_return = ast.NegativeExpression(to_return)
    elif bool_negation:
        to_return = ast.BoolNegationExpression(to_return)
    return to_return


def read_array_initializers(tokens: Peekable[Token]) -> List[ast.Expression]:
    initializers: List[ast.Expression] = []

    while tokens.peek().text != ']':
        initializers.append(read_add_expression(tokens))
        if tokens.peek().text != ']':
            expect(tokens, ',')

    return initializers


def read_variable_declaration(tokens: Peekable[Token]) -> ast.VariableDeclaration:
    # VariableDeclaration = "let" name (":" name | "=" AddExpression | ":" name "=" AddExpression);
    expect(tokens, 'let')
    name = expect(tokens, TokenType.identifier).text

    type_ = None
    if tokens.peek().text == ':':
        next(tokens)
        type_ = read_type_reference(tokens)

    needs_initializer = type_ is None
    initializer = None
    if tokens.peek().text == '=' or needs_initializer:
        expect(tokens, '=')
        initializer = read_expression(tokens)

    return ast.VariableDeclaration(name, initializer, type_)


def read_parameters(tokens: Peekable[Token], delimiters: Tuple[str, str]) -> List[ast.Expression]:
    opening, closing = delimiters
    expect(tokens, opening)
    parameters = []
    while tokens.peek().type is not TokenType.eof and tokens.peek().text != closing:
        parameters.append(read_expression(tokens))
        if tokens.peek().text == ',':
            next(tokens)
    expect(tokens, closing)
    return parameters
