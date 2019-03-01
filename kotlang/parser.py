from collections import OrderedDict
import os
from pathlib import Path
import subprocess
from typing import cast, Dict, Iterator, List, MutableMapping, Optional, Tuple, Union

from kotlang import ast
from kotlang.itertools import Peekable
from kotlang.lexer import lex, Token, TokenType


def parse(text: str, filename: str) -> ast.Module:
    tokens = Peekable(lex(text, filename))
    try:
        module = read_module(tokens)
    except UnexpectedToken as e:
        error_context = context_with_pointer(text, e.token.span.line, e.token.span.column)
        raise ParseError(error_context, str(e))
    return module


def context_with_pointer(text: str, line: int, column: int) -> str:
    lines = text.split('\n')
    return '\n'.join(
        [
            '// ...',
            '\n'.join(lines[max(line - 5, 0) : line + 1]),
            '-' * column + '^',
            '\n'.join(lines[line + 1 : line + 5 + 1]),
            '// ...',
        ]
    )


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
            isinstance(w, TokenType)
            and next_token.type == w
            or isinstance(w, str)
            and next_token.text == w
            or next_token == w
        ):
            return next_token
    raise UnexpectedToken(next_token, list(what))


def read_module(tokens: Peekable[Token]) -> ast.Module:
    # Module = (FunctionDefinition | FunctionDeclaration | StructDefinition)*;
    functions = []
    types: List[ast.StructUnion] = []
    imports = []
    includes = []
    while tokens.peek().type is not TokenType.eof:
        next_text = expect_no_eat(tokens, 'def', 'extern', 'struct', 'import', 'cimport', 'union').text
        if next_text == 'extern':
            functions.append(read_function_declaration(tokens))
        elif next_text == 'def':
            functions.append(read_function_definition(tokens))
        elif next_text == 'import':
            imports.append(read_import(tokens))
        elif next_text == 'cimport':
            includes.append(read_cimport(tokens))
        elif next_text == 'union':
            types.append(read_union_definition(tokens))
        else:
            types.append(read_struct_definition(tokens))

    return ast.Module(types, functions, imports, includes, [])


def read_struct_definition(tokens: Peekable[Token]) -> ast.StructUnion:
    # StructDefinition = "struct" name "{" StructMembers "}";
    # StructMembers = StructMember ";" [StructMembers] | empty;
    # StructMember = name ":" name;
    expect(tokens, 'struct')
    name = expect(tokens, TokenType.identifier).text
    members = read_struct_body(tokens)
    return ast.StructUnion(name, members, False)


def read_union_definition(tokens: Peekable[Token]) -> ast.StructUnion:
    # UnionDefinition = "struct" name "{" UnionMembers "}";
    # UnionMembers = UnionMember ";" [UnionMembers] | empty;
    # UnionMember = name ":" name;
    expect(tokens, 'union')
    name = expect(tokens, TokenType.identifier).text
    members = read_struct_body(tokens)
    return ast.StructUnion(name, members, True)


def read_struct_body(tokens: Peekable[Token]) -> List[Tuple[str, ast.TypeReference]]:
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
    return members


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


def read_import(tokens: Peekable[Token]) -> str:
    expect(tokens, 'import')
    module_name = expect(tokens, TokenType.identifier).text
    expect(tokens, ';')
    return module_name


def read_cimport(tokens: Peekable[Token]) -> str:
    # CImport = "cimport" StringLiteral ";";
    expect(tokens, 'cimport')
    header = expect(tokens, TokenType.string_literal).text
    expect(tokens, ';')
    # Stripping first and last quotation mark characters because that's what string literals have
    return header[1:-1]


def read_function_header(
    tokens: Peekable[Token]
) -> Tuple[str, List[str], ast.ParameterList, ast.TypeReference]:
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
        to_return = ast.BoolLiteral(next_token.text == 'true')
    elif next_token.text == '&':
        subexpression = read_primary_expression(tokens)
        # TODO: make this nicer
        assert isinstance(
            subexpression, ast.VariableReference
        ), 'Can only get address of simple variable at the moment'
        to_return = ast.AddressOf(subexpression)
    elif next_token.text == '*':
        subexpression = read_primary_expression(tokens)
        # TODO: make this nicer
        assert isinstance(
            subexpression, ast.VariableReference
        ), 'Can only get value of simple variable pointer at the moment'
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
