from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
from typing import Dict, Iterable, List, Union

from kotlang import ast

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


def read_header(header: str) -> HeaderContents:
    tu = clang_index.parse(find_header(header), options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)
    types: List[ast.TypeDefinition] = []
    functions: List[ast.Function] = []
    variables: List[ast.VariableDeclaration] = []
    cursors = [tu.cursor]
    defines: Dict[str, cindex.Token] = {}
    while cursors:
        c = cursors.pop()
        if c.kind == cindex.CursorKind.MACRO_DEFINITION:  # type: ignore
            macro_tokens = list(c.get_tokens())[1:]
            # We skip macros here, we're only interested in literal defines
            if len(macro_tokens) == 1:
                defines[c.spelling] = macro_tokens[0]
        elif (
            (
                c.kind is cindex.CursorKind.STRUCT_DECL  # type: ignore
                and c.is_definition()
            )
            or (
                c.kind is cindex.CursorKind.TYPE_REF  # type: ignore
                and c.type.kind is cindex.TypeKind.RECORD  # type: ignore
            )
        ):
            types.append(convert_c_record_definition(c))
        elif (
            c.kind is cindex.CursorKind.FUNCTION_DECL  # type: ignore
            and not c.is_definition()
            # HACK: We don't support block pointers, if we see the caret symbol we can suspect
            # something fishy is going on, let's skip that particular function
            and '^' not in c.type.spelling
        ):
            functions.append(convert_c_function_declaration(c))
        cursors.extend(c.get_children())

    for name, token in defines.items():
        # TODO: import things other than ints here - recursively expand macros and import strings
        if token.kind is cindex.TokenKind.LITERAL and token.spelling.isdigit():  # type: ignore
            variables.append(ast.VariableDeclaration(name, ast.IntegerLiteral(token.spelling)))

    return HeaderContents(types, functions, variables)


def find_header(header: str) -> str:
    paths = [Path(ip) / header for ip in include_paths]
    for p in paths:
        if p.exists():
            return str(p)
    assert False, f'Header {header} not found'


@dataclass
class HeaderContents:
    types: List[ast.TypeDefinition]
    functions: List[ast.Function]
    variables: List[ast.VariableDeclaration]


def merge_header_contents_into_module(headers_contents: Iterable[HeaderContents]) -> ast.Module:
    functions = {}
    types = {}
    variables = {}

    for hc in headers_contents:
        types.update({s.name: s for s in hc.types})
        functions.update({f.name: f for f in hc.functions})
        variables.update({v.name: v for v in hc.variables})

    # HACK: libclang resolves va_list to __va_list_tag structure but the definition of the structure
    # is defined internally by Clang and not returned as part of the AST. In order to fully process
    # types and functions referring to __va_list_tag we need to provide a definition.
    builtin_va_list = ast.get_builtin_va_list_struct()
    types[builtin_va_list.name] = builtin_va_list
    return ast.Module(list(types.values()), list(functions.values()), [], [], list(variables.values()))


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


def convert_c_record_definition(declaration: cindex.Cursor) -> Union[ast.Struct, ast.Union]:
    name = declaration.spelling
    this_type = 'struct'
    if declaration.kind is cindex.CursorKind.TYPE_REF:  # type: ignore
        if 'struct ' in name:
            name = name.replace('struct ', '')
        else:
            assert 'union ' in name
            name = name.replace('union ', '')
            this_type = 'union'
    else:
        assert declaration.kind is cindex.CursorKind.STRUCT_DECL  # type: ignore
    members = [(c.spelling, convert_c_type_reference(c.type)) for c in declaration.type.get_fields()]
    if this_type == 'struct':
        return ast.Struct(name, members)
    else:
        return ast.Union(name, members)


def convert_c_type_reference(ref: cindex.Type) -> ast.TypeReference:
    pointer_level = 0
    ref = ref.get_canonical()
    while True:
        if ref.kind is cindex.TypeKind.POINTER:  # type: ignore
            pointer_level += 1
            ref = ref.get_pointee().get_canonical()
        elif ref.kind in [cindex.TypeKind.CONSTANTARRAY, cindex.TypeKind.INCOMPLETEARRAY]:  # type: ignore
            pointer_level += 1
            ref = ref.get_array_element_type().get_canonical()
        else:
            break
    t: ast.TypeReference
    if ref.kind is cindex.TypeKind.FUNCTIONPROTO:  # type: ignore
        t = ast.FunctionTypeReference(
            [convert_c_type_reference(t) for t in ref.argument_types()],
            convert_c_type_reference(ref.get_result()),
            ref.is_function_variadic(),
        )
    elif ref.kind is cindex.TypeKind.RECORD:  # type: ignore
        t = ast.BaseTypeReference(ref.get_declaration().spelling)
    elif ref.kind is cindex.TypeKind.ENUM:  # type: ignore
        # TODO Make enums strongly typed
        enum_type = ref.get_declaration().enum_type
        t = convert_c_type_reference(enum_type)
    else:
        t = ast.BaseTypeReference(c_types_mapping[ref.kind])
        # HACK: LLVM IR doesn't support void pointers
        if t.name == 'void' and pointer_level > 0:
            t = ast.BaseTypeReference('i8')
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
