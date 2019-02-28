from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Callable, ContextManager, Dict, List, Set

from llvmlite import ir

from kotlang import codegen, typesystem as ts
from kotlang.ast import Namespace
from kotlang.cimport import merge_header_contents_into_module, read_header
from kotlang.parser import parse, ParseError


@dataclass
class Context:
    timer: Callable[[str], ContextManager[None]]

    def timed(self, label: str) -> ContextManager[None]:
        # mypy bug: https://github.com/python/mypy/issues/708
        return self.timer(label)  # type: ignore

    def load_module_text(self, module: str) -> ModuleFile:
        # TODO: make current directory independent and implement module search path
        path = Path('modules') / f'{module}.kot'
        text = self.load_file_text(path)
        return ModuleFile(str(path), text)

    def load_file_text(self, filename: Path) -> str:
        return filename.read_text()

    def compile(self, main_module: str) -> ir.Module:
        modules = {}
        modules_to_parse = [('by_path', main_module)]
        includes_to_parse: Set[str] = set()
        dependency_map = {}

        while modules_to_parse:
            id_type, source = modules_to_parse.pop()
            with self.timed('Reading from storage'):
                if id_type == 'by_path':
                    text = self.load_file_text(Path(source))
                    source = source.split('/')[-1].replace('.kot', '')
                else:
                    text = self.load_module_text(source).text

            with self.timed('Parsing'):
                try:
                    module = parse(text, source)
                except ParseError as e:
                    print(e.context, file=sys.stderr)
                    print(f'Cannot parse {source}.kot: {e.message}', file=sys.stderr)
                    sys.exit(1)
                modules[source] = module
                modules_needed_by_this_module = set(module.imports)
                dependency_map[source] = modules_needed_by_this_module
                modules_to_parse.extend(
                    ('by_name', m) for m in modules_needed_by_this_module - modules.keys()
                )
                includes_to_parse.update(module.includes)

        with self.timed('Reading C headers'):
            includes = {i: read_header(i) for i in includes_to_parse}
        codegen_order = topological_sort(dependency_map)
        namespaces: Dict[str, Namespace] = {}

        builtin_namespace = Namespace()
        builtin_namespace.add_type(ts.VoidType())
        builtin_namespace.add_type(ts.BoolType())
        for signed in {True, False}:
            for bits in {8, 16, 32, 64}:
                builtin_namespace.add_type(ts.IntType(bits, signed=signed))
        for bits in [32, 64, 80]:
            builtin_namespace.add_type(ts.FloatType(bits))

        ir_module = ir.Module(name=main_module)

        with self.timed('Codegen'):
            for name in codegen_order:
                module = modules[name]
                parent_namespaces = [namespaces[i] for i in module.imports] + [builtin_namespace]
                if module.includes:
                    headers_contents = [includes[i] for i in module.includes]
                    c_module = merge_header_contents_into_module(headers_contents)
                    c_namespace = codegen.codegen_module(
                        c_module, ir_module, [builtin_namespace], name + '_c'
                    )
                    parent_namespaces.append(c_namespace)
                namespace = codegen.codegen_module(module, ir_module, parent_namespaces, name)
                namespaces[name] = namespace

        return ir_module


def topological_sort(deps: Dict[str, Set[str]]) -> List[str]:
    """deps contains mapping between module names and lists of their imports.
    The return value is a list of module names so that no module depends on modules
    that come after it.
    """
    # TODO: implement cycle detection
    # Let's make a deep copy of the dictionary, so that we are a good citizen and don't
    # modify the parameter
    deps = {k: set(v) for k, v in deps.items()}
    flat: List[str] = []
    while deps:
        keys_with_no_deps = {k for k, v in deps.items() if not v}
        flat += keys_with_no_deps
        deps = {k: v - keys_with_no_deps for k, v in deps.items() if k not in keys_with_no_deps}
    return flat


@dataclass
class ModuleFile:
    filename: str
    text: str
