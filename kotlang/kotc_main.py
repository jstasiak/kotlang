#!/usr/bin/env python3
import contextlib
import os
import subprocess
import sys
import time
from typing import Any, IO, Iterator

import click
from llvmlite import binding as llvm, ir

from kotlang.lexer import lex
from kotlang.parser import parse, ParseError


class Emitter:
    def __init__(self, optimization_level: int = None) -> None:
        # All these initializations are required for code generation!
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        llvm.initialize_native_asmparser()

        # Create a target machine representing the host
        target = llvm.Target.from_default_triple()
        target_machine = target.create_target_machine(codemodel='default')
        target_machine.set_asm_verbosity(True)
        self._target_machine = target_machine

        pmb = llvm.PassManagerBuilder()
        pmb.opt_level = optimization_level
        self.mpm = llvm.create_module_pass_manager()
        pmb.populate(self.mpm)

    def _module_to_llvm_module(self, module: ir.Module) -> llvm.module.ModuleRef:
        try:
            llvm_module = llvm.parse_assembly(str(module))
        except Exception:
            print('Assembly being parsed:', file=sys.stderr)
            print(module, file=sys.stderr)
            raise
        self.mpm.run(llvm_module)
        return llvm_module

    def module_to_ir(self, module: ir.Module) -> bytes:
        return str(self._module_to_llvm_module(module)).encode()

    def module_to_machine_code(self, module: ir.Module) -> bytes:
        return self._target_machine.emit_object(self._module_to_llvm_module(module))

    def module_to_assembly(self, module: ir.Module) -> bytes:
        return self._target_machine.emit_assembly(self._module_to_llvm_module(module)).encode()


@click.command()
@click.argument('source', nargs=1, type=click.Path(exists=True))
@click.option('-c', '--compile-only', is_flag=True)
@click.option('-o', '--output')
@click.option('-v', '--verbose', count=True)
@click.option('-f', '--output-format', default='obj', type=click.Choice(['asm', 'ir', 'obj']))
@click.option('-O', '--optimization-level', default=0, type=int)
def main(
    source: str,
    compile_only: bool,
    output: str,
    verbose: int,
    output_format: str,
    optimization_level: int,
) -> None:
    assert optimization_level in range(0, 3 + 1)
    timer = timing if verbose >= 2 else dummy_timing
    with timer('Reading from storage'):
        with open(source) as f:
            text = f.read()

    with timer('Parsing'):
        try:
            module = parse(text, '__main__')
        except ParseError as e:
            print(e.context, file=sys.stderr)
            print(f'Cannot parse {os.path.basename(source)}: {e.message}', file=sys.stderr)
            sys.exit(1)
    with timer('Codegen'):
        llvm_module, foreign_libraries = module.codegen_top_level(source)

    base_name = os.path.splitext(source)[0]

    with timer('Initializing LLVM'):
        emitter = Emitter(optimization_level)

    with timer('Generating output'):
        suffix = ''
        if output_format == 'ir':
            content = emitter.module_to_ir(llvm_module)
            suffix = '.ll'
        elif output_format == 'asm':
            content = emitter.module_to_assembly(llvm_module)
            suffix = '.s'
        else:
            content = emitter.module_to_machine_code(llvm_module)
            suffix = '.o'

    if output_format == 'obj' and not compile_only:
        direct_output = base_name + suffix
        final_output = output or base_name
    else:
        direct_output = output or (base_name + suffix)

    with timer('Writing to storage'):
        with stdout_aware_binary_open(direct_output, 'w') as f2:
            f2.write(content)

    if output_format == 'obj' and not compile_only:
        with timer('Linking'):
            command_line = [
                'cc', direct_output,
                *(f'-l{library}' for library in foreign_libraries),
                '-o', final_output,
                '-v',
            ]
            subprocess.check_call(command_line)


@contextlib.contextmanager
def stdout_aware_binary_open(filename: str, mode: str) -> Iterator[IO[bytes]]:
    assert 't' not in mode
    assert 'b' not in mode
    mode += 'b'

    if filename == '-':
        fd = os.fdopen(sys.stdout.fileno(), mode)
    else:
        fd = open(filename, mode)

    try:
        yield fd
    finally:
        if filename != '-':
            fd.close()


@contextlib.contextmanager
def timing(description: str) -> Iterator[None]:
    t0 = time.time()
    yield
    t1 = time.time()
    dt = (t1 - t0) * 1000
    print(f'[timer] {description} took {dt:4f} ms', file=sys.stderr)


@contextlib.contextmanager
def dummy_timing(description: str) -> Iterator[None]:
    yield


if __name__ == '__main__':
    main()
