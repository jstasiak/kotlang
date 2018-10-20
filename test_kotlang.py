from dataclasses import dataclass
import os
from subprocess import CalledProcessError, check_output
import sys
import traceback
from typing import Iterator, Optional

from click.testing import CliRunner
import pytest

from kotlang.kotc_main import main

TEST_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_data'))


@dataclass
class _TestItem:
    source: str
    stdout: bytes
    stderr: bytes
    exit_code: Optional[int]


def load_test_data() -> Iterator[_TestItem]:
    # sorted so that the test ordering is deterministic
    files = sorted(os.listdir(TEST_DATA_DIR))
    source_files = [f for f in files if f.endswith('.kot')]

    for f in source_files:
        yield load_single_test_item(os.path.join(TEST_DATA_DIR, f))


def load_single_test_item(filename: str) -> _TestItem:
    base_name = os.path.splitext(filename)[0]

    stdout_file = base_name + '.stdout'
    stderr_file = base_name + '.stderr'
    exit_code_file = base_name + '.exit_code'
    stdout = read_file(stdout_file) if os.path.exists(stdout_file) else b''
    stderr = read_file(stderr_file) if os.path.exists(stderr_file) else b''
    exit_code = int(read_file(exit_code_file).decode()) if os.path.exists(exit_code_file) else None
    return _TestItem(source=filename, stdout=stdout, stderr=stderr, exit_code=exit_code)


def read_file(filename: str) -> str:
    with open(filename, 'rb') as f:
        return f.read()


@pytest.mark.parametrize('item', load_test_data(), ids=lambda item: os.path.basename(item.source))
def test_compilation(item: _TestItem) -> None:
    # TODO stop hardcoding the binary path
    binary_path = '/tmp/kot-test'
    arguments = [item.source, '-o', binary_path]
    runner = CliRunner()
    compile_result = runner.invoke(main, arguments, catch_exceptions=True)
    if compile_result.exc_info:
        print('Traceback:')
        traceback.print_exception(*compile_result.exc_info, file=sys.stdout)
    if compile_result.exit_code != 0:
        _, exception, exc_traceback = compile_result.exc_info
        print('expected stdout:')
        print(item.stderr.decode())
        print('actual stdout:')
        print(compile_result.output)
        assert compile_result.output and compile_result.output == item.stderr.decode()
    else:
        try:
            run_result = check_output(binary_path, universal_newlines=True)
        except CalledProcessError as e:
            # TODO: make us able to expect and verify non-zero exit codes
            run_result = e.stdout
        assert run_result == item.stdout.decode()
