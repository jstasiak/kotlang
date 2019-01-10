from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Context:
    def load_module_text(self, module: str) -> ModuleFile:
        # TODO: make current directory independent and implement module search path
        path = Path('modules') / f'{module}.kot'
        text = self.load_file_text(path)
        return ModuleFile(str(path), text)

    def load_file_text(self, filename: Path) -> str:
        return filename.read_text()


@dataclass
class ModuleFile:
    filename: str
    text: str
