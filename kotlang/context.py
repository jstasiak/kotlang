from dataclasses import dataclass
from pathlib import Path


@dataclass
class Context:
    def load_module_text(self, module: str) -> str:
        # TODO: make current directory independent and implement module search path
        return self.load_file_text(Path('modules') / f'{module}.kot')

    def load_file_text(self, filename: Path) -> str:
        return filename.read_text()
