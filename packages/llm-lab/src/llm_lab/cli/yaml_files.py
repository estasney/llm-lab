from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import yaml
from pydantic import ValidationError

from llm_lab.errors import CommandError


@contextmanager
def reading_yaml_file(path: Path | str, kind: str) -> Generator[None]:
    try:
        yield
    except yaml.YAMLError as error:
        raise CommandError(f"{path} is not valid YAML: {error}") from error
    except ValidationError as error:
        problems = [f"  {'.'.join(str(part) for part in item['loc'])}: {item['msg']}" for item in error.errors()]
        raise CommandError("\n".join([f"{path} is not a valid {kind}:", *problems])) from error
