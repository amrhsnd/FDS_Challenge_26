"""Streaming readers for the official competition inputs."""
import json
from pathlib import Path


def iter_battles(path):
    """Yield one battle dictionary per nonempty JSONL line without changing it."""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                battle = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}, line {line_number}") from error
            if not isinstance(battle, dict):
                raise ValueError(f"Expected a battle object in {path}, line {line_number}")
            yield battle
