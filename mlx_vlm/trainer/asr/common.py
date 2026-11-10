"""Model-agnostic ASR plumbing: examples are named after forward arguments."""

import inspect
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any


class ASRDataset:
    """Speech rows mapped lazily through a ``row -> example`` processor.

    Rows follow the shared speech format ``{"audio": path, "text": str}`` with
    optional fields such as ``language``. Relative audio paths resolve against
    the folder the split was loaded from.
    """

    def __init__(self, rows: Sequence[Any], process: Callable | None):
        if process is None:
            raise ValueError(
                "This ASR model has no built-in preprocessing; pass "
                "processing_class as a row -> example callable or use prepared=True."
            )
        self.rows, self.process = rows, process
        self.base_path = getattr(rows, "base_path", None)

    def __getitem__(self, index: int):
        row = self.rows[index]
        audio = row.get("audio") if isinstance(row, Mapping) else None
        if self.base_path is not None and isinstance(audio, str):
            row = {**row, "audio": str(Path(self.base_path) / audio)}
        return self.process(row)

    def __len__(self) -> int:
        return len(self.rows)


def named_forward(model, forward: Callable | None = None) -> Callable:
    """Return ``call(fields)`` that passes only the fields the forward names.

    ``forward(model, **fields)`` replaces ``model(**fields)`` for models whose
    own call is an inference interface rather than a logits function.
    """
    function = model if forward is None else partial(forward, model)
    parameters = inspect.signature(function).parameters.values()
    accepted = (
        None
        if any(parameter.kind is parameter.VAR_KEYWORD for parameter in parameters)
        else {parameter.name for parameter in parameters}
    )

    def call(fields: Mapping[str, Any]):
        if accepted is not None:
            fields = {key: value for key, value in fields.items() if key in accepted}
        return function(**fields)

    return call
