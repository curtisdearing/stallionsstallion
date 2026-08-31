"""Atomic file writes, so a crash cannot leave a half-file where a reader looks.

Every pointer and manifest in this pipeline is read by something that has to
decide whether a forecast exists: the snapshot `latest` pointer, `frozen.json`,
the public health status. A torn write to any of them is worse than a missing
one, because a partial JSON document is indistinguishable from a corrupt one
and both used to be read as "nothing here" -- which is the reading that lets a
run overwrite a record it should have preserved.

`os.replace` is atomic within a filesystem, so a reader sees either the whole
previous file or the whole new one, never a prefix of either.
"""

from __future__ import annotations

import json
import os
from typing import Any


def write_text(path: str, text: str, *, encoding: str = "utf-8") -> str:
    """Replace `path` with `text`, atomically. Returns the path."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
    return path


def write_json(path: str, payload: Any, *, indent: int = 2,
               sort_keys: bool = True) -> str:
    """Serialise then replace, atomically.

    Serialising BEFORE opening the target matters: an unserialisable payload
    raises without having touched the file, so a bad write cannot destroy a
    good pointer.
    """
    text = json.dumps(payload, indent=indent, sort_keys=sort_keys) + "\n"
    return write_text(path, text)
