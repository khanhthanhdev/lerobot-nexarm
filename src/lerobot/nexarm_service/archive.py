# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Tar.gz transfer of a directory (a dataset or a checkpoint) between two engines.

Used by remote training: datasets go to the GPU server and checkpoints come back through the engine's
archive endpoints, over the SSH tunnel, instead of manual rsync steps.
"""

import shutil
import tarfile
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path

CHUNK = 1024 * 1024
# A LeRobot dataset with video can be tens of GB; the cap only stops a runaway or hostile stream.
DEFAULT_MAX_BYTES = 200 * 1024**3


class ArchiveError(ValueError):
    pass


def pack(directory: Path, scratch: Path) -> Path:
    """Write ``directory`` to a temporary .tar.gz under ``scratch`` and return its path."""
    scratch.mkdir(parents=True, exist_ok=True)
    target = scratch / f"{uuid.uuid4().hex}.tar.gz"
    with tarfile.open(target, "w:gz") as tar:
        tar.add(directory, arcname=".")
    return target


def read_chunks(path: Path) -> Iterator[bytes]:
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            yield chunk


def _check_member(member: tarfile.TarInfo) -> None:
    name = member.name
    parts = Path(name).parts
    if name.startswith(("/", "\\")) or ".." in parts or (parts and ":" in parts[0]):
        raise ArchiveError(f"unsafe path in archive: {name!r}")
    if not (member.isfile() or member.isdir()):
        raise ArchiveError(f"unsupported entry in archive (links and devices are refused): {name!r}")


def extract(archive: Path, destination: Path, max_bytes: int = DEFAULT_MAX_BYTES) -> int:
    """Extract ``archive`` into ``destination`` (which must not exist); returns the bytes written.

    Extraction goes to a staging directory next to the destination and is moved into place only when
    the whole archive was valid, so a failed transfer leaves nothing behind.
    """
    if destination.exists():
        raise ArchiveError(f"{destination.name} already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".extract-", dir=destination.parent))
    total = 0
    try:
        try:
            with tarfile.open(archive, "r:gz") as tar:
                for member in tar:
                    _check_member(member)
                    total += max(member.size, 0)
                    if total > max_bytes:
                        raise ArchiveError("archive is larger than the allowed size")
                    tar.extract(member, staging, filter="data")
        except (tarfile.TarError, EOFError, OSError) as exc:
            raise ArchiveError(f"not a valid archive: {exc}") from exc
        shutil.move(str(staging), str(destination))
        return total
    finally:
        shutil.rmtree(staging, ignore_errors=True)
