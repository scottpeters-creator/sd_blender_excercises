"""Output naming and directory utilities.

Provides deterministic output path generation from input paths, with
automatic directory creation.  Reusable across all exercises.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def ensure_directory(path: str) -> str:
    """Ensure the directory for *path* exists, creating it if necessary.

    If *path* looks like a file (has an extension), the **parent** directory
    is created.  If it looks like a directory (no extension or ends with
    ``/``), the directory itself is created.

    Returns the (possibly created) directory path.
    """
    p = Path(path)
    if p.suffix or not path.endswith("/"):
        # Looks like a file — ensure parent exists
        directory = p.parent
    else:
        directory = p
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory)


class OutputNamer:
    """Deterministic output path generator.

    Given an output root directory, maps each input file to a per-model
    subdirectory named after the input's stem::

        namer = OutputNamer("/tmp/reports")
        namer("/data/objaverse/abc123.glb", "report.json")
        # → "/tmp/reports/abc123/report.json"

    Directories are created automatically on first call.

    Args:
        output_dir: Root directory for all outputs.
        mkdir:      If True (default), create directories on the fly.
    """

    def __init__(self, output_dir: str, *, mkdir: bool = True) -> None:
        self.output_dir = str(output_dir)
        self._mkdir = mkdir

    def __call__(
        self,
        input_path: str,
        filename: Optional[str] = None,
    ) -> str:
        """Return the output path for *input_path*.

        Args:
            input_path: Path to the source file (e.g., ``model.glb``).
            filename:   Output filename within the model subdirectory.
                        If ``None``, returns the subdirectory path (with
                        trailing ``/``).

        Returns:
            Absolute output path.
        """
        stem = Path(input_path).stem
        model_dir = os.path.join(self.output_dir, stem)

        if self._mkdir:
            os.makedirs(model_dir, exist_ok=True)

        if filename is None:
            return model_dir
        return os.path.join(model_dir, filename)

    def model_dir(self, input_path: str) -> str:
        """Return (and optionally create) the per-model subdirectory."""
        return self(input_path)
