"""Model import/open utilities.

Handles .glb, .gltf, .obj, .fbx, .blend — dispatches to the correct
bpy.ops.import_scene.* operator based on file extension.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import bpy

logger = logging.getLogger("pipeline.bpy.io")

_IMPORTERS = {
    ".glb": lambda p: bpy.ops.import_scene.gltf(filepath=p),
    ".gltf": lambda p: bpy.ops.import_scene.gltf(filepath=p),
    ".obj": lambda p: bpy.ops.import_scene.obj(filepath=p) if hasattr(bpy.ops.import_scene, "obj") else bpy.ops.wm.obj_import(filepath=p),
    ".fbx": lambda p: bpy.ops.import_scene.fbx(filepath=p),
    ".stl": lambda p: bpy.ops.import_mesh.stl(filepath=p) if hasattr(bpy.ops.import_mesh, "stl") else bpy.ops.wm.stl_import(filepath=p),
}


def import_model(
    filepath: str,
    *,
    link: bool = False,
) -> List[bpy.types.Object]:
    """Import a 3D model file and return the newly imported objects.

    Args:
        filepath: Absolute or relative path to the model file.
        link: If True, link rather than append (for .blend only).

    Returns:
        List of bpy.types.Object that were added to the scene.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported.
        RuntimeError: If the import operator fails.
    """
    filepath = os.path.abspath(filepath)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    existing = set(bpy.context.scene.objects)

    if ext == ".blend":
        _import_blend(filepath, link=link)
    elif ext in _IMPORTERS:
        try:
            _IMPORTERS[ext](filepath)
        except Exception as exc:
            raise RuntimeError(f"Import failed for {filepath}: {exc}") from exc
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    new_objects = [o for o in bpy.context.scene.objects if o not in existing]
    logger.info("Imported %d object(s) from '%s'", len(new_objects), os.path.basename(filepath))
    return new_objects


def _import_blend(filepath: str, *, link: bool = False) -> None:
    """Append or link all objects from a .blend file."""
    with bpy.data.libraries.load(filepath, link=link) as (data_from, data_to):
        data_to.objects = data_from.objects

    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)


def open_blend(filepath: str) -> None:
    """Open a .blend file directly (replaces current scene).

    Args:
        filepath: Path to the .blend file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    filepath = os.path.abspath(filepath)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Blend file not found: {filepath}")

    bpy.ops.wm.open_mainfile(filepath=filepath)
    logger.info("Opened blend file: '%s'", os.path.basename(filepath))


def supported_extensions() -> List[str]:
    """Return list of supported import file extensions."""
    return list(_IMPORTERS.keys()) + [".blend"]
