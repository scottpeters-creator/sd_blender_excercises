"""Scene lifecycle management.

Shared across all exercises: reset scene, purge orphan data, move objects.
"""

from __future__ import annotations

import logging
from typing import Optional

import bpy

logger = logging.getLogger("pipeline.bpy.scene")


def reset_scene() -> None:
    """Clear the scene completely — remove all objects, then purge orphan data.

    Equivalent to opening a fresh empty file. Preferred over
    `bpy.ops.wm.read_homefile(use_empty=True)` when you want to keep the
    current file context but start with a blank scene.
    """
    bpy.ops.wm.read_homefile(use_empty=True)
    purge_orphan_data()
    logger.info("Scene reset to empty state")


def purge_orphan_data() -> None:
    """Remove all orphan data blocks (meshes, materials, textures, images, etc.).

    Blender does not garbage-collect automatically in scripted loops.
    Call this between iterations in batch processing (Ex 4, Ex 5) to prevent
    memory leaks.
    """
    for collection_name in (
        "meshes", "materials", "textures", "images",
        "node_groups", "cameras", "lights", "curves",
        "armatures", "actions", "worlds",
    ):
        collection = getattr(bpy.data, collection_name, None)
        if collection is None:
            continue
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)
    logger.debug("Orphan data purged")


def delete_all_objects() -> None:
    """Delete every object in the current scene without resetting the file."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=True)
    logger.debug("All objects deleted from scene")


def move_to_origin(
    obj: bpy.types.Object,
    zero_rotation: bool = False,
) -> None:
    """Move an object so its geometric center is at the world origin.

    Args:
        obj: The Blender object to reposition.
        zero_rotation: If True, also clear rotation.
    """
    bbox_corners = [obj.matrix_world @ bpy.mathutils.Vector(c) for c in obj.bound_box]
    center = sum(bbox_corners, bpy.mathutils.Vector()) / 8
    obj.location -= center
    if zero_rotation:
        obj.rotation_euler = (0.0, 0.0, 0.0)
    logger.debug("Moved '%s' to origin (offset: %s)", obj.name, -center)


def get_scene_bounds(
    scene: Optional[bpy.types.Scene] = None,
) -> tuple:
    """Compute the axis-aligned bounding box of all mesh objects in the scene.

    Returns:
        (min_corner: Vector, max_corner: Vector) in world space.
        Returns (Vector((0,0,0)), Vector((0,0,0))) if scene has no meshes.
    """
    from mathutils import Vector

    scene = scene or bpy.context.scene
    meshes = [o for o in scene.objects if o.type == "MESH"]
    if not meshes:
        return Vector((0, 0, 0)), Vector((0, 0, 0))

    all_coords = []
    for obj in meshes:
        all_coords.extend(obj.matrix_world @ Vector(c) for c in obj.bound_box)

    xs = [v.x for v in all_coords]
    ys = [v.y for v in all_coords]
    zs = [v.z for v in all_coords]
    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))
