"""Mesh collection and geometry analysis.

Used by Ex 1 (mini inspector), Ex 2 (bounding box metadata), and
Ex 4 (batch geometry stats).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import bpy
from mathutils import Vector

logger = logging.getLogger("pipeline.bpy.mesh")


def collect_meshes(
    scene: Optional[bpy.types.Scene] = None,
) -> List[bpy.types.Object]:
    """Return all MESH-type objects in the given scene.

    Args:
        scene: The Blender scene to scan. Defaults to the active scene.

    Returns:
        List of bpy.types.Object with type == 'MESH'.
    """
    scene = scene or bpy.context.scene
    meshes = [o for o in scene.objects if o.type == "MESH"]
    logger.debug("Collected %d mesh object(s)", len(meshes))
    return meshes


def compute_geometry_counts(
    objects: List[bpy.types.Object],
) -> Dict[str, int]:
    """Compute total vertex and face counts across all mesh objects.

    Uses the evaluated (depsgraph) mesh to account for modifiers.

    Returns:
        {"vertices": int, "faces": int}
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    total_verts = 0
    total_faces = 0
    for obj in objects:
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        total_verts += len(mesh.vertices)
        total_faces += len(mesh.polygons)
        eval_obj.to_mesh_clear()
    return {"vertices": total_verts, "faces": total_faces}


def compute_topology(
    objects: List[bpy.types.Object],
) -> Dict[str, int]:
    """Classify faces by vertex count: triangles, quads, ngons.

    Returns:
        {"triangles": int, "quads": int, "ngons": int}
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    tris = 0
    quads = 0
    ngons = 0
    for obj in objects:
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        for poly in mesh.polygons:
            n = poly.loop_total
            if n == 3:
                tris += 1
            elif n == 4:
                quads += 1
            else:
                ngons += 1
        eval_obj.to_mesh_clear()
    return {"triangles": tris, "quads": quads, "ngons": ngons}


def compute_world_bbox(
    objects: List[bpy.types.Object],
) -> Dict[str, Any]:
    """Compute the combined world-space axis-aligned bounding box.

    Returns:
        {
            "center": [cx, cy, cz],
            "dimensions": [dx, dy, dz],
            "min": [minx, miny, minz],
            "max": [maxx, maxy, maxz],
        }
    """
    if not objects:
        zero = [0.0, 0.0, 0.0]
        return {"center": zero, "dimensions": zero, "min": zero, "max": zero}

    all_coords: List[Vector] = []
    for obj in objects:
        all_coords.extend(obj.matrix_world @ Vector(c) for c in obj.bound_box)

    xs = [v.x for v in all_coords]
    ys = [v.y for v in all_coords]
    zs = [v.z for v in all_coords]

    min_v = [min(xs), min(ys), min(zs)]
    max_v = [max(xs), max(ys), max(zs)]
    center = [(a + b) / 2 for a, b in zip(min_v, max_v)]
    dims = [b - a for a, b in zip(min_v, max_v)]

    return {
        "center": [round(v, 4) for v in center],
        "dimensions": [round(v, 4) for v in dims],
        "min": [round(v, 4) for v in min_v],
        "max": [round(v, 4) for v in max_v],
    }


def get_material_names(
    objects: List[bpy.types.Object],
) -> List[str]:
    """Collect unique material names from the given mesh objects.

    Returns:
        Sorted list of material name strings.
    """
    names: set = set()
    for obj in objects:
        for slot in obj.material_slots:
            if slot.material:
                names.add(slot.material.name)
    return sorted(names)
