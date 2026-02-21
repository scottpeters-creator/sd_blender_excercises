"""Spatial utilities: surface sampling and angular sorting.

Used by Ex 3 (procedural camera) for aim target generation and
coherent camera path ordering.
"""

from __future__ import annotations

import logging
import math
import random
from typing import List, Optional, Tuple

import bpy
from mathutils import Vector

logger = logging.getLogger("pipeline.bpy.spatial")


# ---------------------------------------------------------------------------
# Surface sampling for aim targets
# ---------------------------------------------------------------------------

def sample_surface_points(
    count: int,
    scene: Optional[bpy.types.Scene] = None,
) -> List[Tuple[float, float, float]]:
    """Sample random points on mesh surfaces for camera aim targets.

    Picks random faces weighted by area, then random barycentric
    coordinates within each face.

    Args:
        count: Number of surface points to sample.
        scene: Blender scene to sample from.

    Returns:
        List of (x, y, z) world-space surface points.
    """
    scene = scene or bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()

    face_data = []
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        mesh.calc_loop_triangles()
        mat = obj.matrix_world
        for tri in mesh.loop_triangles:
            v0 = mat @ Vector(mesh.vertices[tri.vertices[0]].co)
            v1 = mat @ Vector(mesh.vertices[tri.vertices[1]].co)
            v2 = mat @ Vector(mesh.vertices[tri.vertices[2]].co)
            area = tri.area
            if area > 0:
                face_data.append((v0, v1, v2, area))
        eval_obj.to_mesh_clear()

    if not face_data:
        logger.warning("No mesh faces found for surface sampling")
        return [(0.0, 0.0, 0.0)] * count

    total_area = sum(f[3] for f in face_data)
    weights = [f[3] / total_area for f in face_data]

    points = []
    for _ in range(count):
        idx = _weighted_choice(weights)
        v0, v1, v2, _ = face_data[idx]
        r1, r2 = random.random(), random.random()
        if r1 + r2 > 1.0:
            r1, r2 = 1.0 - r1, 1.0 - r2
        pt = v0 + r1 * (v1 - v0) + r2 * (v2 - v0)
        points.append((pt.x, pt.y, pt.z))

    logger.info("Sampled %d surface aim points from %d triangles", count, len(face_data))
    return points


def _weighted_choice(weights: List[float]) -> int:
    """Pick an index from a list of normalized weights."""
    r = random.random()
    cumulative = 0.0
    for i, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            return i
    return len(weights) - 1


# ---------------------------------------------------------------------------
# Angular sorting
# ---------------------------------------------------------------------------

def sort_by_angle_around_z(
    points: List[Tuple[float, float, float]],
    center: Optional[Tuple[float, float, float]] = None,
    clockwise: bool = True,
) -> List[Tuple[float, float, float]]:
    """Sort points by polar angle around the Z axis relative to a center.

    Produces a coherent sweep path (clockwise or counter-clockwise)
    instead of random jumps.

    Args:
        points: List of (x, y, z) positions.
        center: Reference point for angle calculation. Defaults to
            the centroid of the input points.
        clockwise: If True, sort descending angle (CW when viewed from +Z).

    Returns:
        Sorted copy of the points list.
    """
    if not points:
        return points

    if center is None:
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        cz = sum(p[2] for p in points) / len(points)
        center = (cx, cy, cz)

    def angle_key(p):
        return math.atan2(p[1] - center[1], p[0] - center[0])

    sorted_pts = sorted(points, key=angle_key, reverse=clockwise)
    logger.debug("Sorted %d points by angle around Z (clockwise=%s)", len(sorted_pts), clockwise)
    return sorted_pts
