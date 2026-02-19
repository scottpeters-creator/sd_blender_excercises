"""Animation utilities: detection, orbits, spline paths.

Used by Ex 3 (procedural camera) and Ex 5 (orbit rendering).
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import bpy
from mathutils import Vector

logger = logging.getLogger("pipeline.bpy.animation")


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def is_animated(
    obj: bpy.types.Object,
    threshold: float = 0.01,
) -> bool:
    """Check whether an object has meaningful animation (location drift).

    Evaluates the object's world-space location at the first and last
    keyframes. If the displacement exceeds `threshold`, the object is
    considered animated/moving.

    Args:
        obj: The Blender object to inspect.
        threshold: Minimum displacement (meters) to flag as animated.

    Returns:
        True if the object is animated beyond the threshold.
    """
    if not obj.animation_data or not obj.animation_data.action:
        return False

    action = obj.animation_data.action
    frame_start = int(action.frame_range[0])
    frame_end = int(action.frame_range[1])

    if frame_start == frame_end:
        return False

    scene = bpy.context.scene
    orig_frame = scene.frame_current

    scene.frame_set(frame_start)
    loc_start = obj.matrix_world.translation.copy()

    scene.frame_set(frame_end)
    loc_end = obj.matrix_world.translation.copy()

    scene.frame_set(orig_frame)

    displacement = (loc_end - loc_start).length
    is_moving = displacement > threshold
    logger.debug(
        "'%s': displacement=%.4f, threshold=%.4f, animated=%s",
        obj.name, displacement, threshold, is_moving,
    )
    return is_moving


def has_any_animation(obj: bpy.types.Object) -> bool:
    """Quick check: does the object have any animation data at all?"""
    return obj.animation_data is not None and obj.animation_data.action is not None


# ---------------------------------------------------------------------------
# Orbit animation
# ---------------------------------------------------------------------------

def create_orbit(
    camera: bpy.types.Object,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 5.0,
    height: float = 2.0,
    frame_count: int = 360,
    frame_start: int = 1,
) -> None:
    """Keyframe a circular orbit for the camera around a center point.

    The camera orbits in the XY plane at the given height, with one
    keyframe per frame for smooth motion.

    Args:
        camera: Camera object to animate.
        center: World-space center of the orbit.
        radius: Orbit radius.
        height: Camera height (Z) above the center.
        frame_count: Total number of frames for a full orbit.
        frame_start: First frame number.
    """
    scene = bpy.context.scene
    scene.frame_start = frame_start
    scene.frame_end = frame_start + frame_count - 1

    cx, cy, cz = center
    for i in range(frame_count):
        frame = frame_start + i
        angle = 2.0 * math.pi * i / frame_count
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        z = cz + height

        camera.location = (x, y, z)
        camera.keyframe_insert(data_path="location", frame=frame)

    logger.info(
        "Created orbit: %d frames, radius=%.2f, height=%.2f, center=%s",
        frame_count, radius, height, center,
    )


# ---------------------------------------------------------------------------
# Spline / bezier paths
# ---------------------------------------------------------------------------

def create_bezier_path(
    points: List[Tuple[float, float, float]],
    name: str = "CameraPath",
    resolution: int = 64,
) -> bpy.types.Object:
    """Create a Bezier curve through the given control points.

    Args:
        points: List of (x, y, z) control point coordinates (minimum 2).
        name: Name for the curve object.
        resolution: Preview resolution of the curve.

    Returns:
        The curve object added to the scene.

    Raises:
        ValueError: If fewer than 2 points are provided.
    """
    if len(points) < 2:
        raise ValueError("Need at least 2 control points for a Bezier path")

    curve_data = bpy.data.curves.new(name, type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.resolution_u = resolution

    spline = curve_data.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)

    for i, (x, y, z) in enumerate(points):
        bp = spline.bezier_points[i]
        bp.co = (x, y, z)
        bp.handle_left_type = "AUTO"
        bp.handle_right_type = "AUTO"

    curve_obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(curve_obj)

    logger.info("Created Bezier path '%s' with %d control points", name, len(points))
    return curve_obj


def constrain_to_path(
    obj: bpy.types.Object,
    curve: bpy.types.Object,
    frame_count: int = 150,
    frame_start: int = 1,
    use_fixed_location: bool = False,
) -> bpy.types.Constraint:
    """Constrain an object to follow a curve path over the given frame range.

    Args:
        obj: The object to constrain (typically a camera).
        curve: The curve object to follow.
        frame_count: Duration in frames.
        frame_start: Start frame.
        use_fixed_location: If True, use offset along path instead of percentage.

    Returns:
        The Follow Path constraint.
    """
    curve.data.use_path = True
    curve.data.path_duration = frame_count

    constraint = obj.constraints.new(type="FOLLOW_PATH")
    constraint.target = curve
    constraint.use_fixed_location = use_fixed_location
    constraint.use_curve_follow = True

    constraint.offset = 0.0
    constraint.keyframe_insert(data_path="offset", frame=frame_start)
    constraint.offset = -100.0
    constraint.keyframe_insert(data_path="offset", frame=frame_start + frame_count - 1)

    scene = bpy.context.scene
    scene.frame_start = frame_start
    scene.frame_end = frame_start + frame_count - 1

    logger.info(
        "Constrained '%s' to path '%s' (%d frames)",
        obj.name, curve.name, frame_count,
    )
    return constraint


def create_linear_keyframes(
    camera: bpy.types.Object,
    positions: List[Tuple[float, float, float]],
    frame_count: int = 150,
    frame_start: int = 1,
) -> None:
    """Keyframe linear interpolation between a sequence of positions.

    Frames are distributed evenly across the positions.

    Args:
        camera: Object to animate.
        positions: List of (x, y, z) waypoints (minimum 2).
        frame_count: Total frame count.
        frame_start: Starting frame.
    """
    if len(positions) < 2:
        raise ValueError("Need at least 2 positions for linear keyframes")

    scene = bpy.context.scene
    scene.frame_start = frame_start
    scene.frame_end = frame_start + frame_count - 1

    segment_count = len(positions) - 1
    frames_per_segment = frame_count / segment_count

    for seg in range(segment_count):
        p_start = Vector(positions[seg])
        p_end = Vector(positions[seg + 1])
        f_start = frame_start + int(seg * frames_per_segment)
        f_end = frame_start + int((seg + 1) * frames_per_segment)

        for f in range(f_start, f_end + 1):
            t = (f - f_start) / max(f_end - f_start, 1)
            loc = p_start.lerp(p_end, t)
            camera.location = loc
            camera.keyframe_insert(data_path="location", frame=f)

    logger.info(
        "Created %d linear keyframes across %d waypoints",
        frame_count, len(positions),
    )
