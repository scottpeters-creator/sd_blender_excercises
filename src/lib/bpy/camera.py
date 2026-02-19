"""Camera creation, framing, and constraints.

Used by Ex 2 (thumbnail rendering), Ex 3 (procedural camera), and
Ex 5 (orbit animation).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import bpy
from mathutils import Vector

logger = logging.getLogger("pipeline.bpy.camera")


def create_camera(
    name: str = "Camera",
    location: Tuple[float, float, float] = (0.0, -5.0, 2.0),
    focal_length: float = 50.0,
    sensor_width: float = 36.0,
    clip_start: float = 0.1,
    clip_end: float = 1000.0,
) -> bpy.types.Object:
    """Create a camera and add it to the active scene.

    Returns:
        The new camera object.
    """
    cam_data = bpy.data.cameras.new(name)
    cam_data.lens = focal_length
    cam_data.sensor_width = sensor_width
    cam_data.clip_start = clip_start
    cam_data.clip_end = clip_end

    cam_obj = bpy.data.objects.new(name, cam_data)
    cam_obj.location = location
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    logger.info("Created camera '%s' at %s (focal=%.1fmm)", name, location, focal_length)
    return cam_obj


def frame_objects(
    camera: bpy.types.Object,
    objects: List[bpy.types.Object],
    margin: float = 1.3,
) -> None:
    """Position the camera so that all objects fit within the frame.

    Adjusts distance along the camera's current look direction so the
    combined bounding sphere of the objects is visible with the given margin.

    Args:
        camera: Camera object to reposition.
        objects: Objects that must be in frame.
        margin: Multiplier for extra padding (1.0 = tight fit).
    """
    if not objects:
        return

    all_coords: List[Vector] = []
    for obj in objects:
        all_coords.extend(obj.matrix_world @ Vector(c) for c in obj.bound_box)

    center = sum(all_coords, Vector()) / len(all_coords)
    radius = max((v - center).length for v in all_coords)

    cam_data = camera.data
    fov = 2.0 * math.atan(cam_data.sensor_width / (2.0 * cam_data.lens))
    distance = (radius * margin) / math.tan(fov / 2.0)

    direction = (camera.location - center).normalized()
    if direction.length < 0.001:
        direction = Vector((0.0, -1.0, 0.5)).normalized()

    camera.location = center + direction * distance
    _point_at(camera, center)

    logger.info(
        "Framed %d object(s): center=%s, radius=%.2f, distance=%.2f",
        len(objects), center, radius, distance,
    )


def add_track_to(
    camera: bpy.types.Object,
    target: bpy.types.Object,
    track_axis: str = "TRACK_NEGATIVE_Z",
    up_axis: str = "UP_Y",
) -> bpy.types.Constraint:
    """Add a Track To constraint so the camera always faces the target object.

    Args:
        camera: The camera (or any object) to constrain.
        target: The object the camera will track.
        track_axis: Forward axis for the constraint.
        up_axis: Up axis for the constraint.

    Returns:
        The created constraint.
    """
    constraint = camera.constraints.new(type="TRACK_TO")
    constraint.target = target
    constraint.track_axis = track_axis
    constraint.up_axis = up_axis
    logger.debug("Added Track To constraint on '%s' → '%s'", camera.name, target.name)
    return constraint


def add_track_to_location(
    camera: bpy.types.Object,
    location: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    """Create an empty at the given location and track the camera to it.

    Returns:
        The target empty object.
    """
    empty = bpy.data.objects.new("CameraTarget", None)
    empty.location = location
    empty.empty_display_type = "PLAIN_AXES"
    empty.empty_display_size = 0.1
    bpy.context.collection.objects.link(empty)
    add_track_to(camera, empty)
    logger.info("Camera '%s' tracking location %s via empty", camera.name, location)
    return empty


def get_camera_intrinsics(
    camera: bpy.types.Object,
    scene: Optional[bpy.types.Scene] = None,
) -> Dict[str, Any]:
    """Extract camera intrinsics and extrinsics as a JSON-serializable dict.

    Returns:
        {
            "focal_length": float,
            "sensor_width": float,
            "position": [x, y, z],
            "rotation_euler": [rx, ry, rz],
            "clip_start": float,
            "clip_end": float,
            "resolution": [width, height],
        }
    """
    scene = scene or bpy.context.scene
    cam_data = camera.data
    return {
        "focal_length": cam_data.lens,
        "sensor_width": cam_data.sensor_width,
        "position": [round(v, 4) for v in camera.location],
        "rotation_euler": [round(v, 4) for v in camera.rotation_euler],
        "clip_start": cam_data.clip_start,
        "clip_end": cam_data.clip_end,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
    }


def _point_at(camera: bpy.types.Object, target: Vector) -> None:
    """Rotate the camera to look at a world-space point."""
    direction = target - camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()
