"""Shared Blender (bpy) utility library.

Provides reusable functions for scene management, model I/O, mesh analysis,
camera setup, rendering, and animation across all exercises.

Modules:
    scene      — reset, purge, move_to_origin, get_scene_bounds
    io         — import_model, open_blend
    mesh       — collect_meshes, geometry counts, topology, world bbox, materials
    camera     — create_camera, frame_objects, track-to, intrinsics
    render     — Cycles/Workbench config, GPU, depth/normal/edge passes, video
    animation  — is_animated, orbit, bezier path, linear keyframes

Usage from exercise scripts:

    from lib.bpy.scene import reset_scene, purge_orphan_data
    from lib.bpy.io import import_model
    from lib.bpy.mesh import collect_meshes, compute_geometry_counts
    from lib.bpy.camera import create_camera, frame_objects
    from lib.bpy.render import configure_cycles, enable_gpu, render_frame
    from lib.bpy.animation import create_orbit, is_animated
"""

__all__ = [
    "scene",
    "io",
    "mesh",
    "camera",
    "render",
    "animation",
]
