"""Render engine configuration, GPU setup, and render passes.

Used by Ex 2 (multi-modality thumbnails), Ex 3 (video output), and
Ex 5 (HPC GPU rendering).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional, Tuple

import bpy

logger = logging.getLogger("pipeline.bpy.render")


# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------

def configure_cycles(
    resolution: Tuple[int, int] = (512, 512),
    samples: int = 128,
    use_denoising: bool = True,
    transparent_background: bool = False,
) -> None:
    """Configure the Cycles render engine with common defaults.

    Args:
        resolution: (width, height) in pixels.
        samples: Number of render samples.
        use_denoising: Enable the built-in denoiser.
        transparent_background: Render with alpha background.
    """
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100
    scene.cycles.samples = samples
    scene.cycles.use_denoising = use_denoising
    scene.render.film_transparent = transparent_background
    logger.info(
        "Cycles configured: %dx%d, %d samples, denoise=%s",
        resolution[0], resolution[1], samples, use_denoising,
    )


def configure_workbench(
    resolution: Tuple[int, int] = (512, 512),
    lighting: str = "STUDIO",
    color_type: str = "MATERIAL",
) -> None:
    """Configure the Workbench render engine.

    Args:
        resolution: (width, height) in pixels.
        lighting: Lighting mode ('STUDIO', 'MATCAP', 'FLAT').
        color_type: Color mode ('MATERIAL', 'SINGLE', 'TEXTURE', etc.).
    """
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100
    scene.display.shading.light = lighting
    scene.display.shading.color_type = color_type
    logger.info("Workbench configured: %dx%d", resolution[0], resolution[1])


def set_resolution(width: int, height: int) -> None:
    """Set render resolution without changing the engine."""
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

def enable_gpu() -> None:
    """Enable GPU rendering with OPTIX (preferred) or CUDA fallback.

    On macOS, falls back to CPU silently. Logs device enumeration.
    """
    if sys.platform == "darwin":
        logger.info("macOS detected — using CPU rendering")
        bpy.context.scene.cycles.device = "CPU"
        return

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    logger.info("CUDA_VISIBLE_DEVICES = '%s'", cuda_visible)

    if "cycles" not in bpy.context.preferences.addons:
        logger.error("Cycles addon not enabled — falling back to CPU")
        bpy.context.scene.cycles.device = "CPU"
        return

    bpy.context.scene.cycles.device = "GPU"
    render_prefs = bpy.context.preferences.addons["cycles"].preferences

    for gpu_type in ("OPTIX", "CUDA"):
        render_prefs.compute_device_type = gpu_type
        render_prefs.refresh_devices()
        available = [d for d in render_prefs.devices if d.type == gpu_type]
        if not available:
            logger.info("No %s devices found", gpu_type)
            continue

        gpu_count = 0
        for d in available:
            d.use = True
            gpu_count += 1
            logger.info("  Enabled: %s (type: %s)", d.name, d.type)

        logger.info("Using %s with %d GPU(s)", gpu_type, gpu_count)
        return

    logger.warning("No supported GPU found — falling back to CPU")
    bpy.context.scene.cycles.device = "CPU"


# ---------------------------------------------------------------------------
# Render passes / output
# ---------------------------------------------------------------------------

def setup_depth_pass(
    normalize: bool = True,
    output_path: Optional[str] = None,
    file_format: str = "PNG",
) -> None:
    """Enable the Z (depth) render pass via the compositor.

    Args:
        normalize: If True, add a Normalize node so depth maps to 0–1 range.
        output_path: If set, add a File Output node writing to this path.
        file_format: Image format for the file output node.
    """
    scene = bpy.context.scene
    scene.render.use_compositing = True
    scene.view_layers[0].use_pass_z = True

    tree = _ensure_compositor_tree(scene)
    nodes = tree.nodes
    links = tree.links

    render_layers = _get_or_create_node(nodes, "CompositorNodeRLayers", "Render Layers")

    if normalize:
        norm = _get_or_create_node(nodes, "CompositorNodeNormalize", "Normalize Depth")
        links.new(render_layers.outputs["Depth"], norm.inputs[0])
        depth_out = norm.outputs[0]
    else:
        depth_out = render_layers.outputs["Depth"]

    if output_path:
        file_node = _make_file_output(
            tree, os.path.dirname(output_path),
            os.path.basename(output_path), file_format,
        )
        links.new(depth_out, file_node.inputs[0])

    logger.info("Depth pass enabled (normalize=%s)", normalize)


def setup_normal_pass(
    output_path: Optional[str] = None,
    file_format: str = "PNG",
) -> None:
    """Enable the Normal render pass via the compositor.

    Args:
        output_path: If set, add a File Output node.
        file_format: Image format for the file output node.
    """
    scene = bpy.context.scene
    scene.render.use_compositing = True
    scene.view_layers[0].use_pass_normal = True

    tree = _ensure_compositor_tree(scene)
    nodes = tree.nodes
    links = tree.links

    render_layers = _get_or_create_node(nodes, "CompositorNodeRLayers", "Render Layers")

    if output_path:
        file_node = _make_file_output(
            tree, os.path.dirname(output_path),
            os.path.basename(output_path), file_format,
        )
        links.new(render_layers.outputs["Normal"], file_node.inputs[0])

    logger.info("Normal pass enabled")


def setup_edge_output(
    line_thickness: float = 1.5,
) -> None:
    """Enable Freestyle line-art rendering for edge/contour output.

    Args:
        line_thickness: Thickness of rendered lines in pixels.
    """
    scene = bpy.context.scene
    scene.render.use_freestyle = True
    view_layer = scene.view_layers[0]
    view_layer.use_freestyle = True

    if view_layer.freestyle_settings.linesets:
        lineset = view_layer.freestyle_settings.linesets[0]
    else:
        lineset = view_layer.freestyle_settings.linesets.new("LineSet")

    lineset.linestyle.thickness = line_thickness
    lineset.linestyle.color = (0.0, 0.0, 0.0)

    logger.info("Freestyle edge output enabled (thickness=%.1f)", line_thickness)


def render_frame(
    output_path: str,
    file_format: str = "PNG",
    color_depth: str = "8",
) -> str:
    """Render the current frame and save to disk.

    Args:
        output_path: Destination file path (without extension — Blender adds it).
        file_format: 'PNG', 'JPEG', 'OPEN_EXR', etc.
        color_depth: '8', '16', or '32' depending on format.

    Returns:
        The output path used.
    """
    scene = bpy.context.scene
    scene.render.image_settings.file_format = file_format
    scene.render.image_settings.color_depth = color_depth
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    logger.info("Rendered frame → '%s'", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Video output
# ---------------------------------------------------------------------------

def configure_video_output(
    output_path: str,
    fps: int = 30,
    codec: str = "H264",
    format: str = "FFMPEG",
    quality: str = "HIGH",
) -> None:
    """Configure FFmpeg video output settings.

    Args:
        output_path: Destination video file path.
        fps: Frames per second.
        codec: Video codec ('H264', 'MPEG4', etc.).
        format: Container format ('FFMPEG').
        quality: Output quality ('HIGH', 'MEDIUM', 'LOW').
    """
    scene = bpy.context.scene
    img_settings = scene.render.image_settings
    if hasattr(img_settings, "media_type"):
        img_settings.media_type = "VIDEO"
    img_settings.file_format = format
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = codec
    scene.render.ffmpeg.constant_rate_factor = quality
    scene.render.fps = fps
    scene.render.filepath = output_path
    logger.info(
        "Video output configured: %s, %d fps, codec=%s → '%s'",
        format, fps, codec, output_path,
    )


def render_animation() -> None:
    """Render the full animation timeline."""
    bpy.ops.render.render(animation=True)
    logger.info("Animation render complete")


# ---------------------------------------------------------------------------
# Multi-pass output (single render → multiple files)
# ---------------------------------------------------------------------------

def setup_multipass_output(
    output_dir: str,
    textured: bool = True,
    normal: bool = True,
    depth: bool = True,
    edge: bool = True,
    normalize_depth: bool = True,
    edge_thickness: float = 1.5,
    file_format: str = "PNG",
) -> None:
    """Build a compositor node graph that outputs multiple render passes
    from a single ``bpy.ops.render.render()`` call.

    Each enabled pass gets its own ``CompositorNodeOutputFile`` node.
    After calling this function, a single render writes all requested
    images simultaneously.

    Args:
        output_dir:      Directory for output files.
        textured:        Save the Combined (textured/lit) image.
        normal:          Save the Normal pass.
        depth:           Save the Depth pass.
        edge:            Save Freestyle line-art as a separate pass.
        normalize_depth: Apply a Normalize node to map depth to 0-1.
        edge_thickness:  Freestyle line thickness in pixels.
        file_format:     Image format for all outputs (default: PNG).
    """
    scene = bpy.context.scene
    view_layer = scene.view_layers[0]

    scene.render.use_compositing = True
    tree = _ensure_compositor_tree(scene)
    tree.nodes.clear()

    rl = tree.nodes.new(type="CompositorNodeRLayers")
    rl.location = (0, 0)

    x_offset = 600
    y_pos = 0

    if textured:
        fo = _make_file_output(tree, output_dir, "render_textured", file_format)
        fo.location = (x_offset, y_pos)
        tree.links.new(rl.outputs["Image"], fo.inputs[0])
        y_pos -= 200

    if normal:
        view_layer.use_pass_normal = True
        fo = _make_file_output(tree, output_dir, "render_normal", file_format)
        fo.location = (x_offset, y_pos)
        tree.links.new(rl.outputs["Normal"], fo.inputs[0])
        y_pos -= 200

    if depth:
        view_layer.use_pass_z = True
        if normalize_depth:
            norm = tree.nodes.new(type="CompositorNodeNormalize")
            norm.label = "Normalize Depth"
            norm.location = (x_offset - 200, y_pos)
            tree.links.new(rl.outputs["Depth"], norm.inputs[0])
            depth_source = norm.outputs[0]
        else:
            depth_source = rl.outputs["Depth"]
        fo = _make_file_output(tree, output_dir, "render_depth", file_format)
        fo.location = (x_offset, y_pos)
        tree.links.new(depth_source, fo.inputs[0])
        y_pos -= 200

    if edge:
        scene.render.use_freestyle = True
        view_layer.use_freestyle = True
        view_layer.freestyle_settings.as_render_pass = True

        if view_layer.freestyle_settings.linesets:
            lineset = view_layer.freestyle_settings.linesets[0]
        else:
            lineset = view_layer.freestyle_settings.linesets.new("LineSet")
        lineset.linestyle.thickness = edge_thickness
        lineset.linestyle.color = (0.0, 0.0, 0.0)

        fo = _make_file_output(tree, output_dir, "render_edge", file_format)
        fo.location = (x_offset, y_pos)
        tree.links.new(rl.outputs["Freestyle"], fo.inputs[0])

    enabled = [p for p, on in [("textured", textured), ("normal", normal),
                                ("depth", depth), ("edge", edge)] if on]
    logger.info("Multi-pass compositor configured: %s → %s", enabled, output_dir)


def clear_compositor() -> None:
    """Remove all compositor nodes, resetting to a blank graph."""
    scene = bpy.context.scene
    tree = _get_compositor_tree(scene)
    if tree is not None:
        tree.nodes.clear()
    scene.render.use_freestyle = False
    logger.debug("Compositor cleared")


def _make_file_output(
    tree: bpy.types.NodeTree,
    output_dir: str,
    filename: str,
    file_format: str,
) -> bpy.types.Node:
    """Create a CompositorNodeOutputFile configured for a single image.

    Handles the Blender 5.0 API (``directory`` / ``file_name``) and
    falls back to the older API (``base_path`` / ``file_slots``) for
    compatibility.
    """
    fo = tree.nodes.new(type="CompositorNodeOutputFile")
    fo.label = filename
    fo.format.file_format = file_format
    fo.format.color_mode = "RGB"

    if hasattr(fo, "directory"):
        fo.directory = output_dir
        fo.file_name = filename
    else:
        fo.base_path = output_dir
        fo.file_slots[0].path = filename
    return fo


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_compositor_tree(scene: bpy.types.Scene):
    """Return the compositor node tree, or None if it doesn't exist.

    Handles the Blender 5.0 API change from ``scene.node_tree`` to
    ``scene.compositing_node_group``.
    """
    if hasattr(scene, "compositing_node_group"):
        return scene.compositing_node_group
    return getattr(scene, "node_tree", None)


def _ensure_compositor_tree(scene: bpy.types.Scene):
    """Return (and create if needed) the compositor node tree.

    In Blender 5.0+ the tree is ``scene.compositing_node_group`` and
    starts as None.  In older versions it is ``scene.node_tree`` and
    is created by setting ``scene.use_nodes = True``.
    """
    if hasattr(scene, "compositing_node_group"):
        tree = scene.compositing_node_group
        if tree is None:
            tree = bpy.data.node_groups.new("Compositor", "CompositorNodeTree")
            scene.compositing_node_group = tree
        return tree
    # Blender < 5.0 path
    scene.use_nodes = True
    return scene.node_tree


def _get_or_create_node(
    nodes: bpy.types.Nodes,
    node_type: str,
    label: str,
) -> bpy.types.Node:
    """Find an existing node by label or create a new one."""
    for node in nodes:
        if node.label == label and node.bl_idname == node_type:
            return node
    node = nodes.new(type=node_type)
    node.label = label
    return node
