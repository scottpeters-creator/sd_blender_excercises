"""Render steps: configure engine, produce images, and render animations."""

from __future__ import annotations

import os
from typing import Any, Dict

from lib.naming import ensure_directory
from lib.pipeline import CompletedState
from lib.pipeline_steps.blender_step import BlenderStep

_DEFAULT_SAMPLES = 64
_DEFAULT_RESOLUTION = (512, 512)


class ConfigureRendererStep(BlenderStep):
    """Configure Cycles engine, resolution, and GPU."""

    def __init__(
        self,
        resolution: tuple = _DEFAULT_RESOLUTION,
        samples: int = _DEFAULT_SAMPLES,
        requires: set | list | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="configure_renderer",
            requires=requires if requires is not None else ["lighting_ready"],
            provides=["renderer_ready"],
            **kwargs,
        )
        self._resolution = resolution
        self._samples = samples

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        from lib.bpy.render import configure_cycles, enable_gpu

        samples = context.get("render_samples", self._samples)
        configure_cycles(
            resolution=self._resolution,
            samples=samples,
            use_denoising=True,
            transparent_background=False,
        )
        enable_gpu()

        context["renderer_ready"] = True
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=["renderer_ready"],
        )


class RenderTexturedStep(BlenderStep):
    """Render the model with its original materials."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="render_textured",
            requires=["renderer_ready"],
            provides=["textured_rendered"],
            idempotent=False,
            **kwargs,
        )

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        from lib.bpy.render import render_frame

        output_dir = context.get("output_path", "/tmp")
        ensure_directory(output_dir + "/")
        context["render_output_dir"] = output_dir

        path = os.path.join(output_dir, "render_textured")
        render_frame(path, file_format="PNG")
        context["textured_rendered"] = True
        return CompletedState(
            success=True, timestamp=CompletedState.now_iso(),
            duration_s=0.0, provides=["textured_rendered"],
        )


class RenderNormalStep(BlenderStep):
    """Render a normal map by overriding all materials with a normal shader."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="render_normal",
            requires=["textured_rendered"],
            provides=["normal_rendered"],
            idempotent=False,
            **kwargs,
        )

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        import bpy
        from lib.bpy.render import render_frame

        output_dir = context["render_output_dir"]
        scene = bpy.context.scene

        original_materials = {}
        normal_mat = bpy.data.materials.new("_NormalOverride")
        normal_mat.use_nodes = True
        nt = normal_mat.node_tree
        nt.nodes.clear()
        geom = nt.nodes.new("ShaderNodeNewGeometry")
        emit = nt.nodes.new("ShaderNodeEmission")
        output = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(geom.outputs["Normal"], emit.inputs["Color"])
        nt.links.new(emit.outputs["Emission"], output.inputs["Surface"])

        for obj in scene.objects:
            if obj.type == "MESH":
                original_materials[obj.name] = [slot.material for slot in obj.material_slots]
                if not obj.material_slots:
                    obj.data.materials.append(normal_mat)
                else:
                    for slot in obj.material_slots:
                        slot.material = normal_mat

        path = os.path.join(output_dir, "render_normal")
        render_frame(path, file_format="PNG")

        for obj in scene.objects:
            if obj.name in original_materials:
                for i, mat in enumerate(original_materials[obj.name]):
                    if i < len(obj.material_slots):
                        obj.material_slots[i].material = mat

        bpy.data.materials.remove(normal_mat)
        context["normal_rendered"] = True
        return CompletedState(
            success=True, timestamp=CompletedState.now_iso(),
            duration_s=0.0, provides=["normal_rendered"],
        )


class RenderDepthStep(BlenderStep):
    """Render a depth map by overriding all materials with a Z-depth shader."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="render_depth",
            requires=["normal_rendered"],
            provides=["depth_rendered"],
            idempotent=False,
            **kwargs,
        )

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        import bpy
        from lib.bpy.render import render_frame

        output_dir = context["render_output_dir"]
        scene = bpy.context.scene

        cam = context.get("camera_obj")
        clip_start = cam.data.clip_start if cam else 0.1
        clip_end = cam.data.clip_end if cam else 100.0

        original_materials = {}
        depth_mat = bpy.data.materials.new("_DepthOverride")
        depth_mat.use_nodes = True
        nt = depth_mat.node_tree
        nt.nodes.clear()
        cam_data = nt.nodes.new("ShaderNodeCameraData")
        map_range = nt.nodes.new("ShaderNodeMapRange")
        map_range.inputs["From Min"].default_value = clip_start
        map_range.inputs["From Max"].default_value = clip_end
        map_range.inputs["To Min"].default_value = 1.0
        map_range.inputs["To Max"].default_value = 0.0
        emit = nt.nodes.new("ShaderNodeEmission")
        output = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(cam_data.outputs["View Z Depth"], map_range.inputs["Value"])
        nt.links.new(map_range.outputs["Result"], emit.inputs["Color"])
        nt.links.new(emit.outputs["Emission"], output.inputs["Surface"])

        for obj in scene.objects:
            if obj.type == "MESH":
                original_materials[obj.name] = [slot.material for slot in obj.material_slots]
                if not obj.material_slots:
                    obj.data.materials.append(depth_mat)
                else:
                    for slot in obj.material_slots:
                        slot.material = depth_mat

        old_bg = scene.render.film_transparent
        scene.render.film_transparent = True
        path = os.path.join(output_dir, "render_depth")
        render_frame(path, file_format="PNG")
        scene.render.film_transparent = old_bg

        for obj in scene.objects:
            if obj.name in original_materials:
                for i, mat in enumerate(original_materials[obj.name]):
                    if i < len(obj.material_slots):
                        obj.material_slots[i].material = mat

        bpy.data.materials.remove(depth_mat)
        context["depth_rendered"] = True
        return CompletedState(
            success=True, timestamp=CompletedState.now_iso(),
            duration_s=0.0, provides=["depth_rendered"],
        )


class RenderEdgeStep(BlenderStep):
    """Render edge/contour lines using Freestyle."""

    def __init__(self, line_thickness: float = 1.5, **kwargs: Any) -> None:
        super().__init__(
            name="render_edge",
            requires=["depth_rendered"],
            provides=["render_complete"],
            idempotent=False,
            **kwargs,
        )
        self._line_thickness = line_thickness

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        import bpy
        from lib.bpy.render import render_frame

        output_dir = context["render_output_dir"]
        scene = bpy.context.scene
        view_layer = scene.view_layers[0]

        scene.render.use_freestyle = True
        view_layer.use_freestyle = True
        scene.render.film_transparent = True

        if view_layer.freestyle_settings.linesets:
            lineset = view_layer.freestyle_settings.linesets[0]
        else:
            lineset = view_layer.freestyle_settings.linesets.new("LineSet")
        if lineset.linestyle is not None:
            lineset.linestyle.thickness = self._line_thickness
            lineset.linestyle.color = (0.0, 0.0, 0.0)
        else:
            ls = bpy.data.linestyles.new("EdgeStyle")
            ls.thickness = self._line_thickness
            ls.color = (0.0, 0.0, 0.0)
            lineset.linestyle = ls

        original_materials = {}
        white_mat = bpy.data.materials.new("_WhiteOverride")
        white_mat.use_nodes = True
        nt = white_mat.node_tree
        nt.nodes.clear()
        emit = nt.nodes.new("ShaderNodeEmission")
        emit.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        emit.inputs["Strength"].default_value = 1.0
        output = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(emit.outputs["Emission"], output.inputs["Surface"])

        for obj in scene.objects:
            if obj.type == "MESH":
                original_materials[obj.name] = [slot.material for slot in obj.material_slots]
                if not obj.material_slots:
                    obj.data.materials.append(white_mat)
                else:
                    for slot in obj.material_slots:
                        slot.material = white_mat

        path = os.path.join(output_dir, "render_edge")
        render_frame(path, file_format="PNG")

        scene.render.use_freestyle = False
        scene.render.film_transparent = False
        for obj in scene.objects:
            if obj.name in original_materials:
                for i, mat in enumerate(original_materials[obj.name]):
                    if i < len(obj.material_slots):
                        obj.material_slots[i].material = mat

        bpy.data.materials.remove(white_mat)
        context["render_complete"] = True
        return CompletedState(
            success=True, timestamp=CompletedState.now_iso(),
            duration_s=0.0, provides=["render_complete"],
        )


class ConfigureVideoOutputStep(BlenderStep):
    """Configure FFmpeg video output (MP4/H264)."""

    def __init__(
        self,
        fps: int = 30,
        codec: str = "H264",
        frame_start: int = 1,
        frame_end: int = 150,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="configure_video_output",
            requires=["renderer_ready"],
            provides=["video_configured"],
            **kwargs,
        )
        self._fps = fps
        self._codec = codec
        self._frame_start = frame_start
        self._frame_end = frame_end

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        import bpy
        from lib.bpy.render import configure_video_output

        output_dir = context.get("output_path", "/tmp")
        ensure_directory(output_dir + "/")
        mode = context.get("camera_mode", "spline")
        video_path = os.path.join(output_dir, f"render_{mode}.mp4")

        scene = bpy.context.scene
        frame_count = context.get("frame_count")
        scene.frame_start = self._frame_start
        scene.frame_end = frame_count if frame_count else self._frame_end

        configure_video_output(
            output_path=video_path,
            fps=self._fps,
            codec=self._codec,
        )

        context["video_configured"] = True
        context["video_path"] = video_path
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=["video_configured"],
            outputs={"video_path": video_path},
        )


class RenderAnimationStep(BlenderStep):
    """Render the full animation timeline to video."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="render_animation",
            requires=["video_configured"],
            provides=["render_path"],
            idempotent=False,
            **kwargs,
        )

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        from lib.bpy.render import render_animation

        render_animation()

        video_path = context.get("video_path", "")
        context["render_path"] = video_path
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=["render_path"],
            outputs={"render_path": video_path},
        )
