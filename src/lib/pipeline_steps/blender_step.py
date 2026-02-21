"""BlenderStep — base class for pipeline steps that interact with bpy.

Adds automatic scene snapshot saving, artifact generation, and screenshot
capture after each successful step. Subclasses implement execute() instead
of run().

Post-execute hooks (all on by default, all skippable):

1. ``_generate_artifact`` — creates a numbered, named object in the
   ``_proc_cam`` collection from the step's output geometry. This makes
   every step's output visible as a distinct object in the Blender
   outliner, independent of the modifier stack.

2. ``_take_screenshot`` — frames the artifact with a debug camera
   (auto clip planes from bounding sphere), renders via Eevee to PNG.
   Works in headless/background mode. Provides a visual record of each
   step's output without opening the .blend file.

3. ``_save_snapshot`` — saves the full ``.blend`` scene.

Subclasses override ``get_artifact(context)`` to return the Blender object
representing their output. Steps that don't produce inspectable geometry
return ``None`` (the default) and hooks 1-2 are skipped.

Naming convention:
    All outputs use zero-padded step index: ``{NNN}_{step_name}``.
    The index comes from ``context["_step_index"]``, set by the Pipeline
    orchestrator as it iterates steps.

    Example outputs for a 17-step pipeline::

        .pipeline/
        ├── 001_open_blend.blend
        ├── 002_normalize_scene.blend
        ├── 003_merge_meshes.blend
        ├── 003_merge_meshes.png        ← screenshot (has get_artifact)
        ├── 004_iso_mesh_from_mesh.blend
        ├── 004_iso_mesh_from_mesh.png
        ├── ...

Control hierarchy (highest priority wins):

    Artifacts:
        1. ``context["save_artifacts"]`` — per-run override
        2. ``self.save_artifact`` — per-step (constructor arg)
        3. Default: True

    Screenshots:
        1. ``context["save_screenshots"]`` — per-run override
        2. ``self.save_screenshot`` — per-step (constructor arg)
        3. Default: True
        4. ``context["screenshot_engine"]`` — render engine
           (default: ``"BLENDER_EEVEE"``, options: ``"CYCLES"``,
           ``"BLENDER_WORKBENCH"``)

    Scene snapshots:
        1. ``context["save_scenes"]`` — per-run override
        2. ``self.save_scene`` — per-step (constructor arg)
        3. Default: True

Implementing get_artifact in a subclass:

    Steps that produce geometry override ``get_artifact()`` to return
    the Blender object representing their output::

        class MyStep(BlenderStep):
            def get_artifact(self, context):
                import bpy
                name = context.get("my_output_key")
                return bpy.data.objects.get(name) if name else None

    The artifact object is used for:
    - Creating a numbered evaluated copy in the outliner
    - Framing and rendering the screenshot

    Steps that modify scene state without producing a single
    inspectable object (e.g., ConfigureRendererStep, CurveAnimStep)
    leave ``get_artifact()`` returning None — they still get a .blend
    snapshot but no artifact or screenshot.
"""

from __future__ import annotations

import logging
import os
from abc import abstractmethod
from typing import Any, Dict, Optional

from lib.pipeline import CompletedState, PipelineStep

logger = logging.getLogger("pipeline.blender_step")

_ARTIFACT_PREFIX = "_proc_cam_"
_DEBUG_CAM_NAME = "_proc_cam_debug_camera"


class BlenderStep(PipelineStep):
    """Base class for steps that interact with Blender's bpy module.

    Subclasses implement ``execute()`` instead of ``run()``.  After a
    successful ``execute()``, artifacts, screenshots, and scene snapshots
    are saved according to the control hierarchy.

    Constructor args (all optional, all default True):
        save_scene: Save a .blend snapshot after this step.
        save_artifact: Create a numbered artifact object in the outliner.
        save_screenshot: Capture a viewport screenshot to PNG.
    """

    def __init__(
        self,
        name: str,
        save_scene: bool = True,
        save_artifact: bool = True,
        save_screenshot: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.save_scene = save_scene
        self.save_artifact = save_artifact
        self.save_screenshot = save_screenshot

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> CompletedState:
        """Implement step logic here (not in run)."""
        raise NotImplementedError()  # pragma: no cover

    def get_artifact(self, context: Dict[str, Any]) -> Optional[Any]:
        """Return the Blender object representing this step's output.

        Subclasses override this to return the object that should be
        registered as a numbered artifact and used for screenshot framing.
        Return None (default) if the step has no inspectable geometry.
        """
        return None

    def run(self, context: Dict[str, Any]) -> CompletedState:
        result = self.execute(context)
        if result.success:
            step_label = self._step_label(context)
            if self._should_generate_artifact(context):
                self._generate_artifact(context, step_label)
            if self._should_take_screenshot(context):
                self._take_screenshot(context, step_label)
            if self._should_save(context):
                self._save_snapshot(context, step_label)
        return result

    # ------------------------------------------------------------------
    # Step label (NNN_name)
    # ------------------------------------------------------------------

    def _step_label(self, context: Dict[str, Any]) -> str:
        idx = context.get("_step_index", 0)
        return f"{idx:03d}_{self.name}"

    # ------------------------------------------------------------------
    # Artifact generation
    # ------------------------------------------------------------------

    def _should_generate_artifact(self, context: Dict[str, Any]) -> bool:
        ctx_flag = context.get("save_artifacts")
        if ctx_flag is not None:
            return bool(ctx_flag)
        return self.save_artifact

    def _generate_artifact(self, context: Dict[str, Any], step_label: str) -> None:
        artifact = self.get_artifact(context)
        if artifact is None:
            return
        try:
            import bpy
            from lib.bpy.geometry_nodes import get_temp_collection

            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_obj = artifact.evaluated_get(depsgraph)

            new_mesh = bpy.data.meshes.new(f"{step_label}_mesh")
            if hasattr(eval_obj.data, "vertices"):
                new_mesh.from_pydata(
                    [v.co.copy() for v in eval_obj.data.vertices],
                    [],
                    [list(p.vertices) for p in eval_obj.data.polygons],
                )
                new_mesh.update()

            artifact_obj = bpy.data.objects.new(step_label, new_mesh)
            col = get_temp_collection()
            col.objects.link(artifact_obj)

            logger.debug(
                "Artifact '%s': %d verts, %d faces",
                step_label, len(new_mesh.vertices), len(new_mesh.polygons),
            )
        except Exception as exc:
            logger.warning("Failed to generate artifact for '%s': %s", self.name, exc)

    # ------------------------------------------------------------------
    # Screenshot capture
    # ------------------------------------------------------------------

    def _should_take_screenshot(self, context: Dict[str, Any]) -> bool:
        ctx_flag = context.get("save_screenshots")
        if ctx_flag is not None:
            return bool(ctx_flag)
        return self.save_screenshot

    def _take_screenshot(self, context: Dict[str, Any], step_label: str) -> None:
        """Render a screenshot of the artifact framed by a debug camera.

        Uses Eevee by default (1 sample for speed). The render engine
        can be overridden via ``context["screenshot_engine"]``.

        The debug camera is automatically positioned to frame the
        artifact's bounding sphere with clip planes scaled to match.
        """
        artifact = self.get_artifact(context)
        if artifact is None:
            return
        artifacts_dir = self._resolve_artifacts_dir(context)
        if not artifacts_dir:
            return
        try:
            import bpy
            import math
            from mathutils import Vector

            cam = self._get_debug_camera()
            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_obj = artifact.evaluated_get(depsgraph)

            if not hasattr(eval_obj.data, "vertices") or len(eval_obj.data.vertices) == 0:
                return

            coords = [eval_obj.matrix_world @ Vector(v.co) for v in eval_obj.data.vertices]
            center = sum(coords, Vector()) / len(coords)
            radius = max((v - center).length for v in coords)

            if radius < 0.001:
                return

            cam.data.clip_start = radius * 0.01
            cam.data.clip_end = radius * 10.0

            fov = 2.0 * math.atan(cam.data.sensor_width / (2.0 * cam.data.lens))
            distance = (radius * 1.3) / math.tan(fov / 2.0)

            direction = Vector((0.6, -0.8, 0.5)).normalized()
            cam.location = center + direction * distance
            rot_quat = (center - cam.location).to_track_quat("-Z", "Y")
            cam.rotation_euler = rot_quat.to_euler()

            bpy.context.scene.camera = cam

            os.makedirs(artifacts_dir, exist_ok=True)
            filepath = os.path.join(artifacts_dir, f"{step_label}.png")

            old_engine = bpy.context.scene.render.engine
            old_res_x = bpy.context.scene.render.resolution_x
            old_res_y = bpy.context.scene.render.resolution_y
            old_pct = bpy.context.scene.render.resolution_percentage
            old_path = bpy.context.scene.render.filepath
            old_samples = getattr(bpy.context.scene.eevee, "taa_render_samples", 16)

            engine = context.get("screenshot_engine", "BLENDER_EEVEE")
            bpy.context.scene.render.engine = engine
            bpy.context.scene.render.resolution_x = 960
            bpy.context.scene.render.resolution_y = 540
            bpy.context.scene.render.resolution_percentage = 100
            bpy.context.scene.render.filepath = filepath
            if hasattr(bpy.context.scene.eevee, "taa_render_samples"):
                bpy.context.scene.eevee.taa_render_samples = 1
            bpy.ops.render.render(write_still=True)

            bpy.context.scene.render.engine = old_engine
            bpy.context.scene.render.resolution_x = old_res_x
            bpy.context.scene.render.resolution_y = old_res_y
            bpy.context.scene.render.resolution_percentage = old_pct
            bpy.context.scene.render.filepath = old_path
            if hasattr(bpy.context.scene.eevee, "taa_render_samples"):
                bpy.context.scene.eevee.taa_render_samples = old_samples

            logger.debug("Screenshot saved: %s", filepath)
        except Exception as exc:
            logger.warning("Failed to take screenshot for '%s': %s", self.name, exc)

    def _get_debug_camera(self):
        """Get or create a debug camera in the temp collection."""
        import bpy
        from lib.bpy.geometry_nodes import get_temp_collection

        if _DEBUG_CAM_NAME in bpy.data.objects:
            return bpy.data.objects[_DEBUG_CAM_NAME]

        cam_data = bpy.data.cameras.new(_DEBUG_CAM_NAME)
        cam_data.lens = 35.0
        cam_obj = bpy.data.objects.new(_DEBUG_CAM_NAME, cam_data)
        col = get_temp_collection()
        col.objects.link(cam_obj)
        return cam_obj

    # ------------------------------------------------------------------
    # Scene snapshot
    # ------------------------------------------------------------------

    def _should_save(self, context: Dict[str, Any]) -> bool:
        ctx_flag = context.get("save_scenes")
        if ctx_flag is not None:
            return bool(ctx_flag)
        return self.save_scene

    def _save_snapshot(self, context: Dict[str, Any], step_label: str) -> None:
        artifacts_dir = self._resolve_artifacts_dir(context)
        if not artifacts_dir:
            return
        try:
            import bpy
            os.makedirs(artifacts_dir, exist_ok=True)
            path = os.path.join(artifacts_dir, f"{step_label}.blend")
            bpy.ops.wm.save_as_mainfile(filepath=path, copy=True)
            logger.debug("Saved scene snapshot: %s", path)
        except Exception as exc:
            logger.warning("Failed to save scene snapshot for '%s': %s", self.name, exc)

    def _resolve_artifacts_dir(self, context: Dict[str, Any]) -> str:
        if context.get("artifacts_dir"):
            return context["artifacts_dir"]
        output_path = context.get("output_path", "")
        if output_path:
            return os.path.join(output_path, ".pipeline")
        return ""
