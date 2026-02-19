"""Reporting steps: write JSON reports and metadata files."""

from __future__ import annotations

import os
from typing import Any, Dict

from lib.naming import ensure_directory
from lib.pipeline import CompletedState, PipelineStep, _short_signature, write_json_atomic
from lib.pipeline_steps.blender_step import BlenderStep


class WriteJSONStep(PipelineStep):
    """Write context['report'] to a JSON file.

    If output_path is a directory, writes ``report.json`` inside it.
    Does NOT use bpy — stays as a plain PipelineStep.
    """

    def __init__(self) -> None:
        super().__init__(
            name="write_json",
            requires=["report"],
            provides=["output_written"],
        )

    def run(self, context: Dict[str, Any]) -> CompletedState:
        output_path = context.get("output_path")
        report = context.get("report")
        if not output_path or report is None:
            return CompletedState(
                success=False,
                timestamp=CompletedState.now_iso(),
                duration_s=0.0,
                error={"message": "missing output_path or report"},
            )
        if os.path.isdir(output_path) or not output_path.endswith(".json"):
            output_path = os.path.join(output_path, "report.json")
        ensure_directory(output_path)
        write_json_atomic(output_path, report)
        context["output_written"] = True
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=["output_written"],
            outputs={"output_path": output_path},
            signature=_short_signature(report),
        )


class WriteMetadataStep(BlenderStep):
    """Write metadata.json with camera intrinsics and bounding box."""

    def __init__(self, resolution: tuple = (512, 512), **kwargs: Any) -> None:
        super().__init__(
            name="write_metadata",
            requires=["render_complete"],
            provides=["metadata_written"],
            **kwargs,
        )
        self._resolution = resolution

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        from lib.bpy.camera import get_camera_intrinsics
        from lib.bpy.mesh import compute_world_bbox

        cam = context.get("camera_obj")
        meshes = context.get("mesh_objects", [])
        output_dir = context.get("render_output_dir", "")

        intrinsics = get_camera_intrinsics(cam) if cam else {}
        bbox = compute_world_bbox(meshes) if meshes else {}

        metadata = {
            "source_file": os.path.basename(context.get("input_path", "")),
            "render_resolution": list(self._resolution),
            "bounding_box": {
                "center": bbox.get("center", [0, 0, 0]),
                "dimensions": bbox.get("dimensions", [0, 0, 0]),
            },
            "camera": {
                "focal_length": intrinsics.get("focal_length", 50.0),
                "position": intrinsics.get("location", [0, 0, 0]),
                "look_at": [0.0, 0.0, 0.0],
            },
        }

        metadata_path = os.path.join(output_dir, "metadata.json")
        ensure_directory(metadata_path)
        write_json_atomic(metadata_path, metadata)

        context["metadata_written"] = True
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=["metadata_written"],
            outputs={"metadata_path": metadata_path},
        )
