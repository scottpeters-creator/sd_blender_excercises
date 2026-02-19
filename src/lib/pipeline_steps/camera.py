"""Camera setup steps."""

from __future__ import annotations

from typing import Any, Dict

from lib.pipeline import CompletedState
from lib.pipeline_steps.blender_step import BlenderStep


class SetupCameraStep(BlenderStep):
    """Create a camera and frame it on the imported objects."""

    def __init__(
        self,
        focal_length: float = 50.0,
        margin: float = 1.4,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="setup_camera",
            requires=["imported_objects"],
            provides=["camera_ready"],
            **kwargs,
        )
        self._focal_length = focal_length
        self._margin = margin

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        from lib.bpy.camera import create_camera, frame_objects
        from lib.bpy.mesh import collect_meshes

        meshes = collect_meshes()
        if not meshes:
            return CompletedState(
                success=False,
                timestamp=CompletedState.now_iso(),
                duration_s=0.0,
                error={"message": "no mesh objects in scene to frame"},
            )

        cam = create_camera(
            name="ThumbnailCam",
            location=(0, -3, 1.5),
            focal_length=self._focal_length,
        )
        frame_objects(cam, meshes, margin=self._margin)

        context["camera_obj"] = cam
        context["mesh_objects"] = meshes
        context["camera_ready"] = True
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=["camera_ready"],
            outputs={"meshes_framed": len(meshes)},
        )
