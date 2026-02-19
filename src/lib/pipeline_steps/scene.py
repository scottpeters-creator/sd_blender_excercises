"""Scene lifecycle steps: prepare and cleanup."""

from __future__ import annotations

from typing import Any, Dict

from lib.pipeline import CompletedState
from lib.pipeline_steps.blender_step import BlenderStep


class PrepareSceneStep(BlenderStep):
    """Reset the Blender scene to an empty state."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="prepare_scene", provides=["scene_prepared"], **kwargs)

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        from lib.bpy.scene import reset_scene
        reset_scene()
        context["scene_prepared"] = True
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=["scene_prepared"],
            outputs={"note": "scene reset to empty state"},
        )


class CleanupSceneStep(BlenderStep):
    """Reset the scene after processing (free memory for batch loops)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="cleanup_scene", provides=["cleaned"], **kwargs)

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        from lib.bpy.scene import reset_scene
        reset_scene()
        context["cleaned"] = True
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=["cleaned"],
            outputs={"note": "scene cleaned"},
        )
