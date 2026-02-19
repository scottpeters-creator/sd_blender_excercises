"""Lighting setup steps."""

from __future__ import annotations

from typing import Any, Dict

from lib.pipeline import CompletedState
from lib.pipeline_steps.blender_step import BlenderStep


class SetupEnvironmentLightStep(BlenderStep):
    """Add a uniform environment (world) light to the scene.

    Uses a simple white Background shader. Configurable strength.
    """

    def __init__(self, strength: float = 1.0, **kwargs: Any) -> None:
        super().__init__(
            name="setup_env_light",
            requires=["camera_ready"],
            provides=["lighting_ready"],
            **kwargs,
        )
        self._strength = strength

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        import bpy

        world = bpy.data.worlds.new("EnvLight")
        world.use_nodes = True
        bg = world.node_tree.nodes["Background"]
        bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        bg.inputs["Strength"].default_value = self._strength
        bpy.context.scene.world = world

        context["lighting_ready"] = True
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=["lighting_ready"],
            outputs={"env_strength": self._strength},
        )
