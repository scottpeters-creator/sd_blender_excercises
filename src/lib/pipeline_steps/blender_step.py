"""BlenderStep — base class for pipeline steps that interact with bpy.

Adds automatic scene snapshot saving after each successful step.
Subclasses implement execute() instead of run().
"""

from __future__ import annotations

import logging
import os
from abc import abstractmethod
from typing import Any, Dict

from lib.pipeline import CompletedState, PipelineStep

logger = logging.getLogger("pipeline.blender_step")


class BlenderStep(PipelineStep):
    """Base class for steps that interact with Blender's bpy module.

    Subclasses implement ``execute()`` instead of ``run()``.  After a
    successful ``execute()``, the Blender scene is optionally saved as a
    ``.blend`` snapshot for debugging and resumability.

    Save behavior uses a three-level control hierarchy (highest priority wins):

    1. ``context["save_scenes"]`` — per-run override (``True`` / ``False``)
    2. ``self.save_scene`` — per-step setting (constructor arg)
    3. Default: ``True`` (on by default)

    Artifact location:

    1. ``context["artifacts_dir"]`` — explicit override
    2. ``{context["output_path"]}/.pipeline/`` — default
    """

    def __init__(self, name: str, save_scene: bool = True, **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        self.save_scene = save_scene

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> CompletedState:
        """Implement step logic here (not in run)."""
        raise NotImplementedError()  # pragma: no cover

    def run(self, context: Dict[str, Any]) -> CompletedState:
        result = self.execute(context)
        if result.success and self._should_save(context):
            self._save_snapshot(context)
        return result

    def _should_save(self, context: Dict[str, Any]) -> bool:
        ctx_flag = context.get("save_scenes")
        if ctx_flag is not None:
            return bool(ctx_flag)
        return self.save_scene

    def _save_snapshot(self, context: Dict[str, Any]) -> None:
        artifacts_dir = self._resolve_artifacts_dir(context)
        if not artifacts_dir:
            return
        try:
            import bpy
            os.makedirs(artifacts_dir, exist_ok=True)
            path = os.path.join(artifacts_dir, f"after_{self.name}.blend")
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
