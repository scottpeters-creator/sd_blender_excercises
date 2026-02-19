"""I/O steps: model import and directory scanning."""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.naming import OutputNamer
from lib.pipeline import CompletedState, GeneratorStep, _short_signature
from lib.pipeline_steps.blender_step import BlenderStep
from lib.work_item import WorkItem

logger = logging.getLogger("pipeline.steps.io")


class ImportModelStep(BlenderStep):
    """Import a 3D model file into the current Blender scene."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="import_model",
            requires=["scene_prepared"],
            provides=["imported_objects"],
            idempotent=False,
            **kwargs,
        )

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        from lib.bpy.io import import_model
        input_path = context.get("input_path")
        if not input_path:
            return CompletedState(
                success=False,
                timestamp=CompletedState.now_iso(),
                duration_s=0.0,
                error={"message": "no input_path in context"},
            )
        objects = import_model(input_path)
        obj_names = [o.name for o in objects]
        context["imported_objects"] = obj_names
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=["imported_objects"],
            outputs={"imported_count": len(obj_names)},
            signature=_short_signature({"imported": obj_names}),
        )


class GrepModelsStep(GeneratorStep):
    """Scan a directory for model files and emit one WorkItem per file.

    Each emitted WorkItem has:
        input_path:  Absolute path to the model file.
        output_path: Per-model subdirectory via OutputNamer.
    """

    def __init__(
        self,
        extension: str = ".glb",
        limit: Optional[int] = None,
    ) -> None:
        super().__init__(name="grep_models", provides=[])
        self.extension = extension if extension.startswith(".") else f".{extension}"
        self.limit = limit

    def generate(self, work_item: WorkItem) -> List[WorkItem]:
        input_dir = work_item.attributes.get("input_dir", "")
        output_dir = work_item.attributes.get("output_dir", "")
        namer = OutputNamer(output_dir)

        pattern = os.path.join(input_dir, f"*{self.extension}")
        files = sorted(glob.glob(pattern))

        if self.limit is not None:
            files = files[: self.limit]

        logger.info(
            "grep_models: found %d %s file(s) in %s%s",
            len(files),
            self.extension,
            input_dir,
            f" (limited to {self.limit})" if self.limit else "",
        )

        items = []
        for filepath in files:
            stem = Path(filepath).stem
            model_dir = namer(filepath)
            items.append(
                WorkItem(
                    id=stem,
                    attributes={
                        "input_path": filepath,
                        "output_path": model_dir,
                    },
                    input_files=[filepath],
                )
            )
        return items
