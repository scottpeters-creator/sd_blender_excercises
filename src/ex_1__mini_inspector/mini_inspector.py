#!/usr/bin/env python
"""mini_inspector.py

Pipeline-driven mini inspector. Analyzes 3D model files and produces JSON
reports with geometry counts, topology breakdown, bounding box, and materials.

Single file:
  python -m ex_1__mini_inspector.mini_inspector model.glb report.json

Batch (directory):
  python -m ex_1__mini_inspector.mini_inspector /path/to/models/ /path/to/reports/ --type glb
  python -m ex_1__mini_inspector.mini_inspector /path/to/models/ /path/to/reports/ --type glb --limit 30
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from lib.pipeline import (
    CompletedState,
    ExitCode,
    Pipeline,
    PipelineStep,
    _short_signature,
    exit_code_from,
)
from lib.work_item import WorkItem

from lib.pipeline_steps.scene import PrepareSceneStep, CleanupSceneStep
from lib.pipeline_steps.io import ImportModelStep, GrepModelsStep
from lib.pipeline_steps.analysis import CollectMeshesStep, ComputeCountsStep
from lib.pipeline_steps.reporting import WriteJSONStep

logger = logging.getLogger("mini_inspector")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Exercise-specific step (report format is Ex 1 specific)
# ---------------------------------------------------------------------------


class AssembleReportStep(PipelineStep):
    def __init__(self) -> None:
        super().__init__(
            name="assemble_report",
            requires=["counts"],
            provides=["report"],
        )

    def run(self, context: Dict[str, Any]) -> CompletedState:
        from lib.bpy.mesh import get_material_names
        counts = context.get("counts", {})
        topology = context.get("topology", {"triangles": 0, "quads": 0, "ngons": 0})
        bbox = context.get("bbox", {})
        meshes = context.get("mesh_objects", [])
        materials = get_material_names(meshes) if meshes else []
        report = {
            "filename": os.path.basename(context.get("input_path", "")),
            "geometry": counts,
            "topology": topology,
            "bounding_box": bbox,
            "materials": materials,
            "mesh_count": len(meshes),
        }
        context["report"] = report
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=["report"],
            outputs={"report_keys": list(report.keys())},
            signature=_short_signature(report),
        )


# ---------------------------------------------------------------------------
# Composed pipeline construction
# ---------------------------------------------------------------------------

scene_prep = Pipeline(
    name="scene_prep", version="1.0",
    steps=[PrepareSceneStep(), ImportModelStep()],
)

analysis = Pipeline(
    name="analysis", version="1.0",
    steps=[CollectMeshesStep(), ComputeCountsStep()],
)

reporting = Pipeline(
    name="reporting", version="1.0",
    steps=[AssembleReportStep(), WriteJSONStep()],
)

cleanup = Pipeline(
    name="cleanup", version="1.0",
    steps=[CleanupSceneStep()],
)


def build_e2e_pipeline(force: bool = False) -> Pipeline:
    """Build the single-file inspector pipeline."""
    return Pipeline(
        name="mini_inspector_e2e",
        version="1.0",
        steps=[scene_prep, analysis, reporting, cleanup],
        force=force,
    )


def build_batch_pipeline(
    extension: str = ".glb",
    limit: Optional[int] = None,
    force: bool = False,
) -> Pipeline:
    """Build the batch inspector pipeline with fan-out over a directory.

    The inner pipeline is wrapped as a single composite step so that
    fan-out runs the ENTIRE pipeline per model before moving to the next.
    """
    per_model = Pipeline(
        name="inspect_model",
        version="1.0",
        steps=[scene_prep, analysis, reporting, cleanup],
        force=force,
    )
    return Pipeline(
        name="mini_inspector_batch",
        version="1.0",
        steps=[
            GrepModelsStep(extension=extension, limit=limit),
            per_model,
        ],
        force=force,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mini_inspector",
        description="Inspect 3D model files and produce JSON geometry reports.",
    )
    parser.add_argument("input", help="Input model file or directory of models")
    parser.add_argument("output", help="Output JSON path (single file) or output directory (batch)")
    parser.add_argument("--type", dest="ext", default="glb",
                        help="File extension to scan for in batch mode (default: glb)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of models to process in batch mode")
    parser.add_argument("--force", action="store_true",
                        help="Force re-running idempotent steps")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview execution plan without running")
    args = parser.parse_args(argv)

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    is_batch = os.path.isdir(input_path)

    if is_batch:
        pipeline = build_batch_pipeline(
            extension=args.ext, limit=args.limit, force=args.force,
        )
        wi = WorkItem(
            id="batch",
            attributes={"input_dir": input_path, "output_dir": output_path},
        )
        result = pipeline.run_item(wi, dry_run=args.dry_run)
    else:
        if not args.dry_run and not os.path.exists(input_path):
            logger.error("Input not found: %s", input_path)
            return int(ExitCode.STEP_EXCEPTION)

        pipeline = build_e2e_pipeline(force=args.force)
        context: Dict[str, Any] = {
            "input_path": input_path,
            "output_path": output_path,
        }
        result = pipeline.run(context, dry_run=args.dry_run)

    if is_batch:
        meta = result.meta or {}
        logger.info(
            "Batch complete: %d items processed, success=%s",
            meta.get("steps_run", 0), result.success,
        )

    return int(exit_code_from(result))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
