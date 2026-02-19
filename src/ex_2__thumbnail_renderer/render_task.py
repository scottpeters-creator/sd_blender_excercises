#!/usr/bin/env python
"""render_task.py

Pipeline-driven thumbnail renderer. Imports a 3D model and produces four
512x512 PNG images (textured, normal, depth, edge) plus metadata.json.

Single file:
  python -m ex_2__thumbnail_renderer.render_task model.glb output_dir/

Batch (directory):
  python -m ex_2__thumbnail_renderer.render_task /path/to/models/ /path/to/renders/ --type glb
  python -m ex_2__thumbnail_renderer.render_task /path/to/models/ /path/to/renders/ --type glb --limit 30
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from lib.naming import OutputNamer
from lib.pipeline import (
    ExitCode,
    Pipeline,
    exit_code_from,
)
from lib.work_item import WorkItem

from lib.pipeline_steps.scene import PrepareSceneStep, CleanupSceneStep
from lib.pipeline_steps.io import ImportModelStep, GrepModelsStep
from lib.pipeline_steps.camera import SetupCameraStep
from lib.pipeline_steps.lighting import SetupEnvironmentLightStep
from lib.pipeline_steps.render import (
    ConfigureRendererStep,
    RenderTexturedStep,
    RenderNormalStep,
    RenderDepthStep,
    RenderEdgeStep,
)
from lib.pipeline_steps.reporting import WriteMetadataStep

logger = logging.getLogger("render_task")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

RENDER_RESOLUTION = (512, 512)
_DEFAULT_SAMPLES = 64


# ---------------------------------------------------------------------------
# Composed pipeline construction
# ---------------------------------------------------------------------------

scene_prep = Pipeline(
    name="scene_prep", version="1.0",
    steps=[PrepareSceneStep(), ImportModelStep()],
)

camera_setup = Pipeline(
    name="camera_setup", version="1.0",
    steps=[SetupCameraStep()],
)

lighting = Pipeline(
    name="lighting", version="1.0",
    steps=[SetupEnvironmentLightStep(strength=1.0)],
)

render = Pipeline(
    name="render", version="1.0",
    steps=[
        ConfigureRendererStep(resolution=RENDER_RESOLUTION, samples=_DEFAULT_SAMPLES),
        RenderTexturedStep(),
        RenderNormalStep(),
        RenderDepthStep(),
        RenderEdgeStep(),
    ],
)

metadata = Pipeline(
    name="metadata", version="1.0",
    steps=[WriteMetadataStep(resolution=RENDER_RESOLUTION)],
)

cleanup = Pipeline(
    name="cleanup", version="1.0",
    steps=[CleanupSceneStep()],
)


def build_e2e_pipeline(force: bool = False) -> Pipeline:
    """Build the single-file thumbnail render pipeline."""
    return Pipeline(
        name="thumbnail_renderer_e2e",
        version="1.0",
        steps=[scene_prep, camera_setup, lighting, render, metadata, cleanup],
        force=force,
    )


def build_batch_pipeline(
    extension: str = ".glb",
    limit: Optional[int] = None,
    force: bool = False,
) -> Pipeline:
    """Build the batch thumbnail render pipeline with fan-out."""
    per_model = Pipeline(
        name="render_model",
        version="1.0",
        steps=[scene_prep, camera_setup, lighting, render, metadata, cleanup],
        force=force,
    )
    return Pipeline(
        name="thumbnail_renderer_batch",
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
        prog="render_task",
        description="Render 3D model thumbnails in 4 modalities (textured, normal, depth, edge).",
    )
    parser.add_argument("input", help="Input model file or directory of models")
    parser.add_argument("output", help="Output directory (single file) or root output dir (batch)")
    parser.add_argument("--type", dest="ext", default="glb",
                        help="File extension for batch mode (default: glb)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max models to process in batch mode")
    parser.add_argument("--samples", type=int, default=_DEFAULT_SAMPLES,
                        help="Cycles render samples (default: 64)")
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
            attributes={
                "input_dir": input_path,
                "output_dir": output_path,
                "render_samples": args.samples,
            },
        )
        result = pipeline.run_item(wi, dry_run=args.dry_run)
    else:
        if not args.dry_run and not os.path.exists(input_path):
            logger.error("Input not found: %s", input_path)
            return int(ExitCode.STEP_EXCEPTION)

        namer = OutputNamer(output_path)
        model_dir = namer(input_path)

        pipeline = build_e2e_pipeline(force=args.force)
        context: Dict[str, Any] = {
            "input_path": input_path,
            "output_path": model_dir,
            "render_samples": args.samples,
        }
        result = pipeline.run(context, dry_run=args.dry_run)

    if is_batch:
        logger.info("Batch complete: success=%s", result.success)

    return int(exit_code_from(result))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
