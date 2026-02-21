#!/usr/bin/env python
"""proc_camera.py

Pipeline-driven procedural camera path generator. Loads a .blend scene,
computes the SDF negative space volume, scatters camera positions along
the navigable spine, and renders a 640x480 MP4 video at 30 fps using Cycles.

v3: Granular SOP-style pipeline — 10 discrete steps, each producing
inspectable geometry in .blend snapshots. Shared steps in lib/pipeline_steps/,
exercise-specific steps here.

Usage:
  python -m ex_3__proc_camera.proc_camera scene.blend output/ --mode spline
  python -m ex_3__proc_camera.proc_camera scene.blend output/ --mode point --threshold 0.5
  python -m ex_3__proc_camera.proc_camera scene.blend output/ --mode spline --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

from lib.naming import ensure_directory
from lib.pipeline import (
    CompletedState,
    ExitCode,
    Pipeline,
    PipelineStep,
    exit_code_from,
    write_json_atomic,
)
from lib.pipeline_steps.blender_step import BlenderStep
from lib.pipeline_steps.io import OpenBlendStep
from lib.pipeline_steps.scene import NormalizeSceneStep, MergeMeshesStep, CleanupSceneStep
from lib.pipeline_steps.volume import (
    IsoMeshFromMeshStep,
    CreateBBoxMeshStep,
    ScatterPointsInMeshStep,
    PerturbPointsStep,
    CullPointsByMeshStep,
    SampleSdfStep,
    NormalizeSdfStep,
    RandomizeWeightStep,
    ColorizeBySdfStep,
    CullByWeightStep,
    OrderPointsStep,
)
from lib.pipeline_steps.render import (
    ConfigureRendererStep,
    ConfigureVideoOutputStep,
    RenderAnimationStep,
)

logger = logging.getLogger("proc_camera")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

RENDER_RESOLUTION = (640, 480)
_DEFAULT_SAMPLES = 64
_FRAME_COUNT = 150
_SPLINE_POINTS = 5
_POINT_WAYPOINTS = 3


# ---------------------------------------------------------------------------
# Exercise-specific steps (8, 9, 10)
# ---------------------------------------------------------------------------


class PointsToSplineStep(BlenderStep):
    """Select N evenly-spaced waypoints from ordered candidates and create a Bezier spline.

    The spline is visible in the .blend snapshot as a curve object.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="points_to_spline",
            requires=["ordered_candidates"],
            provides=["spline_curve", "control_points"],
            **kwargs,
        )

    def get_artifact(self, context):
        import bpy
        name = context.get("spline_curve")
        return bpy.data.objects.get(name) if name else None

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        from lib.bpy.animation import create_bezier_path

        mode = context.get("camera_mode", "spline")
        num_points = _SPLINE_POINTS if mode == "spline" else _POINT_WAYPOINTS
        candidates = context["ordered_candidates"]

        if len(candidates) <= num_points:
            points = candidates
        else:
            step = len(candidates) / num_points
            points = [candidates[int(i * step)] for i in range(num_points)]

        curve = create_bezier_path(points, name="CameraPath", resolution=64)

        logger.info("Created spline from %d waypoints (selected from %d candidates)", len(points), len(candidates))
        context["spline_curve"] = curve.name
        context["control_points"] = points
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=["spline_curve", "control_points"],
            outputs={"num_waypoints": len(points), "curve": curve.name},
        )


class CreateCameraStep(BlenderStep):
    """Create a camera at the first control point and sample surface aim targets."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="create_camera",
            requires=["control_points"],
            provides=["camera_object", "aim_targets"],
            **kwargs,
        )

    def get_artifact(self, context):
        import bpy
        name = context.get("camera_object")
        return bpy.data.objects.get(name) if name else None

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        from lib.bpy.camera import create_camera
        from lib.bpy.spatial import sample_surface_points

        mode = context.get("camera_mode", "spline")
        seed = context.get("camera_seed")
        if seed is not None:
            random.seed(seed)

        points = context["control_points"]
        num_points = len(points)

        camera = create_camera(
            name="ProcCamera",
            location=points[0],
            focal_length=35.0,
        )

        aim_targets = sample_surface_points(count=num_points)

        context["camera_object"] = camera.name
        context["aim_targets"] = aim_targets
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=["camera_object", "aim_targets"],
            outputs={"camera": camera.name, "num_aims": len(aim_targets)},
        )


class CurveAnimStep(BlenderStep):
    """Animate the camera along the spline curve or via linear keyframes.

    Mode A (spline): Constrain camera to follow the Bezier curve path.
    Mode B (point-to-point): Keyframe linear interpolation + look-at
    aimed at the centroid of surface aim targets.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="curve_anim",
            requires=["camera_object", "control_points", "aim_targets", "spline_curve"],
            provides=["animation_configured"],
            **kwargs,
        )

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        import bpy

        mode = context.get("camera_mode", "spline")
        frame_count = context.get("frame_count", _FRAME_COUNT)
        camera = bpy.data.objects[context["camera_object"]]
        points = context["control_points"]
        aim_targets = context["aim_targets"]
        aim_centroid = _centroid(aim_targets)

        if mode == "spline":
            from lib.bpy.animation import constrain_to_path
            curve = bpy.data.objects[context["spline_curve"]]
            constrain_to_path(camera, curve, frame_count=frame_count, frame_start=1)
        else:
            from lib.bpy.animation import create_linear_keyframes
            from lib.bpy.camera import add_track_to_location
            create_linear_keyframes(camera, points, frame_count=frame_count, frame_start=1)
            add_track_to_location(camera, aim_centroid)

        logger.info("Camera animated: mode=%s, frames=%d, aim=(%.1f, %.1f, %.1f)", mode, frame_count, *aim_centroid)
        context["animation_configured"] = True
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=["animation_configured"],
            outputs={"mode": mode, "frames": frame_count, "aim_centroid": list(aim_centroid)},
        )


class WriteCameraLogStep(PipelineStep):
    """Write camera path metadata to JSON."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="write_camera_log",
            requires=["control_points"],
            provides=["log_written"],
            **kwargs,
        )

    def run(self, context: Dict[str, Any]) -> CompletedState:
        output_dir = context.get("output_path", "/tmp")
        ensure_directory(output_dir + "/")
        log_path = os.path.join(output_dir, "camera_log.json")

        aim_targets = context.get("aim_targets", [])
        log_data = {
            "mode": context.get("camera_mode", "spline"),
            "seed": context.get("camera_seed"),
            "sdf_threshold": context.get("sdf_threshold"),
            "max_sdf_value": context.get("max_sdf_value"),
            "control_points": [list(p) for p in context["control_points"]],
            "aim_targets": [list(p) for p in aim_targets],
            "aim_centroid": list(_centroid(aim_targets)),
            "frames": context.get("frame_count", _FRAME_COUNT),
            "resolution": list(RENDER_RESOLUTION),
            "fps": 30,
        }

        write_json_atomic(log_path, log_data)
        logger.info("Camera log written: %s", log_path)

        context["log_written"] = True
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=["log_written"],
            outputs={"log_path": log_path},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _centroid(
    points: List[Tuple[float, float, float]],
) -> Tuple[float, float, float]:
    n = len(points)
    if n == 0:
        return (0.0, 0.0, 0.0)
    return (
        sum(p[0] for p in points) / n,
        sum(p[1] for p in points) / n,
        sum(p[2] for p in points) / n,
    )


# ---------------------------------------------------------------------------
# Pipeline composition
# ---------------------------------------------------------------------------


def build_e2e_pipeline(force: bool = False) -> Pipeline:
    """Build the 17-step procedural camera pipeline."""
    return Pipeline(
        name="proc_camera_e2e",
        version="6.1",
        steps=[
            OpenBlendStep(),                                        # 1  load scene
            NormalizeSceneStep(),                                   # 2  normalize units, transforms, modifiers
            MergeMeshesStep(),                                      # 3  join all scene meshes (copy)
            IsoMeshFromMeshStep(                                    # 3  GN scatter → iso shell
                input_key="merged_mesh",
                output_key="iso_shell",
            ),
            CreateBBoxMeshStep(                                     # 4  AABB box mesh
                output_key="bbox_mesh",
            ),
            ScatterPointsInMeshStep(                                # 5  GN vol scatter (grid at SDF voxel spacing)
                mesh_key="bbox_mesh",
                output_key="camera_candidates",
            ),
            PerturbPointsStep(                                      # 6  GN seeded jitter (break lattice bias)
                input_key="camera_candidates",
                output_key="camera_candidates_perturbed",
            ),
            CullPointsByMeshStep(                                   # 7  SDF cull against iso shell
                points_key="camera_candidates_perturbed",
                mesh_key="iso_shell",
                output_key="camera_candidates_culled",
                keep="outside",
                method="sdf",
            ),
            SampleSdfStep(                                          # 9  sample SDF → sdf_distance attr
                points_key="camera_candidates_culled",
                shell_key="iso_shell",
            ),
            NormalizeSdfStep(),                                     # 10 normalize → sdf_normalized attr
            RandomizeWeightStep(),                                  # 11 (1-norm) × noise → weight attr
            ColorizeBySdfStep(),                                    # 12 weight → color attr (debug)
            CullByWeightStep(                                       # 13 delete where weight < threshold
                output_key="camera_candidates_weighted",
                threshold=0.3,
            ),
            OrderPointsStep(                                        # 14 angular sort
                input_key="camera_candidates_weighted",
                output_key="ordered_candidates",
            ),
            PointsToSplineStep(),                                   # 9  bezier curve
            CreateCameraStep(),                                     # 10 camera + aim targets
            CurveAnimStep(),                                        # 11 animate camera
            ConfigureRendererStep(                                  # 12
                resolution=RENDER_RESOLUTION,
                samples=_DEFAULT_SAMPLES,
                requires={"animation_configured"},
            ),
            ConfigureVideoOutputStep(                               # 13
                fps=30, frame_start=1, frame_end=_FRAME_COUNT,
            ),
            RenderAnimationStep(),                                  # 14
            WriteCameraLogStep(),                                   # 15
            CleanupSceneStep(),                                     # 16
        ],
        force=force,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="proc_camera",
        description="Procedural camera flythrough: SDF negative space -> spine scatter -> render MP4.",
    )
    parser.add_argument("input", help="Input .blend scene file")
    parser.add_argument("output", help="Output directory")
    parser.add_argument(
        "--mode", choices=["spline", "point"], default="spline",
        help="Camera mode: spline (Bezier flythrough) or point (linear with look-at)",
    )
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="SDF threshold as fraction of max (0.0-1.0, default: 0.5). "
                        "Higher = closer to spine.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--frames", type=int, default=_FRAME_COUNT,
                        help=f"Number of animation frames (default: {_FRAME_COUNT})")
    parser.add_argument("--samples", type=int, default=_DEFAULT_SAMPLES,
                        help="Cycles render samples (default: 64)")
    parser.add_argument("--force", action="store_true", help="Force re-running idempotent steps")
    parser.add_argument("--dry-run", action="store_true", help="Preview execution plan without running")
    args = parser.parse_args(argv)

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    if not args.dry_run and not os.path.exists(input_path):
        logger.error("Input not found: %s", input_path)
        return int(ExitCode.STEP_EXCEPTION)

    pipeline = build_e2e_pipeline(force=args.force)
    context: Dict[str, Any] = {
        "input_path": input_path,
        "output_path": output_path,
        "camera_mode": args.mode,
        "camera_seed": args.seed,
        "sdf_threshold": args.threshold,
        "frame_count": args.frames,
        "render_samples": args.samples,
    }
    result = pipeline.run(context, dry_run=args.dry_run)
    return int(exit_code_from(result))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
