# Plan of Record — Exercise 3: Procedural Camera Path Generation and Animation

**Status:** Not Started
**Created:** 2026-02-18

## Overview

Pipeline-driven procedural camera path generator. Loads a 3D scene, generates a camera flythrough animation using one of two modes (random spline or point-to-point interpolation), and renders a 640x480 MP4 video at 30 fps using Cycles.

Repository path: `blender_excercises/src/ex_3__proc_camera/`

## Spec Requirements (from onboarding doc)

1. **Scene Setup** — Load a .blend or model file (Evermotion indoor scenes at `s3://mod3d-west/evermotion/`)
2. **Mode A (Random Spline)** — Generate 5 random points within scene bounds, create a Bezier curve, constrain the camera to follow it smoothly over 150 frames
3. **Mode B (Point-to-Point)** — Pick 3 random positions, animate camera linearly between them with a Look-At constraint focused on (0,0,0)
4. **Rendering** — Cycles engine, 640x480, MP4 output via FFmpeg, 150 frames, 30 fps
5. **Logging** — Output control points and mode selection to console/log file

## Expected Output

```
output/
├── render_spline.mp4       (or render_point.mp4)
└── camera_log.json         (mode, control points, frame count)
```

## Recommended Pipeline Structure

```
proc_camera_e2e
├── scene_prep
│   ├── PrepareSceneStep         → scene.reset_scene() (reuse from lib/pipeline_steps)
│   └── ImportModelStep          → io.import_model() or io.open_blend() (reuse)
├── camera_animation
│   ├── GenerateCameraPathStep   → NEW: generate control points, create camera, apply mode
│   └── SetupAnimationStep       → NEW: keyframe camera path or follow-path constraint
├── render_video
│   ├── ConfigureRendererStep    → reuse from lib/pipeline_steps (adjust resolution to 640x480)
│   ├── ConfigureVideoOutputStep → NEW: render.configure_video_output() for MP4
│   └── RenderAnimationStep      → NEW: render.render_animation()
├── write_log
│   └── WriteCameraLogStep       → NEW: write control points + mode to JSON
└── cleanup
    └── CleanupSceneStep         → reuse from lib/pipeline_steps
```

## Steps to Reuse from `lib/pipeline_steps/`

| Step | Module | Notes |
|---|---|---|
| `PrepareSceneStep` | `scene.py` | Reset scene before import |
| `ImportModelStep` | `io.py` | Handles .glb/.obj/.blend |
| `CleanupSceneStep` | `scene.py` | Free memory after render |
| `ConfigureRendererStep` | `render.py` | Pass `resolution=(640, 480)` |
| `SetupEnvironmentLightStep` | `lighting.py` | Scene may need lighting for Cycles |

## New Steps to Create

| Step | What it does |
|---|---|
| `GenerateCameraPathStep` | Generate random control points within scene AABB, create camera, log points. Accepts `--mode` (spline/point) and `--seed` for reproducibility. |
| `SetupAnimationStep` | Mode A: `animation.create_bezier_path()` + `animation.constrain_to_path()`. Mode B: `animation.create_linear_keyframes()` + `camera.add_track_to_location()` targeting (0,0,0). |
| `ConfigureVideoOutputStep` | `render.configure_video_output(output_path, fps=30, codec="H264")` |
| `RenderAnimationStep` | `render.render_animation()` — renders all 150 frames to MP4 |
| `WriteCameraLogStep` | Write JSON log with mode, seed, control points, frame count |

These new steps should extend `BlenderStep` (not `PipelineStep`) and implement `execute()`. Consider adding them to `lib/pipeline_steps/` if they're reusable by Ex 5 (which also has camera orbit animation).

## Shared bpy Utilities Available

All of these exist in `lib/bpy/` and are ready to use:

| Function | Module | Purpose |
|---|---|---|
| `reset_scene()` | `scene.py` | Clear scene |
| `get_scene_bounds()` | `scene.py` | AABB for random point generation |
| `import_model()` | `io.py` | Import .glb/.obj/.fbx |
| `open_blend()` | `io.py` | Open .blend directly |
| `create_camera()` | `camera.py` | Create and position camera |
| `add_track_to_location()` | `camera.py` | Look-At constraint to a point |
| `frame_objects()` | `camera.py` | Auto-frame objects in view |
| `create_bezier_path()` | `animation.py` | Create Bezier curve from points |
| `constrain_to_path()` | `animation.py` | Follow Path constraint + keyframes |
| `create_linear_keyframes()` | `animation.py` | Linear interpolation between waypoints |
| `configure_cycles()` | `render.py` | Cycles engine setup |
| `enable_gpu()` | `render.py` | OPTIX/CUDA with macOS CPU fallback |
| `configure_video_output()` | `render.py` | FFmpeg MP4 settings |
| `render_animation()` | `render.py` | Render full timeline |

## CLI

```bash
# Mode A: random spline
python -m ex_3__proc_camera.proc_camera scene.blend /path/to/output/ --mode spline

# Mode B: point-to-point
python -m ex_3__proc_camera.proc_camera scene.blend /path/to/output/ --mode point

# With reproducible seed
python -m ex_3__proc_camera.proc_camera scene.blend /path/to/output/ --mode spline --seed 42

# Dry-run
python -m ex_3__proc_camera.proc_camera scene.blend /path/to/output/ --mode spline --dry-run
```

## Key Implementation Notes

1. **Extend `BlenderStep`, implement `execute()`** — not `PipelineStep.run()`. This gives automatic scene snapshot saving to `.pipeline/`.

2. **Use `get_scene_bounds()` for random point generation** — clamp control points to the scene AABB to avoid camera flying into void.

3. **Mode A spline path:** `create_bezier_path(points, name, resolution)` returns a curve object. Then `constrain_to_path(camera, curve, frame_count=150, frame_start=1)` adds the Follow Path constraint with keyframed offset.

4. **Mode B point-to-point:** `create_linear_keyframes(camera, positions, frame_count=150, frame_start=1)` keyframes the camera location. Then `add_track_to_location(camera, (0, 0, 0))` adds the Look-At constraint.

5. **Deferred bpy imports** — all `from lib.bpy.*` calls inside `execute()` methods, not at module top level.

6. **Output naming** — use `OutputNamer` if batch mode is needed, or simple path construction for single-scene mode.

7. **Video output requires setting frame range** — set `scene.frame_start = 1`, `scene.frame_end = 150` before calling `render_animation()`.

8. **Blender 5.0 compat** — `World.use_nodes` and `Material.use_nodes` emit deprecation warnings. The `_ensure_compositor_tree()` helper in `lib/bpy/render.py` handles compositor API changes.

## Deliverables

1. `proc_camera.py` — pipeline composition + CLI
2. `README.md` — mode switching instructions, GPU notes
3. `Plan_of_Record.md` — this file (update with implementation results)

## Input Data

- **Local:** `/Users/scott.peters/dev/source/scott_data/evermotion/AI33_001`
- **S3:** `s3://mod3d-west/evermotion/`
