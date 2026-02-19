# Plan of Record — Exercise 3: Procedural Camera Path Generation and Animation

## Overview

Purpose: implement a headless Blender script that loads a 3D scene/model and produces a 150-frame camera flythrough video (640×480, 30 fps) using two camera path modes: (A) random spline and (B) point-to-point interpolation with a Look-At constraint. Output must be an MP4 rendered with Cycles and a console log of generated control points.

Repository path: `blender_excercises/src/ex_3__proc_camera`

## Scope
- Implement `proc_camera.py` runnable in Blender background mode:
  - Example: `blender --background --python proc_camera.py -- --mode spline --input scene.blend --output out.mp4`
- Support two camera modes: `spline` and `point`.
- Produce an MP4 file (150 frames) and a small log text file capturing generated control points and selected mode.

## Objectives / Success Criteria
- Script accepts `--mode` and `--output` arguments and prints generated control points.
- Rendered MP4 is 640×480 at 30 fps with smooth camera motion and correct look-at behavior for both modes.
- Script uses Cycles and includes GPU enablement logic (with macOS fallback to CPU) for cluster runs.

## Requirements & Constraints
- Language: Python 3.10+ (Blender embedded Python when run via `blender`).
- Dependencies: `bpy`, `mathutils`, `random`, stdlib (`argparse`, `logging`, `json`, `os`, `sys`).
- Execution: headless Blender (`--background`).

## Implementation Approach (high level)
1. Scene setup
   - Load scene or `.blend` file; if an input model is provided (GLTF/OBJ), import and set up scene.
2. Two camera generation modes
   - Mode A (Random Spline): generate N=5 random points within the scene AABB; build a Curve object, create a camera, add a Follow Path constraint and animate along the curve across 150 frames; orient the camera to look slightly ahead or at a small moving target.
   - Mode B (Point-to-Point): sample 3 random points; animate the camera's location keyframes between them over 150 frames; add a Track To/Locked Track constraint to keep the camera looking at `(0,0,0)`.
3. Rendering
   - Configure Cycles renderer, set resolution to 640×480 and 30 fps; configure FFmpeg output to MP4; ensure device selection logic enables GPU (OPTIX/CUDA) when available.
4. Logging
   - Emit a small JSON or plaintext log alongside output outlining the mode and control points used.

## Deliverables
1. `proc_camera.py` — main script.
2. `README.md` — usage examples and how to enable GPU renders on the cluster.
3. `Plan_of_Record.md` — this file.

## Timeline & Milestones (suggested)
- Day 0: Create POR and scaffolding.
- Day 1: Implement camera path generation and basic animation keyframes; render a quick smoke test (e.g., 10 frames) locally.
- Day 2: Integrate Cycles settings, FFmpeg MP4 output, and GPU enablement.
- Day 3: Run full 150-frame render, capture logs, and validate output video.

## Testing & Validation
- Generate a deterministic seed for randomness for reproducible tests.
- Verify generated log contains the same control points used for rendering.
- Play the MP4 to confirm smooth motion and correct look-at constraints.

## Risks & Mitigations
- Risk: Cluster GPUs may require enabling specific Cycles compute backends. Mitigation: include `_enable_gpu` helper and fallback to CPU; document required Blender module/version.
- Risk: Very large scenes may place control points outside usable bounds. Mitigation: clamp random points to scene AABB and provide configurable margins.

## Implementation Notes / APIs
- Use `bpy.data.curves.new(..., type='CURVE')` and `bpy.data.objects.new(..., curve)` to create spline paths.
- Use `camera.constraints.new('FOLLOW_PATH')` and animate the `eval_time` or use keyframes on `offset_factor`/`offset` as appropriate.
- For Look-At: use `Track To` or `Locked Track` constraint targeting an empty at the center.

## Next Steps
1. Confirm POR. I will scaffold `proc_camera.py` and a `README.md` next, then run a short smoke render if you want.
