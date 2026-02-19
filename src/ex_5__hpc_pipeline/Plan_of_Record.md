# Plan of Record — Exercise 5: HPC Batch Processing for Human Scans

## Overview

Purpose: implement a Blender-based batch driver that processes human `.blend` files on an HPC cluster, filters out animated subjects, centers static subjects, and renders RGB + 32-bit EXR depth frames (360 frames per subject) using an NVIDIA H100 GPU. Provide a SLURM submit script to run the job on the cluster.

Repository path: `blender_excercises/src/ex_5__hpc_pipeline`

## Scope
- Implement `process_batch.py` to run inside Blender (headless) and a SLURM `submit_jobs.sh` script to request GPU resources and run the job on an H100 node.
- For each `.blend` file provided, the driver must:
  - Detect if the subject is animated; skip if animated.
  - Center static subjects to world origin.
  - Configure `RenderFlythroughHuman` (or equivalent) to render 360 frames (RGB + Depth EXR) at 640×480.

## Objectives / Success Criteria
- Correctly skip animated subjects and log skip messages.
- Produce per-model frame sequences for RGB (PNG) and depth (32-bit EXR).
- Provide a working `submit_jobs.sh` with SBATCH directives requesting 1 node and 1 H100 GPU and a 12-hour walltime.

## Requirements & Constraints
- Language: Python 3.10+ (Blender embedded Python).
- Dependencies: `bpy`, `mathutils`, stdlib (`argparse`, `logging`, `os`, `sys`).
- Cluster: SLURM scheduler with an available partition supporting H100 GPUs; Blender module available (e.g., `module load blender/4.0`).

## Implementation Approach (high level)
1. Driver (`process_batch.py`)
   - Iterate through provided `.blend` files or a directory listing.
   - For each file: `bpy.ops.wm.open_mainfile(filepath=...)` to load the blend.
   - Determine animation: implement `_is_subject_animated(obj)` which checks `object.animation_data` and inspects keyframe ranges for location/pose changes; if animated, log and skip.
   - If static: translate/center the subject to `(0,0,0)`.
   - Configure renderer: set Cycles, enable GPU via helper, set resolution 640×480, set output paths for RGB PNGs and 32-bit EXR depth maps (ensure `use_z` pass and EXR settings).
   - Create an orbit animation for the camera: 360 frames, camera orbits at fixed radius and looks at subject center.
   - Render frames (no video compilation) and save per-frame outputs.
2. SLURM script (`submit_jobs.sh`)
   - SBATCH directives: request 1 node, `--gpus=1`, account/partition as required, `--time=12:00:00`.
   - Load environment (e.g., `module load blender/4.0`) and call Blender:
     `blender --background --python process_batch.py -- --input-dir /path/to/blends --output /path/to/output`

## Deliverables
1. `process_batch.py` — Blender driver.
2. `submit_jobs.sh` — SBATCH script to request H100 and run the job.
3. `Plan_of_Record.md` — this file.

## Timeline & Milestones (suggested)
- Day 0: POR and scaffolding.
- Day 1: Implement `_is_subject_animated` and subject-centering logic; smoke test on a few `.blend` files locally.
- Day 2: Implement rendering loop, GPU enablement, and EXR depth output.
- Day 3: Finalize `submit_jobs.sh` and validate on a test GPU node.

## Testing & Validation
- Confirm `_is_subject_animated` correctly detects moving vs static subjects by checking keyframe presence and location deltas across frames.
- Render small sample (e.g., 10 frames) on a local GPU or CPU to validate orbit and depth outputs.

## Risks & Mitigations
- Risk: EXR settings and GPU drivers differ across cluster nodes. Mitigation: document Blender version and include runtime checks for Cycles/OPTIX availability; fallback to CPU with a clear log.
- Risk: Large render jobs may exceed walltime. Mitigation: allow each subject to be its own SLURM array task or chunk subjects per job.

## Implementation Notes / APIs
- To check animation: inspect `obj.animation_data` and `obj.animation_data.action` keyframe ranges, or sample translation across a short frame window.
- To render depth EXR: enable `scene.view_layers[0].use_pass_z = True` and configure `scene.render.image_settings.file_format = 'OPEN_EXR'` and `scene.render.image_settings.color_depth = '32'`.

## Next Steps
1. Confirm POR. I will scaffold `process_batch.py` and `submit_jobs.sh` next on confirmation.
