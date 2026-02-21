# Blender Exercises Pipeline

A composable, PDG-inspired pipeline framework for automating 3D data engineering tasks with Blender. Built as a series of onboarding exercises that progressively introduce headless model analysis, multi-modality rendering, procedural camera animation, batch processing with S3, and HPC cluster submission.

## What

Five exercises that build on a shared pipeline framework:

| Exercise | Task | Status |
|---|---|---|
| **Ex 1** -- Mini Inspector | Import a .glb model, extract geometry counts, topology, bounding box, materials. Output a JSON report. | Complete |
| **Ex 2** -- Thumbnail Renderer | Render 4 modalities (textured, normal, depth, edge) at 512x512 per model, plus metadata.json. | Functional |
| **Ex 3** -- Procedural Camera | Generate camera flythrough animations (spline and point-to-point modes), render to MP4. | In progress (v6) |
| **Ex 4** -- Batch Validator | Scan S3 for .glb files, process thousands with memory management, upload JSON reports back to S3. | Not started |
| **Ex 5** -- HPC Pipeline | Filter animated subjects, normalize to origin, render 360-frame orbits (RGB + Depth) on SLURM with H100 GPUs. | Not started |

Each exercise produces a thin script (~200 lines) that composes reusable pipeline steps -- the framework handles orchestration, logging, state management, and artifact tracking.

## Why

Real 3D data engineering pipelines process thousands of assets through multi-step workflows. This project solves three problems:

**Composability.** Steps are independent units with declared inputs (`requires`) and outputs (`provides`). Pipelines are built by composing steps, and pipelines can nest inside other pipelines (Composite pattern). Exercise 1's scene preparation pipeline is reused directly inside Exercise 2.

**Batch processing at scale.** A `GeneratorStep` scans a directory and fans out into one `WorkItem` per model. Each model flows through the complete pipeline independently. The framework handles per-model output directories, idempotent re-runs, and error resilience.

**Execution isolation.** Blender's `bpy` module is a global singleton -- one scene, one process. The framework provides four scheduler backends to control where steps execute:

- `LocalScheduler` -- in-process (fast, shared state)
- `PoolScheduler` -- persistent worker pool with process isolation
- `SubprocessScheduler` -- fresh process per work item
- `SLURMScheduler` -- HPC batch job submission via submitit

## How

### Prerequisites

- Python 3.11+
- [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main)
- Blender 5.0+ (as a standalone app or the `bpy` pip package)

### Setup

```bash
# Create and activate the conda environment
conda create -n blender_exercises python=3.11
conda activate blender_exercises

# Install bpy (match your Python version)
pip install bpy

# Install the project in editable mode
cd blender_exercises
pip install -e .
```

### Run Exercise 1 -- Mini Inspector

```bash
# Single file
python -m ex_1__mini_inspector.mini_inspector model.glb report.json

# Batch -- all .glb files in a directory
python -m ex_1__mini_inspector.mini_inspector /path/to/models/ /path/to/reports/ --type glb

# First 30 only
python -m ex_1__mini_inspector.mini_inspector /path/to/models/ /path/to/reports/ --type glb --limit 30

# Preview execution plan without running
python -m ex_1__mini_inspector.mini_inspector /path/to/models/ /path/to/reports/ --type glb --dry-run
```

### Run Exercise 2 -- Thumbnail Renderer

```bash
# Single file (renders 4 modalities + metadata.json)
python -m ex_2__thumbnail_renderer.render_task model.glb /path/to/output/

# Batch with custom render samples
python -m ex_2__thumbnail_renderer.render_task /path/to/models/ /path/to/renders/ --type glb --limit 10 --samples 64
```

### Run Exercise 3 -- Procedural Camera

```bash
# Spline flythrough with seeded jitter (deterministic)
python -m ex_3__proc_camera.proc_camera scene.blend /path/to/output/ --mode spline --seed 42 --frames 150

# Point-to-point with look-at
python -m ex_3__proc_camera.proc_camera scene.blend /path/to/output/ --mode point

# Quick test (10 frames, low samples)
python -m ex_3__proc_camera.proc_camera scene.blend /path/to/output/ --mode spline --seed 42 --frames 10 --samples 16
```

### Run Tests

```bash
# 96 unit tests (no Blender required)
python -m unittest lib.test_pipeline -v
```

## Architecture

```
PipelineStep (ABC)
    ├── Pipeline           composable orchestrator (nestable)
    ├── StepGroup          parallel/sequential groups
    ├── GeneratorStep      fan-out: 1 WorkItem -> N WorkItems
    ├── CollectorStep      fan-in: N WorkItems -> 1 WorkItem
    └── BlenderStep        bpy-specific base (artifacts, screenshots, snapshots)
          ├── NormalizeSceneStep       (unit scale, transforms, modifiers, normals)
          ├── PrepareSceneStep / MergeMeshesStep / CleanupSceneStep
          ├── ImportModelStep / OpenBlendStep
          ├── IsoMeshFromMeshStep / ScatterPointsInMeshStep / PerturbPointsStep
          ├── CullPointsByMeshStep / OrderPointsStep
          ├── SetupCameraStep / ConfigureRendererStep
          ├── RenderTexturedStep / NormalStep / DepthStep / EdgeStep
          └── ...20+ reusable steps total
```

The orchestrator runs each step through a composable **middleware chain** (idempotent skip, requirements check, logging, timing, validation, state write). Custom middleware (retry, profiling) can be injected without modifying step code.

Exercise scripts are pure composition:

```python
from lib.pipeline import Pipeline
from lib.pipeline_steps.scene import PrepareSceneStep, CleanupSceneStep
from lib.pipeline_steps.io import ImportModelStep, GrepModelsStep
from lib.pipeline_steps.camera import SetupCameraStep
from lib.pipeline_steps.render import ConfigureRendererStep, RenderTexturedStep

pipeline = Pipeline(
    name="my_pipeline",
    steps=[
        GrepModelsStep(extension=".glb", limit=10),
        Pipeline(name="per_model", steps=[
            PrepareSceneStep(),
            ImportModelStep(),
            SetupCameraStep(),
            ConfigureRendererStep(),
            RenderTexturedStep(),
            CleanupSceneStep(),
        ]),
    ],
)
```

### Pipeline Artifacts, Screenshots, and Snapshots

Every `BlenderStep` subclass runs three post-execute hooks after each successful step:

1. **Artifact generation** — creates a numbered, named object in the Blender outliner so each step's output is independently visible and inspectable (not buried in a modifier stack).

2. **Screenshot capture** — frames the artifact with a debug camera (auto clip planes from bounding sphere), renders a 960x540 PNG via Eevee. Works headless. Provides a visual record without opening `.blend` files.

3. **Scene snapshot** — saves the full `.blend` scene for interactive debugging.

All outputs use zero-padded step numbering for correct sort order:

```
output/
├── render_spline.mp4       <-- deliverables
├── camera_log.json
└── .pipeline/              <-- auto-saved per-step artifacts
    ├── 001_open_blend.blend
    ├── 002_normalize_scene.blend
    ├── 003_merge_meshes.blend
    ├── 003_merge_meshes.png     ← screenshot (step has geometry)
    ├── 004_iso_mesh_from_mesh.blend
    ├── 004_iso_mesh_from_mesh.png
    ├── ...
```

Steps that produce geometry override `get_artifact(context)` to return their output object. Steps without inspectable geometry (e.g., `ConfigureRendererStep`) return `None` — they get a `.blend` snapshot but no artifact or screenshot.

All three hooks are on by default and controllable at per-run and per-step levels:

```python
# Disable screenshots for a production batch run
context["save_screenshots"] = False

# Disable all artifacts for a specific step
ConfigureRendererStep(save_artifact=False, save_screenshot=False)

# Change screenshot engine (default: BLENDER_EEVEE)
context["screenshot_engine"] = "CYCLES"
```

## Project Structure

```
blender_exercises/
├── pyproject.toml              package definition
├── context.md                  detailed framework documentation
├── README.md                   this file
├── dev_tasks/                  pipeline framework design docs (PORs)
├── source/                     original exercise specifications
└── src/
    ├── lib/
    │   ├── pipeline.py         orchestrator, middleware, DAG solver
    │   ├── work_item.py        WorkItem dataclass
    │   ├── scheduler.py        Local/Pool/Subprocess/SLURM schedulers
    │   ├── naming.py           OutputNamer, ensure_directory
    │   ├── test_pipeline.py    96 unit tests
    │   ├── pipeline_steps/     20+ reusable step classes
    │   └── bpy/                Blender utility functions (GN builders, camera, render, etc.)
    ├── ex_1__mini_inspector/   Exercise 1 (complete)
    ├── ex_2__thumbnail_renderer/  Exercise 2 (functional)
    ├── ex_3__proc_camera/      Exercise 3 (v6 — in progress)
    ├── ex_4__batch_validator/  Exercise 4 (not started)
    └── ex_5__hpc_pipeline/     Exercise 5 (not started)
```

## Documentation

- `context.md` -- comprehensive framework reference, how-to guide, API docs, and lessons learned. **Read this first if you're implementing an exercise.**
- Each exercise has a `Plan_of_Record.md` with implementation details and status.
- `dev_tasks/001_pipeline_dev/POR.md` -- original pipeline redesign plan.
- `dev_tasks/002_pdg_refactor/POR.md` -- PDG-Lite evolution plan.

## Key Design Decisions

1. **Blender-agnostic orchestration.** The pipeline library never imports `bpy`. All Blender logic lives in step implementations. Tests run without Blender.

2. **Depth-first execution for bpy batch processing.** Fan-out wraps the entire per-model pipeline as a single composite step, ensuring all steps complete for one model before starting the next. This prevents bpy's shared global scene from being corrupted between models.

3. **Deferred bpy imports.** All `from lib.bpy.*` imports are inside step `execute()` methods, not at module top level. This allows pipeline composition and testing without bpy.

4. **Blender 5.0 compatibility.** The framework handles Blender 5.0 API changes (`scene.compositing_node_group` instead of `scene.node_tree`, `Material.use_nodes` deprecation, missing `CompositorNodeComposite`, `CompositorNodeOutputFile.directory` instead of `base_path`).

5. **Scene normalization.** `NormalizeSceneStep` runs early in the pipeline to fix common Blender data model issues: sets `scale_length=1.0` so code values match UI display, makes multi-user meshes single-user, applies unapplied transforms and modifiers, fixes negative-scale normals. This ensures all downstream steps operate on clean, predictable geometry.

6. **GN-native spatial operations.** All geometry operations (iso mesh, volume scatter, SDF cull, seeded jitter) run as Geometry Nodes evaluations, not Python loops. GN trees are auto-laid-out via the `@auto_layout` decorator for immediate readability when inspecting `.blend` snapshots.

7. **Uniform introspection.** Every pipeline step produces both a live GN modifier (for interactive parameter tweaking) and a separate numbered artifact object in the outliner (for visual inspection and headless screenshots). This mirrors Houdini's "inspect any node" workflow within Blender's more restrictive architecture.
