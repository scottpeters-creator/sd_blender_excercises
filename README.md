# Blender Exercises Pipeline

A composable, PDG-inspired pipeline framework for automating 3D data engineering tasks with Blender. Built as a series of onboarding exercises that progressively introduce headless model analysis, multi-modality rendering, procedural camera animation, batch processing with S3, and HPC cluster submission.

## What

Five exercises that build on a shared pipeline framework:

| Exercise | Task | Status |
|---|---|---|
| **Ex 1** -- Mini Inspector | Import a .glb model, extract geometry counts, topology, bounding box, materials. Output a JSON report. | Complete |
| **Ex 2** -- Thumbnail Renderer | Render 4 modalities (textured, normal, depth, edge) at 512x512 per model, plus metadata.json. | Functional |
| **Ex 3** -- Procedural Camera | Generate camera flythrough animations (spline and point-to-point modes), render to MP4. | Not started |
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
conda create -n blender_excercises python=3.11
conda activate blender_excercises

# Install bpy (match your Python version)
pip install bpy

# Install the project in editable mode
cd blender_excercises
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
    └── BlenderStep        bpy-specific base (auto scene snapshots)
          ├── PrepareSceneStep
          ├── ImportModelStep
          ├── SetupCameraStep
          ├── ConfigureRendererStep
          ├── RenderTexturedStep / NormalStep / DepthStep / EdgeStep
          └── ...13 reusable steps total
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

### Scene Snapshots

All `BlenderStep` subclasses automatically save `.blend` snapshots after each successful step to a `.pipeline/` directory alongside deliverables. This enables debugging (open any snapshot in Blender's GUI) and pipeline resumability.

```
output/model_stem/
├── render_textured.png     <-- deliverables
├── metadata.json
└── .pipeline/              <-- auto-saved artifacts
    ├── after_prepare_scene.blend
    ├── after_import_model.blend
    └── after_setup_camera.blend
```

Snapshots are on by default and controllable at three levels:
- Per-run: `context["save_scenes"] = False`
- Per-step: `RenderTexturedStep(save_scene=False)`
- Default: on

## Project Structure

```
blender_excercises/
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
    │   ├── pipeline_steps/     13 reusable step classes
    │   └── bpy/                Blender utility functions
    ├── ex_1__mini_inspector/   Exercise 1 (complete)
    ├── ex_2__thumbnail_renderer/  Exercise 2 (functional)
    ├── ex_3__proc_camera/      Exercise 3 (not started)
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
