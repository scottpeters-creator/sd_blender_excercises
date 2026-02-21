# Project Context — Blender Exercises Pipeline

This is the canonical context file for the blender_exercises project. Downstream agents implementing exercises 1–5 should read this first.

---

## 1. Project Structure

```
blender_exercises/
├── pyproject.toml                      ← pip install -e . for package imports
├── context.md                          ← you are here
├── source/
│   └── Onboarding for 3D Data Engineering ...md   ← exercise specifications
├── dev_tasks/
│   ├── 001_pipeline_dev/
│   │   └── POR.md                      ← pipeline redesign plan of record (complete)
│   └── 002_pdg_refactor/
│       └── POR.md                      ← PDG-Lite evolution plan of record (complete)
└── src/
    ├── lib/
    │   ├── __init__.py                 ← package init
    │   ├── pipeline.py                 ← composable pipeline library + PDG-Lite extensions
    │   ├── work_item.py                ← WorkItem dataclass (typed data packet)
    │   ├── scheduler.py                ← Scheduler ABC + Local/Pool/Subprocess/SLURM
    │   ├── naming.py                   ← OutputNamer + ensure_directory
    │   ├── test_pipeline.py            ← 96 unit tests (all passing)
    │   ├── pipeline_steps/             ← REUSABLE STEP LIBRARY (the "nodes")
    │   │   ├── __init__.py
    │   │   ├── scene.py                ← PrepareSceneStep, CleanupSceneStep
    │   │   ├── io.py                   ← ImportModelStep, GrepModelsStep
    │   │   ├── analysis.py             ← CollectMeshesStep, ComputeCountsStep
    │   │   ├── camera.py               ← SetupCameraStep
    │   │   ├── lighting.py             ← SetupEnvironmentLightStep
    │   │   ├── render.py               ← ConfigureRendererStep, RenderTexturedStep,
    │   │   │                              RenderNormalStep, RenderDepthStep, RenderEdgeStep,
    │   │   │                              ConfigureVideoOutputStep, RenderAnimationStep
    │   │   └── reporting.py            ← WriteJSONStep, WriteMetadataStep
    │   └── bpy/                        ← shared Blender utility functions
    │       ├── __init__.py
    │       ├── scene.py                ← reset_scene, purge_orphan_data, move_to_origin
    │       ├── io.py                   ← import_model, open_blend
    │       ├── mesh.py                 ← collect_meshes, compute_geometry_counts, etc.
    │       ├── camera.py               ← create_camera, frame_objects, get_camera_intrinsics
    │       ├── render.py               ← configure_cycles, enable_gpu, render_frame, etc.
    │       └── animation.py            ← is_animated, create_orbit, create_bezier_path
    ├── ex_1__mini_inspector/           ← COMPLETE
    │   ├── __init__.py
    │   ├── mini_inspector.py           ← composition + CLI (~200 lines)
    │   └── Plan_of_Record.md
    ├── ex_2__thumbnail_renderer/       ← COMPLETE (functional)
    │   ├── __init__.py
    │   ├── render_task.py              ← composition + CLI (~170 lines)
    │   └── Plan_of_Record.md
    └── ex_3__proc_camera/              ← COMPLETE
        ├── __init__.py
        ├── proc_camera.py              ← composition + CLI (~240 lines)
        └── Plan_of_Record.md
```

New exercises should follow the naming pattern `src/ex_N__short_name/`.

---

## 2. Pipeline Library Reference

The pipeline library is **Blender-agnostic** — it never imports `bpy`. All Blender-specific logic lives in step `run()` methods inside exercise scripts.

### 2.1 Class Hierarchy

```
PipelineStep (ABC)
    ├── Pipeline        (IS-A PipelineStep, contains N steps — composable/nestable)
    ├── StepGroup       (IS-A PipelineStep, contains N steps, supports parallel flag)
    ├── GeneratorStep   (IS-A PipelineStep, fan-out: 1 WorkItem → N WorkItems)
    ├── CollectorStep   (IS-A PipelineStep, fan-in:  N WorkItems → 1 WorkItem)
    └── BlenderStep     (IS-A PipelineStep, base for bpy steps — auto scene snapshots)
```

**Key design: the Composite pattern.** `Pipeline` extends `PipelineStep`, so a pipeline can be placed inside another pipeline. This enables reuse across exercises (e.g., Ex 1's `scene_prep` pipeline can be embedded in Ex 2's pipeline).

**BlenderStep** (`lib/pipeline_steps/blender_step.py`) extends `PipelineStep` for any step that interacts with `bpy`. Subclasses implement `execute(context)` instead of `run(context)`. After each successful `execute()`, the Blender scene is optionally saved as a `.blend` snapshot for debugging and resumability.

Scene save control (highest priority wins):

| Level | How | Example |
|---|---|---|
| Per-run (context) | `context["save_scenes"] = False` | Disable for production batch runs |
| Per-step (constructor) | `RenderTexturedStep(save_scene=False)` | Skip snapshot for a specific step |
| Default | `BlenderStep(save_scene=True)` | On by default |

Artifacts are saved to `{output_path}/.pipeline/after_{step_name}.blend` by default, or to `context["artifacts_dir"]` if set explicitly.

### 2.2 Core Classes

#### `CompletedState` (dataclass)
The universal step result. Every `run()` must return one.

| Field | Type | Purpose |
|---|---|---|
| `success` | `bool` | Did the step succeed? |
| `timestamp` | `str` | ISO 8601 UTC timestamp |
| `duration_s` | `float` | Set by orchestrator (steps can pass 0.0) |
| `provides` | `List[str]` | Context keys this step populated |
| `outputs` | `Dict[str, Any]` | Small summary data |
| `error` | `Optional[Dict]` | `{"message": ...}` or `{"type": ..., "message": ...}` |
| `meta` | `Dict[str, Any]` | Arbitrary metadata |
| `signature` | `Optional[str]` | Short hash for idempotency checks |

#### `PipelineStep` (ABC)
Base class for all steps. Constructor parameters:

| Param | Default | Purpose |
|---|---|---|
| `name` | (required) | Unique step identifier |
| `requires` | `set()` | Context keys that must exist before this step runs |
| `provides` | `set()` | Context keys this step will populate |
| `idempotent` | `True` | If True, orchestrator skips if already succeeded |
| `continue_on_error` | `False` | If True, pipeline continues after this step fails |
| `version` | `"0.0.0"` | Semantic version string |
| `scheduler` | `None` | Per-step execution backend override |

Subclasses must implement:
- `run(context: Dict[str, Any]) -> CompletedState`

May optionally override:
- `validate(context)` — called by orchestrator after successful run; returns `False` if `provides` keys missing
- `rollback(context)` — called by orchestrator on failure or exception

#### `Pipeline(PipelineStep)`
Composable orchestrator. Constructor parameters:

| Param | Default | Purpose |
|---|---|---|
| `name` | `"pipeline"` | Pipeline identity |
| `version` | `"1.0"` | Version string |
| `steps` | `[]` | List of `PipelineStep` or `Pipeline` instances |
| `requires` | auto-inferred | External deps not satisfied by children |
| `provides` | auto-inferred | Union of all children's provides |
| `force` | `False` | Force re-run of idempotent steps |
| `middleware` | `default_middleware` | Composable middleware chain for step orchestration |
| `resolve_order` | `False` | If True, topologically sort steps by requires/provides |
| `scheduler` | `None` | Execution backend (LocalScheduler, SubprocessScheduler, etc.) |

Run signatures:
- `pipeline.run(context, *, dry_run=False) -> CompletedState` — backward-compatible dict-based entry point
- `pipeline.run_item(work_item, *, dry_run=False) -> CompletedState` — WorkItem-based entry point (supports fan-out/fan-in)

The orchestrator owns (via middleware chain):
- **Timing** — wraps each `step.run()`, overwrites `duration_s` on the returned `CompletedState`
- **`step_states` writes** — steps never touch `context["step_states"]`; the orchestrator records results with dot-delimited namespaced keys (e.g., `"mini_inspector_e2e.scene_prep.prepare_scene"`)
- **Validation** — calls `step.validate(context)` after success; treats failure as step failure
- **Logging** — `logging.getLogger("pipeline")` logs START/OK/SKIP/FAIL for every step

#### `GeneratorStep(PipelineStep)` and `CollectorStep(PipelineStep)`
Fan-out / fan-in primitives for batch processing:

- `GeneratorStep.generate(work_item) -> List[WorkItem]` — produce N work items from 1
- `CollectorStep.collect(work_items) -> WorkItem` — merge N work items into 1

The orchestrator handles multiplication automatically: downstream steps run once per emitted item until a CollectorStep (or pipeline end) collects them.

#### `StepGroup(PipelineStep)`
Groups steps for sequential or parallel execution.

| Param | Default | Purpose |
|---|---|---|
| `name` | (required) | Group identity |
| `steps` | `[]` | Child steps |
| `parallel` | `False` | If True, uses `ThreadPoolExecutor` |
| `requires`/`provides` | auto-inferred | Same inference as Pipeline |

When `parallel=True`, constructor validates that child steps have **non-overlapping `provides`** sets (raises `ValueError` otherwise).

#### `ExitCode` (IntEnum) and `exit_code_from()`
For CLI `sys.exit()` mapping:

| Code | Name | Meaning |
|---|---|---|
| 0 | `OK` | Success |
| 1 | `MISSING_REQUIREMENT` | Step's `requires` keys not in context |
| 2 | `STEP_EXCEPTION` | Step raised an exception |
| 3 | `INVALID_RETURN` | Step returned non-CompletedState |
| 4 | `STEP_FAILED` | Step returned `success=False` |

### 2.3 WorkItem (`src/lib/work_item.py`)

A `WorkItem` is a typed data packet that flows through the pipeline independently, replacing the raw dict for batch and fan-out scenarios.

| Field | Type | Purpose |
|---|---|---|
| `id` | `str` | Unique identifier (e.g., filename stem) |
| `attributes` | `Dict[str, Any]` | Key-value data (replaces context dict) |
| `input_files` | `List[str]` | Tagged input file paths |
| `output_files` | `List[str]` | Tagged output file paths |
| `parent_id` | `Optional[str]` | Fan-out provenance (ID of parent item) |
| `meta` | `Dict[str, Any]` | Scheduler hints, timing, etc. |

Key methods: `to_json()`, `from_json(data)`, `validate_serializable()`, `generate_id(prefix)`.

### 2.4 Reusable Step Library (`src/lib/pipeline_steps/`)

Pre-built step subclasses organized by domain. These are the building blocks ("nodes") for composing pipelines. Exercise scripts import and assemble them — they should not redefine shared steps.

Steps that interact with `bpy` extend `BlenderStep` (auto scene snapshots). Steps that are pure Python extend `PipelineStep` directly.

| Module | Steps | Base class | Used by |
|---|---|---|---|
| `blender_step.py` | `BlenderStep` | PipelineStep | All bpy steps |
| `scene.py` | `PrepareSceneStep`, `CleanupSceneStep` | BlenderStep | All exercises |
| `io.py` | `ImportModelStep` | BlenderStep | All exercises |
| `io.py` | `GrepModelsStep` | GeneratorStep | All exercises |
| `analysis.py` | `CollectMeshesStep`, `ComputeCountsStep` | BlenderStep | Ex 1, 2, 4 |
| `camera.py` | `SetupCameraStep` | BlenderStep | Ex 2, 5 |
| `lighting.py` | `SetupEnvironmentLightStep` | BlenderStep | Ex 2, 5 |
| `render.py` | `ConfigureRendererStep`, `RenderTexturedStep`, `RenderNormalStep`, `RenderDepthStep`, `RenderEdgeStep` | BlenderStep | Ex 2, 5 |
| `render.py` | `ConfigureVideoOutputStep`, `RenderAnimationStep` | BlenderStep | Ex 3, 5 |
| `reporting.py` | `WriteJSONStep` | PipelineStep | Ex 1, 2, 4 |
| `reporting.py` | `WriteMetadataStep` | BlenderStep | Ex 2 |

Import convention:
```python
from lib.pipeline_steps.scene import PrepareSceneStep, CleanupSceneStep
from lib.pipeline_steps.io import ImportModelStep, GrepModelsStep
from lib.pipeline_steps.camera import SetupCameraStep
from lib.pipeline_steps.lighting import SetupEnvironmentLightStep
from lib.pipeline_steps.render import ConfigureRendererStep, RenderTexturedStep
from lib.pipeline_steps.render import ConfigureVideoOutputStep, RenderAnimationStep
from lib.pipeline_steps.reporting import WriteJSONStep, WriteMetadataStep
```

Exercise-specific steps (e.g., `AssembleReportStep` in Ex 1 with a custom report schema, `OpenBlendStep` in Ex 3 for scene-replacing loads) stay in the exercise script.

### 2.5 Schedulers (`src/lib/scheduler.py`)

Schedulers decouple *what to do* from *where to do it*.

| Class | Purpose |
|---|---|
| `Scheduler` (ABC) | `execute(step, work_item) -> CompletedState` |
| `LocalScheduler` | Run in-process (default, shared bpy state) |
| `PoolScheduler` | Persistent pool of N worker processes (amortized startup, full isolation) |
| `SubprocessScheduler` | Spawn a fresh process per work item (full isolation, per-item cost) |
| `SLURMScheduler` | Submit as HPC batch job via `submitit` (cluster execution) |

Configuration: `Pipeline(steps=[...], scheduler=LocalScheduler())` or per-step override via `PipelineStep(..., scheduler=...)`.

Scheduler routing applies to both leaf steps and composite Pipeline steps. When a scheduler is set, the entire step (including nested sub-pipelines) is dispatched through `scheduler.execute()`.

### 2.5 Middleware Chain

The orchestrator's `_execute_step()` is decomposed into a composable middleware chain:

```python
default_middleware = [
    _idempotent_skip_mw,      # skip if already succeeded
    _state_write_mw,           # write result to step_states
    _requirements_check_mw,    # check requires keys exist
    _logging_mw,               # log START/OK/FAIL
    _timing_mw,                # wall-clock timing
    _execute_mw,               # try/except, validate, rollback
]

Pipeline(steps=[...], middleware=[retry_mw, *default_middleware])
```

Each middleware has the signature `(ctx: _StepExecContext, next_fn: () -> CompletedState) -> CompletedState`. It can short-circuit by returning without calling `next_fn()`, or wrap the downstream result.

### 2.6 Helpers

- `write_json_atomic(path, data)` — atomic JSON write via temp file + move
- `_short_signature(value)` — deterministic SHA1-based short hash for idempotency

### 2.7 What Steps Must Do (and Not Do)

**DO:**
- Set your `provides` keys on `context` (e.g., `context["mesh_objects"] = meshes`)
- Return a `CompletedState` with `success=True/False`
- Declare `requires` and `provides` accurately in `__init__`
- For bpy steps: extend `BlenderStep` and implement `execute()` (not `run()`)
- For pure Python steps: extend `PipelineStep` and implement `run()`

**DO NOT:**
- Write to `context["step_states"]` — the orchestrator owns this
- Time yourself — pass `duration_s=0.0`, orchestrator overwrites
- Call `context.setdefault("step_states", {})` — orchestrator handles this
- Override `run()` on a `BlenderStep` — implement `execute()` instead

---

## 3. How-To Guide: Building Pipelines

This section walks through every pattern the framework supports, from the simplest to the most advanced. Each example is self-contained.

### 3.1 Define a step

Every step extends `PipelineStep` and implements `run(context)`.

```python
from lib.pipeline import PipelineStep, CompletedState

class ComputeCountsStep(PipelineStep):
    def __init__(self) -> None:
        super().__init__(
            name="compute_counts",
            requires=["mesh_objects"],     # must exist in context before this step runs
            provides=["counts"],           # this step promises to set this key
        )

    def run(self, context):
        meshes = context["mesh_objects"]
        counts = {"vertices": len(meshes) * 100, "faces": len(meshes) * 50}
        context["counts"] = counts        # fulfill the provides contract
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,               # orchestrator will overwrite
            provides=["counts"],
            outputs={"counts": counts},   # small summary for logging
        )
```

**Rules:**
- `name` must be unique within a pipeline.
- `requires` keys must be present in the context dict before `run()` is called. The orchestrator checks this automatically and fails the step if any are missing.
- `provides` keys must be set on `context` by `run()`. The orchestrator calls `validate()` after success to verify.
- Return `duration_s=0.0` — the orchestrator wraps your `run()` with timing middleware and overwrites it.

### 3.2 Compose a linear pipeline

Chain steps into a `Pipeline`. Steps run in list order. The context dict flows through all of them.

```python
from lib.pipeline import Pipeline

scene_prep = Pipeline(
    name="scene_prep",
    version="1.0",
    steps=[PrepareSceneStep(), ImportModelStep()],
)

analysis = Pipeline(
    name="analysis",
    version="1.0",
    steps=[CollectMeshesStep(), ComputeCountsStep()],
)

# Nest pipelines inside other pipelines (Composite pattern)
e2e = Pipeline(
    name="mini_inspector_e2e",
    version="1.0",
    steps=[scene_prep, analysis, ReportStep(), CleanupStep()],
)

# Run it
context = {"input_path": "model.glb", "output_path": "report.json"}
result = e2e.run(context)
# result.success is True/False
# context now contains all keys set by all steps
```

**Key behavior:**
- If any step fails (returns `success=False`), the pipeline stops and returns the failure unless `continue_on_error=True` on that step.
- Idempotent steps are skipped on re-run (context carries `step_states` from prior runs).
- Use `force=True` on the Pipeline to re-run all steps regardless.

### 3.3 Reuse pipelines across exercises

Because `Pipeline` IS-A `PipelineStep`, you can embed one exercise's pipeline inside another:

```python
from ex_1__mini_inspector.mini_inspector import scene_prep, cleanup

# Ex 2 reuses Ex 1's scene preparation and cleanup
ex2_pipeline = Pipeline(
    name="thumbnail_renderer",
    version="1.0",
    steps=[scene_prep, CameraSetupStep(), RenderStep(), MetadataStep(), cleanup],
)
```

### 3.4 Run steps in parallel with StepGroup

Use `StepGroup(parallel=True)` when steps are independent and can execute concurrently:

```python
from lib.pipeline import StepGroup

renders = StepGroup(
    name="render_modalities",
    parallel=True,
    steps=[
        RenderTexturedStep(),   # provides=["textured_path"]
        RenderNormalStep(),     # provides=["normal_path"]
        RenderDepthStep(),      # provides=["depth_path"]
        RenderEdgeStep(),       # provides=["edge_path"]
    ],
)
```

**Constraint:** Parallel children must have **non-overlapping `provides`** sets. The constructor raises `ValueError` if two children claim the same key. This prevents data races on the shared context.

### 3.5 Auto-order steps with the DAG solver

If you prefer not to manually order steps, enable `resolve_order=True`. The Pipeline topologically sorts steps based on their `requires`/`provides` contracts:

```python
p = Pipeline(
    name="auto_ordered",
    steps=[
        WriteReportStep(),      # requires=["counts"]
        ComputeCountsStep(),    # requires=["mesh_objects"], provides=["counts"]
        CollectMeshesStep(),    # provides=["mesh_objects"]
    ],
    resolve_order=True,
    # Actual execution order: Collect → Compute → Write
)
```

**Behavior:**
- Declared order is the tiebreaker when multiple valid orderings exist.
- Cycles raise `ValueError` at construction time (not at runtime).
- `resolve_order=False` (the default) preserves list order — zero risk to existing code.

### 3.6 Fan-out / fan-in with WorkItem

For batch processing or parameterized execution (e.g., 1 model → 4 render modalities), use `GeneratorStep` and `CollectorStep` with `WorkItem`:

```python
from lib.pipeline import GeneratorStep, CollectorStep, Pipeline
from lib.work_item import WorkItem

class ModalitySplitter(GeneratorStep):
    def __init__(self):
        super().__init__(name="split_modalities", provides=[])

    def generate(self, work_item):
        return [
            WorkItem(
                id=f"{work_item.id}_{m}",
                attributes={**work_item.attributes, "modality": m},
            )
            for m in ["textured", "normal", "depth", "edge"]
        ]

class RenderModalityStep(PipelineStep):
    def __init__(self):
        super().__init__(name="render", provides=["render_path"])

    def run(self, context):
        modality = context["modality"]  # parameterized by the generated work item
        path = f"/tmp/{context.get('model_name', 'model')}_{modality}.png"
        context["render_path"] = path
        return CompletedState(
            success=True, timestamp=CompletedState.now_iso(),
            duration_s=0.0, provides=["render_path"],
        )

class ResultCollector(CollectorStep):
    def __init__(self):
        super().__init__(name="collect", provides=["all_renders"])

    def collect(self, work_items):
        all_paths = [wi.attributes.get("render_path") for wi in work_items]
        merged = dict(work_items[0].attributes)
        merged["all_renders"] = all_paths
        return WorkItem(id="merged", attributes=merged)

pipeline = Pipeline(
    name="render_pipeline",
    steps=[ModalitySplitter(), RenderModalityStep(), ResultCollector()],
)

# Must use run_item() for fan-out support
wi = WorkItem(id="model_001", attributes={"model_name": "chair"})
result = pipeline.run_item(wi)
# wi.attributes["all_renders"] == [4 paths]
```

**How it works:**
1. `GeneratorStep.generate()` is called once per current work item. It returns N new WorkItems.
2. All downstream steps run once **per generated item** (until a `CollectorStep` or end of pipeline).
3. `CollectorStep.collect()` receives all accumulated items and returns a single merged item.
4. After collection, subsequent steps operate on the single merged item, and its attributes are merged back into the original context.

**Important:** Fan-out requires `pipeline.run_item(work_item)`, not `pipeline.run(context)`. Each generated item gets its own independent `step_states` to prevent false idempotent skips.

### 3.7 Batch processing with WorkItem (Ex 4 pattern)

For processing many files through the same pipeline, create a WorkItem per file:

```python
from lib.work_item import WorkItem
from ex_1__mini_inspector.mini_inspector import build_e2e_pipeline

pipeline = build_e2e_pipeline()

for filepath in file_list:
    wi = WorkItem(
        id=Path(filepath).stem,
        attributes={"input_path": filepath, "output_path": f"{filepath}.json"},
        input_files=[filepath],
    )
    result = pipeline.run_item(wi)
    if result.success:
        print(f"OK: {wi.id}")
    else:
        print(f"FAIL: {wi.id} — {result.error}")
```

Each WorkItem carries its own independent `attributes` dict, so pipeline state doesn't leak between iterations.

### 3.8 Add custom middleware (retry, profiling)

Middleware wraps step execution. Each middleware receives `(ctx, next_fn)` and returns a `CompletedState`. Call `next_fn()` to proceed to the next middleware, or return early to short-circuit.

**Retry middleware:**

```python
from lib.pipeline import Pipeline, default_middleware

def retry_middleware(ctx, next_fn):
    for attempt in range(3):
        cs = next_fn()
        if cs.success:
            return cs
    return cs  # return last failure after 3 attempts

pipeline = Pipeline(
    name="resilient",
    steps=[FlakyRenderStep()],
    middleware=[retry_middleware, *default_middleware],
)
```

**Profiling middleware:**

```python
import time

def profiling_middleware(ctx, next_fn):
    start = time.perf_counter()
    cs = next_fn()
    elapsed = time.perf_counter() - start
    print(f"[PROFILE] {ctx.step_key}: {elapsed:.4f}s")
    return cs

pipeline = Pipeline(
    name="profiled",
    steps=[...],
    middleware=[*default_middleware[:3], profiling_middleware, *default_middleware[3:]],
)
```

**Default middleware chain (for reference):**

```
_idempotent_skip_mw → _state_write_mw → _requirements_check_mw → _logging_mw → _timing_mw → _execute_mw → step.run()
```

Insert custom middleware at any point. Middleware that should fire before idempotent skip goes at the front. Middleware that should wrap execution goes between `_requirements_check_mw` and `_logging_mw`.

### 3.9 Use schedulers for process isolation

Schedulers control **where** a step executes. This is useful for memory isolation (bpy leaks in loops) or HPC submission.

```python
from lib.pipeline import Pipeline
from lib.scheduler import LocalScheduler, SubprocessScheduler

# All steps run in a fresh subprocess (memory isolation)
pipeline = Pipeline(
    name="isolated",
    steps=[ImportStep(), AnalyzeStep(), CleanupStep()],
    scheduler=SubprocessScheduler(timeout=300),
)

# Or override per-step for GPU-heavy work
render_step = RenderStep(scheduler=SubprocessScheduler(timeout=600))
pipeline = Pipeline(
    name="mixed",
    steps=[PrepStep(), render_step, ReportStep()],
    scheduler=LocalScheduler(),  # default for other steps
)
```

**Behavior:**
- `LocalScheduler`: calls `step.run(work_item.attributes)` directly in-process. Default when no scheduler is set.
- `SubprocessScheduler`: serializes the WorkItem to JSON, spawns a fresh Python process, deserializes the CompletedState result. Context mutations in the subprocess do **not** propagate back — only the CompletedState return value survives.
- `SLURMScheduler`: stub, raises `NotImplementedError` (implementation deferred to Ex 5).
- Step-level schedulers override pipeline-level schedulers.

### 3.10 Wire up the CLI

Every exercise should have a `main()` function that parses args, builds the pipeline, and translates the result to an exit code:

```python
import argparse, sys, logging
from lib.pipeline import Pipeline, ExitCode, exit_code_from

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def build_pipeline(force=False):
    return Pipeline(
        name="ex_N_e2e",
        version="1.0",
        steps=[...],
        force=force,
    )

def main(argv=None):
    parser = argparse.ArgumentParser(prog="ex_N")
    parser.add_argument("input", help="Input model file")
    parser.add_argument("output", help="Output JSON path")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    context = {"input_path": args.input, "output_path": args.output}
    result = build_pipeline(force=args.force).run(context, dry_run=args.dry_run)
    return int(exit_code_from(result))

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

### 3.11 Preview execution with dry-run

Dry-run logs the planned execution order without calling any `run()` methods:

```python
result = pipeline.run(context, dry_run=True)
# Logs:
#   DRY-RUN plan for [pipeline_name]:
#     pipeline_name.step_a — READY
#     [pipeline] pipeline_name.sub_pipeline — READY
#       pipeline_name.sub_pipeline.step_b — BLOCKED (missing: ['x'])
```

This is useful for debugging step ordering, verifying requirements, and understanding nested pipeline structure.

---

## 4. Shared Blender Library (`src/lib/bpy/`)

The `lib/bpy/` package contains reusable Blender functions shared across exercises. Each module is focused on a single domain. These functions are **not** pipeline steps — they are utility functions that step `run()` methods call internally.

### 4.1 Cross-Exercise Usage

| Module | Ex 1 | Ex 2 | Ex 3 | Ex 4 | Ex 5 |
|---|---|---|---|---|---|
| `scene.py` | x | x | x | x | x |
| `io.py` | x | x | x | x | x |
| `mesh.py` | x | x | | x | |
| `camera.py` | | x | x | | x |
| `render.py` | | x | x | | x |
| `animation.py` | | | x | | x |

### 4.2 `scene.py` — Scene Lifecycle

| Function | Signature | Purpose |
|---|---|---|
| `reset_scene()` | `() -> None` | Clear scene via `read_homefile(use_empty=True)` + purge orphans |
| `purge_orphan_data()` | `() -> None` | Remove all unused data blocks (meshes, materials, textures, images, etc.) — critical for batch loops |
| `delete_all_objects()` | `() -> None` | Delete all objects without resetting the file |
| `move_to_origin(obj, zero_rotation=False)` | `(Object, bool) -> None` | Translate object so its bbox center is at world origin |
| `get_scene_bounds(scene=None)` | `(Scene?) -> (Vector, Vector)` | World-space AABB of all meshes — returns (min_corner, max_corner) |

### 4.3 `io.py` — Model Import / Open

| Function | Signature | Purpose |
|---|---|---|
| `import_model(filepath, link=False)` | `(str, bool) -> List[Object]` | Auto-detect extension (.glb/.gltf/.obj/.fbx/.stl/.blend), call correct import op, return new objects |
| `open_blend(filepath)` | `(str) -> None` | Open a .blend file directly (replaces current scene) |
| `supported_extensions()` | `() -> List[str]` | Return list of supported file extensions |

`import_model` tracks objects before/after import to return only the newly added ones. Raises `FileNotFoundError`, `ValueError` (bad extension), or `RuntimeError` (import failure).

### 4.4 `mesh.py` — Mesh Collection & Geometry Analysis

| Function | Signature | Purpose |
|---|---|---|
| `collect_meshes(scene=None)` | `(Scene?) -> List[Object]` | Filter scene objects where `type == 'MESH'` |
| `compute_geometry_counts(objects)` | `(List[Object]) -> {"vertices": int, "faces": int}` | Total vertex + face counts via evaluated depsgraph (respects modifiers) |
| `compute_topology(objects)` | `(List[Object]) -> {"triangles": int, "quads": int, "ngons": int}` | Classify faces by vertex count |
| `compute_world_bbox(objects)` | `(List[Object]) -> {"center": [...], "dimensions": [...], "min": [...], "max": [...]}` | Combined world-space AABB across all objects |
| `get_material_names(objects)` | `(List[Object]) -> List[str]` | Sorted unique material names from all slots |

All geometry functions use `evaluated_get(depsgraph)` + `to_mesh()` / `to_mesh_clear()` for accurate modifier-applied results.

### 4.5 `camera.py` — Camera Setup & Framing

| Function | Signature | Purpose |
|---|---|---|
| `create_camera(name, location, focal_length, ...)` | `(...) -> Object` | Create camera, add to scene, set as active |
| `frame_objects(camera, objects, margin=1.3)` | `(Object, List[Object], float) -> None` | Reposition camera so all objects fit in frame (bounding sphere + FOV math) |
| `add_track_to(camera, target)` | `(Object, Object) -> Constraint` | Add Track To constraint (camera always faces target object) |
| `add_track_to_location(camera, location)` | `(Object, tuple) -> Object` | Create empty at location + track camera to it; returns the empty |
| `get_camera_intrinsics(camera, scene=None)` | `(Object, Scene?) -> Dict` | Extract focal length, sensor, position, rotation, resolution as JSON-serializable dict |

`frame_objects` calculates the bounding sphere radius of all objects, then positions the camera at the correct distance for the current FOV with a configurable margin multiplier.

### 4.6 `render.py` — Render Engine & Passes

| Function | Signature | Purpose |
|---|---|---|
| `configure_cycles(resolution, samples, use_denoising, transparent_background)` | `(tuple, int, bool, bool) -> None` | Set Cycles engine + resolution + samples |
| `configure_workbench(resolution, lighting, color_type)` | `(tuple, str, str) -> None` | Set Workbench engine for fast preview renders |
| `set_resolution(width, height)` | `(int, int) -> None` | Change resolution without touching engine settings |
| `enable_gpu()` | `() -> None` | OPTIX → CUDA fallback, macOS CPU fallback, logs all devices |
| `setup_depth_pass(normalize, output_path, file_format)` | `(bool, str?, str) -> None` | Enable Z pass in compositor, optional Normalize node + File Output |
| `setup_normal_pass(output_path, file_format)` | `(str?, str) -> None` | Enable Normal pass in compositor |
| `setup_edge_output(line_thickness)` | `(float) -> None` | Enable Freestyle line-art rendering |
| `render_frame(output_path, file_format, color_depth)` | `(str, str, str) -> str` | Render current frame to disk |
| `configure_video_output(output_path, fps, codec, ...)` | `(...) -> None` | Set up FFmpeg MP4 output |
| `render_animation()` | `() -> None` | Render full animation timeline |

`enable_gpu()` implements the OPTIX/CUDA detection pattern from the onboarding doc, with macOS CPU fallback and full device logging.

### 4.7 `animation.py` — Animation Utilities

| Function | Signature | Purpose |
|---|---|---|
| `is_animated(obj, threshold=0.01)` | `(Object, float) -> bool` | Check if object drifts between first/last keyframes (for Ex 5 filtering) |
| `has_any_animation(obj)` | `(Object) -> bool` | Quick check: does object have animation_data + action? |
| `create_orbit(camera, center, radius, height, frame_count, frame_start)` | `(...) -> None` | Keyframe a circular XY-plane orbit (Ex 5: 360 frames around subject) |
| `create_bezier_path(points, name, resolution)` | `(List[tuple], str, int) -> Object` | Create Bezier curve from control points, return curve object |
| `constrain_to_path(obj, curve, frame_count, frame_start)` | `(Object, Object, int, int) -> Constraint` | Add Follow Path constraint + keyframe offset for smooth traversal |
| `create_linear_keyframes(camera, positions, frame_count, frame_start)` | `(Object, List[tuple], int, int) -> None` | Keyframe linear interpolation between waypoints (Ex 3 Mode B) |

### 4.8 Import Convention

From exercise scripts:

```python
from lib.bpy.scene import reset_scene, purge_orphan_data
from lib.bpy.io import import_model
from lib.bpy.mesh import collect_meshes, compute_geometry_counts, compute_topology, compute_world_bbox
from lib.bpy.camera import create_camera, frame_objects, get_camera_intrinsics
from lib.bpy.render import configure_cycles, enable_gpu, render_frame, setup_depth_pass
from lib.bpy.animation import create_orbit, is_animated, create_bezier_path
```

---

## 5. Context Model

A single shared `Dict[str, Any]` flows through the entire pipeline tree. The `requires`/`provides` contracts enforce discipline. There is no scoping or context-copying between nested pipelines (except during fan-out, where each generated WorkItem gets its own attributes dict).

Standard context keys across exercises:

| Key | Type | Set by | Used by |
|---|---|---|---|
| `input_path` | `str` | CLI harness | ImportModelStep |
| `output_path` | `str` | CLI harness | WriteJSONStep |
| `step_states` | `Dict[str, Dict]` | Orchestrator | Orchestrator (idempotent skip) |
| `scene_prepared` | `bool` | PrepareSceneStep | ImportModelStep |
| `imported_objects` | `List[str]` | ImportModelStep | CollectMeshesStep |
| `mesh_objects` | `List` | CollectMeshesStep | ComputeCountsStep |
| `counts` | `Dict` | ComputeCountsStep | AssembleReportStep |
| `report` | `Dict` | AssembleReportStep | WriteJSONStep |
| `output_written` | `bool` | WriteJSONStep | — |
| `cleaned` | `bool` | CleanupSceneStep | — |

Exercise-specific steps should add their own keys following the same pattern.

---

## 6. Import Path Convention

The project uses a `pyproject.toml` with `src/` layout. After one-time setup (`pip install -e .` from the project root), all packages are importable from anywhere:

```bash
conda activate blender_exercises
cd blender_exercises    # project root (where pyproject.toml lives)
pip install -e .         # one-time — registers src/ as package root
```

Then in exercise scripts — **no `sys.path` hacks needed**:

```python
# Pipeline framework
from lib.pipeline import (
    CompletedState, Pipeline, PipelineStep, StepGroup,
    GeneratorStep, CollectorStep,
    ExitCode, exit_code_from, write_json_atomic, default_middleware,
)
from lib.work_item import WorkItem
from lib.naming import OutputNamer, ensure_directory
from lib.scheduler import LocalScheduler, PoolScheduler, SubprocessScheduler

# Reusable pipeline steps (the "nodes")
from lib.pipeline_steps.scene import PrepareSceneStep, CleanupSceneStep
from lib.pipeline_steps.io import ImportModelStep, GrepModelsStep
from lib.pipeline_steps.analysis import CollectMeshesStep, ComputeCountsStep
from lib.pipeline_steps.camera import SetupCameraStep
from lib.pipeline_steps.lighting import SetupEnvironmentLightStep
from lib.pipeline_steps.render import ConfigureRendererStep, RenderTexturedStep
from lib.pipeline_steps.reporting import WriteJSONStep, WriteMetadataStep
```

**Important:**
- Blender `bpy` imports are **deferred** inside step `run()` methods, not at module top level. This allows the module to be imported for pipeline composition and unit testing without bpy present.
- Exercise scripts should **not redefine shared steps** — import them from `lib.pipeline_steps`. Only exercise-specific steps (e.g., custom report formats) live in the exercise script.

---

## 7. Exercise Composition Patterns

Each exercise reuses and extends the pipeline infrastructure:

| Exercise | Script | Status | Composition Pattern |
|---|---|---|---|
| **Ex 1** — Mini Inspector | `mini_inspector.py` | **Complete** | `GrepModelsStep` fan-out → `inspect_model` (depth-first per item): `scene_prep → analysis → reporting → cleanup`. Single-file and batch modes. See `ex_1__mini_inspector/Plan_of_Record.md`. |
| **Ex 2** — Thumbnail Renderer | `render_task.py` | **Functional** | `GrepModelsStep` fan-out → `render_model` (depth-first): `scene_prep → camera_setup → lighting → render (textured/normal/depth/edge) → metadata → cleanup`. All steps from `lib/pipeline_steps/`. See `ex_2__thumbnail_renderer/Plan_of_Record.md`. |
| **Ex 3** — Procedural Camera | `proc_camera.py` | **Complete** | `OpenBlendStep` (preserves full scene) → `GenerateCameraPathStep` → `SetupAnimationStep` (mode A spline / mode B point-to-point) → `ConfigureRendererStep` + `ConfigureVideoOutputStep` + `RenderAnimationStep` → `WriteCameraLogStep` → `CleanupSceneStep`. Exercise-specific steps in script; video steps shared in `lib/pipeline_steps/render.py`. See `ex_3__proc_camera/Plan_of_Record.md`. |
| **Ex 4** — Batch Validator | `batch_validator.py` | Not started | Reuse `GrepModelsStep` for fan-out. Embed Ex 1's `build_e2e_pipeline()` as the per-model step. `PoolScheduler` or `SubprocessScheduler` for memory isolation on large batches. |
| **Ex 5** — HPC Pipeline | `process_batch.py` | Not started | `filter_animated → normalize → configure_renderer → orbit_animation → render_frames` composed pipeline. `SLURMScheduler` for HPC submission. |

---

## 8. Running and Testing

### Environment setup (one-time)
```bash
conda activate blender_exercises
cd blender_exercises          # project root (where pyproject.toml lives)
pip install -e .               # registers src/ packages for clean imports
```

The `bpy` module is installed in the `blender_exercises` conda env. Blender GUI is at `/Applications/Blender.app/Contents/MacOS/blender`.

### Unit tests (Blender-free, 96 tests)
```bash
python -m unittest lib.test_pipeline -v
```

Tests cover: flat/nested pipelines, missing requirements, idempotent skip, exception handling + rollback, validation, StepGroup (sequential + parallel), dry-run, exit codes, helpers, middleware chain, DAG solver, WorkItem, fan-out/fan-in, schedulers (Local, Pool, SLURM mock), and scheduler routing.

### Running Ex 1 — Mini Inspector
```bash
# Single file
python -m ex_1__mini_inspector.mini_inspector \
  /path/to/model.glb /path/to/report.json

# Batch — all .glb files in a directory
python -m ex_1__mini_inspector.mini_inspector \
  /path/to/models/ /path/to/reports/ --type glb

# Batch — first 30 only
python -m ex_1__mini_inspector.mini_inspector \
  /path/to/models/ /path/to/reports/ --type glb --limit 30

# Dry-run (preview plan without executing)
python -m ex_1__mini_inspector.mini_inspector \
  /path/to/models/ /path/to/reports/ --type glb --limit 5 --dry-run
```

### Running Ex 2 — Thumbnail Renderer
```bash
# Single file (4 renders: textured, normal, depth, edge + metadata.json)
python -m ex_2__thumbnail_renderer.render_task \
  /path/to/model.glb /path/to/renders/

# Batch
python -m ex_2__thumbnail_renderer.render_task \
  /path/to/models/ /path/to/renders/ --type glb --limit 10

# Custom render samples (lower = faster, higher = less noise)
python -m ex_2__thumbnail_renderer.render_task \
  /path/to/model.glb /path/to/renders/ --samples 64
```

### Running Ex 3 — Procedural Camera
```bash
# Mode A: random spline flythrough
python -m ex_3__proc_camera.proc_camera \
  /path/to/scene.blend /path/to/output/ --mode spline

# Mode B: point-to-point with look-at
python -m ex_3__proc_camera.proc_camera \
  /path/to/scene.blend /path/to/output/ --mode point

# Reproducible seed + custom frame count
python -m ex_3__proc_camera.proc_camera \
  /path/to/scene.blend /path/to/output/ --mode spline --seed 42 --frames 10

# Dry-run
python -m ex_3__proc_camera.proc_camera \
  /path/to/scene.blend /path/to/output/ --mode spline --dry-run
```

### Input data locations
- **Objaverse** (Ex 1, 2, 4): 300 .glb files at `/Users/scott.peters/dev/source/scott_data/objaverse/` (~2.7 GB)
- **Evermotion** (Ex 3): Indoor .blend scenes at `/Users/scott.peters/dev/source/scott_data/evermotion/AI33_001`

### Opening a model in Blender GUI (for visual verification)
```bash
/Applications/Blender.app/Contents/MacOS/blender /path/to/model.glb
```

---

## 9. Key Design Decisions

1. **Blender-agnostic orchestration.** The pipeline library never imports `bpy`. This means all orchestration, testing, and CI checks work without Blender installed. Blender-specific logic lives only in step `run()` implementations.

2. **Shared mutable context.** One `Dict[str, Any]` flows through the entire tree. This is simple and matches the linear data flow of all five exercises. Fan-out creates independent copies per work item; fan-in merges them back.

3. **Orchestrator owns bookkeeping.** Steps only do work and return results. The orchestrator handles timing, `step_states`, validation, logging, and error recording via a composable middleware chain. This eliminates the dual-write bug and boilerplate from step implementations.

4. **Auto-inference of contracts.** When `requires`/`provides` are not explicitly set on a Pipeline or StepGroup, they are inferred from children: `requires` = external deps not satisfied internally; `provides` = union of all children's provides.

5. **Composition over hooks.** Pre/post logic is expressed by nesting pipelines, not callbacks. Ex 4 wraps the entire Ex 1 pipeline as a single step inside a download/upload loop.

6. **Shared bpy library separate from pipeline.** The `lib/bpy/` package contains Blender utility functions that steps call internally. The pipeline library (`lib/pipeline.py`) never imports `bpy`. This separation means orchestration tests run without Blender, while bpy utilities are tested with Blender present.

7. **Middleware for extensibility.** The `_execute_step()` monolith is decomposed into 6 composable middleware functions. Users inject retry, profiling, or caching middleware without modifying step code or the orchestrator.

8. **WorkItem as unit of work.** The raw context dict is still supported (`run(context)`), but `WorkItem` enables fan-out/fan-in, subprocess serialization, and independent per-item state. This is the PDG-Lite core insight: one pipeline run can process many independent items.

9. **Scheduler decouples what from where.** Steps declare *what* to do. Schedulers control *where* it runs (in-process, subprocess, SLURM). This lets the same pipeline definition run locally for development and on HPC for production.

---

## 10. Next Steps for Each Exercise

### Ex 1 — Mini Inspector — COMPLETE

See `src/ex_1__mini_inspector/Plan_of_Record.md` for full implementation details.

Exports available for reuse by other exercises:
- `scene_prep` — Pipeline: PrepareSceneStep + ImportModelStep
- `cleanup` — Pipeline: CleanupSceneStep
- `GrepModelsStep` — GeneratorStep that scans a directory for model files
- `build_e2e_pipeline()` — full single-file inspector
- `build_batch_pipeline()` — full batch inspector with fan-out

### Ex 2 — Thumbnail Renderer — FUNCTIONAL

See `src/ex_2__thumbnail_renderer/Plan_of_Record.md` for full implementation details.

Renders 4 modalities (textured, normal, depth, edge) at 512x512 plus metadata.json per model. Uses material override shaders for each modality (4 sequential Cycles renders). Batch mode via `GrepModelsStep` fan-out.

All steps imported from `lib/pipeline_steps/` — exercise script is pure composition + CLI (~170 lines).

### Ex 3 — Procedural Camera — COMPLETE

See `src/ex_3__proc_camera/Plan_of_Record.md` for full implementation details.

Generates procedural camera flythrough animations in two modes (spline / point-to-point), renders to 640x480 MP4 at 30 fps via Cycles.

**Key design decision:** Uses `OpenBlendStep` (calls `open_blend()`) instead of `PrepareSceneStep` + `ImportModelStep`. Evermotion `.blend` scenes are fully-dressed environments — `open_blend()` preserves lighting, materials, and world settings that `import_model()` would strip.

**Pipeline structure:**
```
proc_camera_e2e
├── scene_prep
│   └── OpenBlendStep             → io.open_blend() (preserves full scene)
├── camera_animation
│   ├── GenerateCameraPathStep    → random points in AABB, create camera
│   └── SetupAnimationStep        → mode A (bezier + follow path) or mode B (linear + look-at)
├── render_video
│   ├── ConfigureRendererStep     → reused, requires={"animation_configured"}
│   ├── ConfigureVideoOutputStep  → shared (lib/pipeline_steps/render.py)
│   └── RenderAnimationStep       → shared (lib/pipeline_steps/render.py)
├── write_log
│   └── WriteCameraLogStep        → JSON with mode, seed, control points
└── cleanup
    └── CleanupSceneStep          → reused
```

**Exercise-specific steps (in `proc_camera.py`):** `OpenBlendStep`, `GenerateCameraPathStep`, `SetupAnimationStep`, `WriteCameraLogStep`.

**Shared steps added to `lib/pipeline_steps/render.py`:** `ConfigureVideoOutputStep`, `RenderAnimationStep` — reusable by Ex 5.

**`ConfigureRendererStep` requires override:** The shared step defaults to `requires=["lighting_ready"]` for Ex 2's chain. Ex 3 passes `requires={"animation_configured"}` at construction time — no class modification needed.

**Blender 5.0 compat fix:** `ImageFormatSettings.media_type` must be set to `"VIDEO"` before `file_format = "FFMPEG"`. Fixed in `lib/bpy/render.py:configure_video_output()` with `hasattr` guard.

### Ex 4 — Batch Validator — NOT STARTED

Create `src/ex_4__batch_validator/batch_validator.py`

- Reuse `GrepModelsStep` from Ex 1 for fan-out
- Embed Ex 1's `build_e2e_pipeline()` as the per-model composite step
- Consider `SubprocessScheduler` for memory isolation on 2k+ file batches (bpy leaks memory across iterations even with `purge_orphan_data()`)
- Handle pagination (>1000 files), error resilience via `continue_on_error=True`
- Shared bpy calls: same as Ex 1

### Ex 5 — HPC Pipeline — NOT STARTED

Create `src/ex_5__hpc_pipeline/process_batch.py` + `submit_jobs.sh`

- `filter_animated → normalize → configure_renderer → orbit_animation → render_frames` composed pipeline
- Implement `SLURMScheduler.execute()` (currently a stub in `lib/scheduler.py`)
- Shared bpy calls: `io.open_blend()`, `animation.is_animated()`, `scene.move_to_origin()`, `camera.create_camera()`, `camera.add_track_to_location()`, `animation.create_orbit()`, `render.configure_cycles()`, `render.enable_gpu()`, `render.setup_depth_pass()`, `render.render_frame()`
- SLURM: 1 node, 1 H100, 12h walltime

---

## 11. Critical Lessons for Implementing New Exercises

These are hard-won lessons from Ex 1 and Ex 2 that apply to all subsequent exercises:

1. **Extend `BlenderStep`, not `PipelineStep`, for bpy steps.** All steps that interact with `bpy` must extend `BlenderStep` and implement `execute()` (not `run()`). This gives automatic `.blend` scene snapshot saving after each step for debugging. Pure Python steps (file I/O, scanning) extend `PipelineStep` and implement `run()` as before.

2. **Depth-first execution for bpy batch processing.** When using `GeneratorStep` fan-out with bpy-dependent steps, wrap the entire per-model pipeline as a single `Pipeline` step. Do NOT place individual bpy steps at the same level as the `GeneratorStep` — this causes breadth-first execution which corrupts the shared bpy scene between models.

3. **Deferred bpy imports.** Place `from lib.bpy.* import ...` inside `execute()` methods, not at module top level. This keeps the module importable for pipeline composition and unit testing without bpy.

4. **Every `provides` key must be set.** If your step declares `provides=["key"]`, you MUST set `context["key"]` in `execute()`. The validation middleware will fail the step otherwise.

5. **Use `OutputNamer` for batch output.** Per-model subdirectories (`{output_dir}/{stem}/output.ext`) prevent collisions and support multiple output files per model.

6. **Use `ensure_directory()` before writing files.** Output directories may not exist. Call `ensure_directory(output_path)` before `write_json_atomic()` or any file write.

7. **Package imports, not sys.path.** After `pip install -e .`, use `from lib.pipeline import ...` directly. No `sys.path` hacks.

8. **Blender 5.0 API changes.** The `bpy` module in Blender 5.0 has breaking changes: `scene.node_tree` became `scene.compositing_node_group`, `CompositorNodeComposite` was removed, `CompositorNodeOutputFile.base_path` became `directory`, `Material.use_nodes` is deprecated, and `ImageFormatSettings.media_type` must be set to `"VIDEO"` before setting `file_format = "FFMPEG"` for video output. See `lib/bpy/render.py` for compatibility helpers (`_ensure_compositor_tree`, `_get_compositor_tree`, `media_type` guard in `configure_video_output`).

9. **Scene snapshots to `.pipeline/` directory.** `BlenderStep` automatically saves `.blend` files to `{output_path}/.pipeline/after_{step_name}.blend`. Disable for production with `context["save_scenes"] = False`. Customize location with `context["artifacts_dir"] = "/custom/path/"`.

10. **Reuse steps from `lib/pipeline_steps/`, don't redefine them.** 15 pre-built steps are available. Exercise scripts should only define exercise-specific steps (e.g., custom report formats). Import shared steps from the library.

11. **`open_blend()` vs `import_model()` for .blend files.** `import_model()` appends objects only — it strips world settings, lighting, and cameras. `open_blend()` replaces the entire scene, preserving everything. For fully-dressed scenes (Evermotion interiors), use `open_blend()`. For importing objects into a clean scene, use `import_model()`.

12. **`ConfigureRendererStep` requires is overridable.** The shared step defaults to `requires=["lighting_ready"]` (Ex 2's chain). For exercises where lighting is baked into the scene, override at construction: `ConfigureRendererStep(requires={"animation_configured"})`. No class modification needed.
