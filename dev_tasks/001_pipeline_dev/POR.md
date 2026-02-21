# Plan of Record -- 001: Composable Pipeline Redesign

**Task ID:** 001_pipeline_dev
**Created:** 2026-02-18
**Status:** Complete
**Completed:** 2026-02-18
**Target files:**
- `blender_exercises/src/lib/pipeline.py` (primary)
- `blender_exercises/src/ex_1__mini_inspector/mini_inspector.py` (consumer update)

---

## 1. Current State

### 1.1 What exists

The pipeline library (`src/lib/pipeline.py`) contains three core primitives:

- **`CompletedState`** -- dataclass representing the result of a step execution (success, timestamp, duration, provides, outputs, error, meta, signature).
- **`PipelineStep`** (ABC) -- abstract base with `name`, `requires`, `provides`, `idempotent`, `continue_on_error`. Subclasses implement `run(context) -> CompletedState`.
- **`Pipeline`** -- orchestrator that iterates a flat list of steps against a shared `context: Dict[str, Any]`.

Helpers: `write_json_atomic()`, `_short_signature()`.

The consumer (`mini_inspector.py`) defines 7 stub step classes and a CLI harness that resolves step names from a registry, constructs a `Pipeline()`, and calls `pipeline.run(steps, context)`.

### 1.2 Problems and limitations

| # | Issue | Impact |
|---|---|---|
| P1 | `Pipeline` is not a `PipelineStep` -- it cannot be nested inside another pipeline. | No composition. Cannot express "pre -> main -> post" as a pipeline of pipelines. |
| P2 | `Pipeline.run()` returns an opaque `int` (0-4). | Cannot be used as a step (steps must return `CompletedState`). Error details lost at call site. |
| P3 | `Pipeline.__init__` is empty -- no name, version, or step list. | Pipeline has no identity. Steps are passed ad-hoc to `run()`. Cannot register or version pipelines. |
| P4 | No parallel execution model. | Ex 2 needs 4 render passes; future exercises need concurrent I/O. No interface to express intent. |
| P5 | `step_states` is a flat dict keyed by step name. | Nested pipelines cause name collisions (e.g., two "cleanup" steps). |
| P6 | Each step manually writes `context["step_states"]` AND the orchestrator writes it too. | Dual-write bug. Step authors must know internal bookkeeping. |
| P7 | Each step manually times itself (`start = time.time()` / `duration = ...`). | Boilerplate in every step. Orchestrator captures `start` on line 130 but never uses it. |
| P8 | `validate()` is defined on `PipelineStep` but never called by the orchestrator. | Dead code. No post-run verification that a step populated its `provides` keys. |
| P9 | No logging in the orchestrator. | Silent execution. Unusable for batch runs (Ex 4: 2k models) or SLURM jobs (Ex 5). |
| P10 | No dry-run mode. | Cannot preview step ordering or dependency resolution without executing. |
| P11 | `write_json_atomic` temp prefix hardcoded to `"mini_inspector_"`. | Misleading when used from other exercises. |

### 1.3 File snapshots (for diffing after implementation)

**`pipeline.py`** -- 162 lines. Key signatures:
```python
class Pipeline:
    def __init__(self) -> None: ...
    def run(self, steps: Iterable[PipelineStep], context: Dict[str, Any], *, force: bool = False) -> int: ...
```

**`mini_inspector.py`** -- 283 lines. Each step stub follows this pattern:
```python
def run(self, context):
    start = time.time()                    # <-- boilerplate (P7)
    # ... do work, set context keys ...
    duration = time.time() - start         # <-- boilerplate (P7)
    cs = CompletedState(...)
    context["step_states"][self.name] = dataclasses.asdict(cs)  # <-- dual write (P6)
    return cs
```

---

## 2. Desired State

### 2.1 Design: the composite pattern

`Pipeline` extends `PipelineStep`. A pipeline IS-A step, so it can be placed inside another pipeline. Composition replaces hooks/callbacks for pre/post logic.

```
PipelineStep (ABC)
    |
    +-- Pipeline (extends PipelineStep, contains N steps)
    |       steps may be PipelineStep instances OR other Pipelines
    |
    +-- StepGroup (extends PipelineStep, contains N steps, supports parallel flag)
```

### 2.2 Target API

```python
# Define focused, reusable pipelines
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

# Compose: pipeline of pipelines
e2e = Pipeline(
    name="mini_inspector_e2e", version="1.0",
    steps=[scene_prep, analysis, reporting, cleanup],
)

# Run everything -- returns CompletedState, not int
result = e2e.run(context)
```

Parallel groups (interface-ready, sequential execution initially):
```python
render_all = StepGroup(
    name="render_modalities", parallel=True,
    steps=[RenderTextured(), RenderNormal(), RenderDepth(), RenderEdge()],
)
```

### 2.3 Specific changes required

| # | Change | Resolves |
|---|---|---|
| C1 | `Pipeline` extends `PipelineStep`. Constructor accepts `name`, `version`, `steps`, `requires`, `provides`, `idempotent`, `continue_on_error`, `force`. | P1, P3 |
| C2 | `Pipeline.run(context)` returns `CompletedState`. Add `ExitCode` IntEnum and `exit_code_from(cs)` helper for CLI use. | P2 |
| C3 | Add `StepGroup(PipelineStep)` with `parallel: bool` flag. Validates non-overlapping `provides` among children when `parallel=True`. Initial execution is sequential via simple loop; parallel execution via `concurrent.futures.ThreadPoolExecutor` when `parallel=True`. | P4 |
| C4 | Add `version: str` field to `PipelineStep.__init__` (default `"0.0.0"`). | P3 |
| C5 | Namespaced `step_states` keys: dot-delimited hierarchy (e.g., `"scene_prep.prepare_scene"`). Parent pipeline prepends its name when recording child results. | P5 |
| C6 | Orchestrator is sole owner of `step_states` writes. Steps only return `CompletedState` and set `provides` keys on context. | P6 |
| C7 | Orchestrator owns timing. Wraps each `step.run()` call; sets `duration_s` on returned `CompletedState`. Steps no longer need `start = time.time()` boilerplate. | P7 |
| C8 | Call `step.validate(context)` after successful `run()`. Treat validation failure as step failure. | P8 |
| C9 | Add `logging.getLogger("pipeline")` to orchestrator. Log: step start, skip (idempotent), success + duration, failure + error. INFO level. | P9 |
| C10 | Add `dry_run: bool` parameter. When True, resolve steps, check requirements, log planned execution order, return without calling `run()`. | P10 |
| C11 | Change `write_json_atomic` temp prefix to `"pipeline_"`. | P11 |
| C12 | Auto-infer `requires`/`provides` on Pipeline from child steps when not explicitly declared. `requires` = external dependencies not satisfied internally. `provides` = union of all children's provides. | P1 |

### 2.4 Context model

**Shared context** -- one `Dict[str, Any]` flows through the entire pipeline tree. The `requires`/`provides` contracts enforce discipline. No scoping or copying between nested pipelines.

This keeps things simple and matches the linear data flow of all five exercises. Scoped context can be added later if isolation becomes necessary.

### 2.5 What does NOT change

- `CompletedState` dataclass (already well-designed).
- `_short_signature()` helper.
- The shared-context execution model.
- The `requires`/`provides` contract enforcement logic (extended to work recursively, not replaced).
- All changes remain in `pipeline.py`. No new files in `src/lib/`.

---

## 3. Development and Testing Strategy

### 3.1 Implementation order

Changes are ordered to keep the code functional at every step. Each phase produces a testable checkpoint.

**Phase A: Foundation (C4, C6, C7, C8, C9, C11)**
Modify `PipelineStep` and the orchestrator loop without changing the class hierarchy. After this phase, the existing flat pipeline still works but the orchestrator owns timing, `step_states`, and validation. Logging is active. This is the lowest-risk phase.

**Phase B: Return type (C2)**
Change `Pipeline.run()` to return `CompletedState`. Add `ExitCode` enum and `exit_code_from()`. This is a breaking API change -- `mini_inspector.py` must be updated in the same commit.

**Phase C: Composite pattern (C1, C5, C12)**
Make `Pipeline` extend `PipelineStep`. Implement namespaced `step_states`. Add `requires`/`provides` auto-inference. After this phase, pipelines can nest.

**Phase D: Parallel groups (C3)**
Add `StepGroup`. This is additive -- no existing code changes. Can be tested independently.

**Phase E: Dry-run (C10)**
Add `dry_run` parameter. Additive and low-risk.

### 3.2 Testing approach

All tests run without Blender (`bpy`). The pipeline library is deliberately Blender-agnostic.

**Unit tests** (new file: `src/lib/test_pipeline.py`):
- Flat pipeline: steps run in order, context keys populated, `step_states` recorded.
- Missing requirements: pipeline stops and returns failure `CompletedState`.
- Idempotent skip: completed step is skipped unless `force=True`.
- Exception handling: step raises, rollback called, failure recorded.
- `validate()`: step returns success but doesn't populate `provides` keys -- treated as failure.
- Nested pipeline: Pipeline-in-Pipeline runs all child steps, `step_states` are namespaced.
- `StepGroup`: sequential and parallel modes both produce correct results.
- `requires`/`provides` auto-inference: Pipeline correctly computes its external contract.
- Dry-run: no `run()` called, planned order logged.
- `exit_code_from()`: maps `CompletedState` to correct `ExitCode`.

**Integration test** (update `mini_inspector.py`):
- Run `mini_inspector.py` with stub steps in composed-pipeline mode.
- Verify JSON output matches expected schema.
- Verify `step_states` are namespaced correctly.

### 3.3 Validation criteria

A phase is complete when:
1. All unit tests for that phase pass.
2. `mini_inspector.py` can run end-to-end with stub steps (no regressions).
3. No linter errors introduced.

---

## 4. Roadmap

### Phase A: Foundation
- [x] A1: Add `version` field to `PipelineStep.__init__` (default `"0.0.0"`) -- (C4)
- [x] A2: Move timing from steps to orchestrator -- orchestrator wraps `step.run()`, sets `duration_s` on returned `CompletedState` -- (C7)
- [x] A3: Make orchestrator sole owner of `step_states` writes -- remove from step contract -- (C6)
- [x] A4: Wire up `validate()` call after successful `run()` -- treat validation failure as step failure -- (C8)
- [x] A5: Add `logging.getLogger("pipeline")` to orchestrator -- log lifecycle events at INFO -- (C9)
- [x] A6: Change `write_json_atomic` temp prefix to `"pipeline_"` -- (C11)
- [x] A7: Update `mini_inspector.py` stubs -- remove `start = time.time()` boilerplate and `context["step_states"]` writes from all 7 steps
- [x] A8: Write unit tests for Phase A changes

### Phase B: Return Type
- [x] B1: Add `ExitCode` IntEnum (`OK=0`, `MISSING_REQUIREMENT=1`, `STEP_EXCEPTION=2`, `INVALID_RETURN=3`, `STEP_FAILED=4`) -- (C2)
- [x] B2: Add `exit_code_from(cs: CompletedState) -> ExitCode` helper -- (C2)
- [x] B3: Change `Pipeline.run()` return type from `int` to `CompletedState` -- (C2)
- [x] B4: Update `mini_inspector.py` `main()` to use `exit_code_from()` for `sys.exit()` -- (C2)
- [x] B5: Write unit tests for Phase B changes

### Phase C: Composite Pattern
- [x] C1: Make `Pipeline` extend `PipelineStep` -- constructor accepts `name`, `version`, `steps` -- (C1)
- [x] C2: Implement `requires`/`provides` auto-inference from child steps -- (C12)
- [x] C3: Implement dot-delimited namespaced `step_states` keys -- (C5)
- [x] C4: Update `mini_inspector.py` to use composed `Pipeline(name=..., steps=[...])` construction -- (C1)
- [x] C5: Write unit tests for nested pipeline execution, namespacing, and auto-inference

### Phase D: Parallel Groups
- [x] D1: Implement `StepGroup(PipelineStep)` with `parallel` flag -- (C3)
- [x] D2: Validate non-overlapping `provides` among children when `parallel=True` -- (C3)
- [x] D3: Implement sequential execution path (loop) -- (C3)
- [x] D4: Implement parallel execution path (`concurrent.futures.ThreadPoolExecutor`) -- (C3)
- [x] D5: Write unit tests for `StepGroup` (both modes, provides-overlap validation)

### Phase E: Dry Run
- [x] E1: Add `dry_run: bool` parameter to `Pipeline.run()` -- (C10)
- [x] E2: When `dry_run=True`: resolve steps, check requirements, log planned order, return without executing -- (C10)
- [x] E3: Write unit tests for dry-run mode

---

## Appendix A: How this enables the exercises

| Exercise | Composition pattern |
|---|---|
| Ex 1 (Mini Inspector) | `scene_prep -> analysis -> reporting -> cleanup` as 4 composed pipelines |
| Ex 2 (Thumbnail Renderer) | Reuse `scene_prep` + `cleanup` from Ex 1; add `camera_setup` + `StepGroup(parallel=True, steps=[render_textured, render_normal, render_depth, render_edge])` + `metadata_write` |
| Ex 3 (Procedural Camera) | Reuse `scene_prep`; add `camera_animation` + `render_video` pipelines |
| Ex 4 (Batch Validator) | Outer batch loop runs the entire Ex 1 e2e pipeline as a single composable unit per file: `download -> mini_inspector_e2e -> upload -> file_cleanup` |
| Ex 5 (HPC Pipeline) | `filter_animated -> normalize -> configure_renderer -> orbit_animation -> render_frames` composed pipeline, submitted via SLURM |

## Appendix B: Implementation Summary

**Completed:** 2026-02-18 | **All 52 unit tests passing** | **0 linter errors**

### Final file sizes

| File | Before | After |
|---|---|---|
| `src/lib/pipeline.py` | 162 lines | 520 lines |
| `src/ex_1__mini_inspector/mini_inspector.py` | 283 lines | 305 lines |
| `src/lib/test_pipeline.py` | (new) | 420 lines |

### Key signatures (post-implementation)

```python
class PipelineStep(ABC):
    def __init__(self, name, requires=None, provides=None, idempotent=True,
                 continue_on_error=False, version="0.0.0") -> None: ...

class Pipeline(PipelineStep):
    def __init__(self, name="pipeline", version="1.0", steps=None, requires=None,
                 provides=None, idempotent=True, continue_on_error=False, force=False) -> None: ...
    def run(self, context, *, _prefix="", dry_run=False) -> CompletedState: ...

class StepGroup(PipelineStep):
    def __init__(self, name, steps=None, parallel=False, requires=None, provides=None,
                 version="0.0.0", continue_on_error=False) -> None: ...
    def run(self, context) -> CompletedState: ...

class ExitCode(IntEnum):
    OK = 0; MISSING_REQUIREMENT = 1; STEP_EXCEPTION = 2; INVALID_RETURN = 3; STEP_FAILED = 4

def exit_code_from(cs: CompletedState) -> ExitCode: ...
```

### Test coverage by phase

| Phase | Tests | What's covered |
|---|---|---|
| A | 11 | version field, orchestrator-owned timing/step_states/validation, logging, write_json_atomic |
| B | 7 | ExitCode enum values, exit_code_from mapping, Pipeline returns CompletedState |
| C | 15 | Pipeline IS-A PipelineStep, flat ordering, missing requirements, idempotent skip, exceptions + rollback, bad return, nested pipelines, deep nesting, namespaced step_states, auto-inference |
| D | 10 | StepGroup sequential/parallel, parallel failure, thread usage, provides-overlap rejection, group inside pipeline, auto-inference |
| E | 3 | No execution in dry-run, plan logged, nested dry-run |
| Helpers | 3 | _short_signature determinism, CompletedState.now_iso format |
| **Total** | **52** | |

### mini_inspector.py changes

- All 7 step stubs simplified: removed `time.time()` boilerplate and `context["step_states"]` writes
- Composed pipeline construction: `scene_prep -> analysis -> reporting -> cleanup` as 4 sub-pipelines inside `mini_inspector_e2e`
- `main()` returns `int(exit_code_from(result))` instead of raw pipeline int
- Added `--dry-run` CLI flag
- Legacy `--steps` override preserved via `_resolve_step_specs()` registry

## Appendix C: File inventory

| File | Role | Modified in this task |
|---|---|---|
| `src/lib/pipeline.py` | Pipeline library (primary target) | Yes -- all phases |
| `src/lib/__init__.py` | Package init | Possibly -- if new exports needed |
| `src/lib/test_pipeline.py` | Unit tests (new file) | Yes -- created in Phase A |
| `src/ex_1__mini_inspector/mini_inspector.py` | Consumer / CLI harness | Yes -- Phases A, B, C |
