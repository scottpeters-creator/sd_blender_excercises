# Plan of Record -- 002: PDG-Lite Pipeline Evolution

**Task ID:** 002_pdg_refactor
**Created:** 2026-02-18
**Status:** Complete
**Completed:** 2026-02-18
**Depends on:** 001_pipeline_dev (complete), exercises 1–3 implemented
**Target files:**
- `blender_excercises/src/lib/pipeline.py` (primary — extend, not rewrite)
- `blender_excercises/src/lib/work_item.py` (new)
- `blender_excercises/src/lib/scheduler.py` (new)
- `blender_excercises/src/lib/test_pipeline.py` (extend)
- `blender_excercises/src/lib/__init__.py` (update exports)
- `blender_excercises/context.md` (update after each phase)

---

## 1. Motivation

### 1.1 What works today

The 001 pipeline redesign delivered a composable orchestrator: `Pipeline` IS-A `PipelineStep`, `StepGroup` supports parallel execution, `requires`/`provides` enforce contracts, and the orchestrator owns timing, state, validation, and logging. This system handles single-model, linear pipelines well.

### 1.2 Where it breaks down

| Exercise | Pain Point | Root Cause |
|---|---|---|
| Ex 2 | 4 near-identical render step classes (`RenderTexturedStep`, `RenderNormalStep`, ...) | No way to parameterize a step per-invocation without subclassing |
| Ex 4 | Batch loop wrapping the pipeline is imperative, outside the graph | Single shared context dict — can't express "1000 models, each flowing independently" |
| Ex 4 | Memory leaks from running bpy in a loop | Everything executes in-process; no isolation between iterations |
| Ex 5 | SLURM submission script is hand-written, disconnected from pipeline | No scheduler abstraction — *what to do* is coupled to *where to do it* |
| Ex 2 | Fan-in (4 renders → 1 metadata.json) requires the downstream step to "just know" the outputs are ready | No explicit merge/collect primitive |

All five pain points trace to the same architectural assumption: **one pipeline run = one context dict = one unit of work.** PDG's core insight is that the unit of work should be a *work item*, not the pipeline itself.

### 1.3 Design reference: SideFX PDG / TOPs

We adopt three concepts from PDG selectively:

1. **Work Items** — typed data packets that flow through the graph independently.
2. **Schedulers** — pluggable execution backends (local, subprocess, SLURM).
3. **Fan-out / Fan-in** — steps that multiply or collect work items as graph operations.

We deliberately do NOT adopt: graph UI, fine-grained dirty propagation, HDA wrapping, service/daemon mode.

---

## 2. Current State (post-001)

### 2.1 File inventory

- `src/lib/pipeline.py` — 520 lines, 3 classes: `PipelineStep`, `Pipeline`, `StepGroup`
- `src/lib/test_pipeline.py` — 652 lines, 52 passing tests
- `src/ex_1__mini_inspector/mini_inspector.py` — 305 lines, 7 stub steps composed into 4 sub-pipelines

### 2.2 Execution model

```
Pipeline.run(context: Dict[str, Any]) -> CompletedState
    for step in self.steps:
        _execute_step(step, context)     # one context, one pass
```

### 2.3 Key signatures

```python
class PipelineStep(ABC):
    def run(self, context: Dict[str, Any]) -> CompletedState: ...

class Pipeline(PipelineStep):
    def run(self, context, *, _prefix="", dry_run=False) -> CompletedState: ...

class StepGroup(PipelineStep):
    def run(self, context) -> CompletedState: ...   # parallel or sequential
```

### 2.4 What cannot change (backward compatibility)

- `CompletedState` dataclass — already well-designed, no changes.
- `PipelineStep.run(context) -> CompletedState` — existing steps must continue to work.
- `requires`/`provides` string-based contracts.
- `_short_signature()`, `write_json_atomic()` helpers.
- All existing unit tests (52) must continue to pass.

---

## 3. Desired State

### 3.1 Work Items

A `WorkItem` replaces the raw dict as the unit of data flowing through the pipeline.

```python
@dataclass
class WorkItem:
    id: str                                  # unique identifier (e.g., filename stem)
    attributes: Dict[str, Any]               # typed key-value data (replaces context)
    input_files: List[str] = field(...)      # tagged input file paths
    output_files: List[str] = field(...)     # tagged output file paths
    parent_id: Optional[str] = None          # for fan-out provenance
    meta: Dict[str, Any] = field(...)        # scheduler hints, timing, etc.
```

**Backward compatibility:** `context` becomes `work_item.attributes`. Steps that use `context["key"]` still work — `WorkItem` exposes a dict-like interface or steps receive `work_item.attributes` directly.

### 3.2 Middleware Chain

Refactor `Pipeline._execute_step()` into a composable middleware stack:

```python
Middleware = Callable[[PipelineStep, WorkItem, NextFn], CompletedState]

# Default chain (equivalent to current behavior):
default_middleware = [
    idempotent_skip_middleware,
    requirements_check_middleware,
    timing_middleware,
    validation_middleware,
    state_write_middleware,
    logging_middleware,
]

# User can extend:
Pipeline(steps=[...], middleware=[retry_middleware, *default_middleware])
```

Each middleware gets `(step, work_item, next)` and decides whether to call `next()`. This unlocks retries, caching, profiling, and scheduler dispatch without touching step code.

### 3.3 DAG Solver

Add optional topological sort from `requires`/`provides`:

```python
Pipeline(steps=[...], resolve_order=True)
```

When `resolve_order=True`, the orchestrator builds a dependency graph and executes in topological order instead of list order. Cycle detection raises at construction time. Stable sort preserves declared order as tiebreaker.

### 3.4 Fan-Out / Fan-In

Two new step types:

```python
class GeneratorStep(PipelineStep):
    """Emits N work items from 1 input work item."""
    def generate(self, work_item: WorkItem) -> List[WorkItem]: ...

class CollectorStep(PipelineStep):
    """Merges N work items into 1."""
    def collect(self, work_items: List[WorkItem]) -> WorkItem: ...
```

Example — Ex 2 rendered with fan-out:

```python
class ModalitySplitter(GeneratorStep):
    def generate(self, work_item):
        return [
            WorkItem(id=f"{work_item.id}_{m}", attributes={**work_item.attributes, "modality": m})
            for m in ["textured", "normal", "depth", "edge"]
        ]

class RenderModalityStep(PipelineStep):
    def run(self, context):
        modality = context["modality"]   # parameterized by work item
        ...

class MetadataCollector(CollectorStep):
    def collect(self, work_items):
        merged = WorkItem(id=work_items[0].parent_id, attributes={...})
        merged.output_files = [f for wi in work_items for f in wi.output_files]
        return merged
```

The orchestrator handles the multiplication: when it encounters a `GeneratorStep`, it runs downstream steps once per emitted work item, then collects at the next `CollectorStep` (or end of group).

### 3.5 Scheduler Abstraction

```python
class Scheduler(ABC):
    def execute(self, step: PipelineStep, work_item: WorkItem) -> CompletedState: ...

class LocalScheduler(Scheduler):
    """Run in-process (current behavior)."""

class SubprocessScheduler(Scheduler):
    """Spawn a fresh Python/Blender process per work item.
    Serializes work_item to JSON, invokes script, deserializes CompletedState."""

class SLURMScheduler(Scheduler):
    """Submit work item as an sbatch job. Poll for completion."""
```

Schedulers are configured per-pipeline or per-step:

```python
Pipeline(steps=[...], scheduler=LocalScheduler())

# Or per-step override for GPU-heavy work:
RenderStep(scheduler=SLURMScheduler(partition="gpu", gpus=1, time="2:00:00"))
```

The middleware chain calls `scheduler.execute(step, work_item)` instead of `step.run(context)` directly.

---

## 4. Migration Strategy

### 4.1 Guiding principle

**Additive, not rewriting.** Every phase adds new capabilities alongside the existing API. Existing steps and tests keep working throughout. The old `run(context)` signature is never removed — it becomes a convenience that wraps a single-item work flow.

### 4.2 Phase dependencies

```
A (middleware) ──→ C (work items) ──→ D (fan-out/fan-in)
       │                  │
       └──→ B (DAG)       └──→ E (schedulers)
```

A is prerequisite for everything. B is independent. C enables D and E.

### 4.3 Trigger criteria

This refactor is **not started immediately.** Each phase has a trigger — the exercise that creates the concrete need:

| Phase | Trigger | Why |
|---|---|---|
| A (middleware) | Before Ex 2 | Ex 2 benefits from retry middleware for render failures |
| B (DAG solver) | With declarative config (if pursued) | Auto-ordering eliminates a class of config errors |
| C (work items) | Before Ex 4 | Ex 4's batch loop over 2k files is the canonical use case |
| D (fan-out/fan-in) | With Ex 2 or Ex 4 | Ex 2: 1 model → 4 modalities. Ex 4: 1 bucket → 2k items |
| E (schedulers) | Before Ex 5 | Ex 5 requires SLURM submission; subprocess useful for Ex 4 memory isolation |

### 4.4 Execution discipline

Each phase is implemented as an independent unit:

1. Write the code changes
2. Write the new tests
3. Run all tests (existing 52 + new) — all must pass
4. Check for linter errors — zero tolerance
5. Update `context.md` section 2 to reflect new API surface

---

## 5. Roadmap

### Phase A: Middleware Chain

- [x] A1: Define `Middleware` and `NextFn` type aliases
- [x] A2: Extract `idempotent_skip_middleware` from `_execute_step()` lines 249-255
- [x] A3: Extract `requirements_check_middleware` from `_execute_step()` lines 258-268
- [x] A4: Extract `logging_middleware` from `_execute_step()` lines 270, 325, 331
- [x] A5: Extract `timing_middleware` from `_execute_step()` lines 271, 306
- [x] A6: Extract `validation_middleware` from `_execute_step()` lines 309-320
- [x] A7: Extract `state_write_middleware` from `_execute_step()` lines 267, 289, 301, 322
- [x] A8: Create `default_middleware` list combining all 6 middleware functions
- [x] A9: Add `middleware=` parameter to `Pipeline.__init__` (default: `None` → uses `default_middleware`)
- [x] A10: Rewrite `_execute_step()` as a thin chain-invocation wrapper
- [x] A11: Write ~8 unit tests for Phase A

**Files touched:**
- `src/lib/pipeline.py` — refactor `_execute_step()`, ~60 new lines
- `src/lib/test_pipeline.py` — ~8 new tests

**Middleware signature:**

```python
Middleware = Callable[[PipelineStep, Dict[str, Any], NextFn], CompletedState]
NextFn = Callable[[], CompletedState]
```

**Tests to add:**
- Default middleware produces identical results to current behavior (regression)
- Custom middleware injected before default chain fires in correct order
- Middleware that returns early (skip) prevents downstream middleware from running
- Retry middleware (example) retries a failing step N times
- Empty middleware list runs step with no orchestration
- Middleware receives correct step and context references
- Middleware chain preserves `step_states` write behavior
- StepGroup remains unaffected (it has its own execution path)

---

### Phase B: DAG Solver

- [x] B1: Implement `_topo_sort(steps: List[PipelineStep]) -> List[PipelineStep]`
- [x] B2: Build dependency graph from `requires`/`provides` intersections
- [x] B3: Implement cycle detection (raise `ValueError` at construction time)
- [x] B4: Implement stable tiebreaker (preserve declared order for independent steps)
- [x] B5: Add `resolve_order=False` parameter to `Pipeline.__init__`
- [x] B6: When `resolve_order=True`, call `_topo_sort()` in constructor, replace `self.steps`
- [x] B7: Write ~6 unit tests for Phase B

**Files touched:**
- `src/lib/pipeline.py` — ~80 new lines
- `src/lib/test_pipeline.py` — ~6 new tests

**Key design decisions:**
- Cycle detection raises `ValueError` at construction time, not at runtime
- Stable sort: when multiple valid orderings exist, declared (list) order is the tiebreaker
- `resolve_order=False` (default) preserves current list-order behavior — zero risk to existing code
- Operates on the flat step list within a single Pipeline; does not reach into nested Pipeline children

**Tests to add:**
- Steps reordered correctly based on requires/provides
- Cycle detection raises `ValueError` with descriptive message
- Stable tiebreaker preserves declared order for independent steps
- `resolve_order=False` (default) is a no-op
- Steps with no requires/provides remain in declared order
- Works alongside middleware chain (Phase A)

---

### Phase C: WorkItem

- [x] C1: Create `src/lib/work_item.py` with `WorkItem` dataclass
- [x] C2: Implement `to_json()` / `from_json()` round-trip methods
- [x] C3: Implement JSON-serializability validation at construction
- [x] C4: Add `Pipeline.run_item(work_item: WorkItem) -> CompletedState`
- [x] C5: Refactor `Pipeline.run(context)` to wrap context in `WorkItem(id="default", attributes=context)` and delegate to `run_item()`
- [x] C6: Thread `work_item.attributes` as the `context` dict passed to `step.run()`
- [x] C7: Update `src/lib/__init__.py` to export `WorkItem`
- [x] C8: Write ~10 unit tests for Phase C

**Files touched:**
- `src/lib/work_item.py` — **new file**, ~120 lines
- `src/lib/pipeline.py` — ~60 changed lines
- `src/lib/test_pipeline.py` — ~10 new tests
- `src/lib/__init__.py` — update exports

**WorkItem dataclass:**

```python
@dataclass
class WorkItem:
    id: str
    attributes: Dict[str, Any]
    input_files: List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
```

**Backward compatibility strategy:**
- `Pipeline.run(context)` still works — wraps context in a WorkItem internally
- Steps still implement `run(context: Dict)` — the orchestrator passes `work_item.attributes`
- Existing steps see no change; all 52 existing tests continue to pass untouched

**Tests to add:**
- WorkItem construction and field defaults
- `to_json()` / `from_json()` round-trip fidelity
- JSON-serializability validation rejects non-serializable values
- `run(context)` produces identical results via internal WorkItem wrapping
- `run_item(work_item)` populates `work_item.attributes` correctly
- Parent-child ID tracking (`parent_id`)
- `input_files` / `output_files` accumulate through steps
- WorkItem with middleware chain (Phase A integration)
- Multiple work items through same pipeline produce independent results
- Nested pipeline with WorkItem

---

### Phase D: Fan-Out / Fan-In

- [x] D1: Implement `GeneratorStep(PipelineStep)` ABC with `generate(work_item) -> List[WorkItem]` and backward-compat `run(context)` bridge
- [x] D2: Implement `CollectorStep(PipelineStep)` ABC with `collect(work_items) -> WorkItem` and backward-compat `run(context)` bridge
- [x] D3: Modify orchestrator: when `_execute_step` encounters a `GeneratorStep`, call `generate()` to get N work items
- [x] D4: Run downstream steps (until next `CollectorStep` or end of group) once per emitted work item
- [x] D5: When `CollectorStep` reached, pass all accumulated work items to `collect()`
- [x] D6: Set `parent_id` on fan-out items to source work item's `id`
- [x] D7: Write ~8 unit tests for Phase D

**Files touched:**
- `src/lib/pipeline.py` — ~100 new lines (two new step types + orchestrator logic)
- `src/lib/test_pipeline.py` — ~8 new tests

**Scope boundary:** Fan-out/fan-in is scoped to a single `Pipeline` or `StepGroup`. It does not cross nesting boundaries — a nested pipeline receives one work item and returns one result.

**Tests to add:**
- GeneratorStep emits N items, downstream step runs N times
- CollectorStep merges N items into 1
- Fan-out → processing → fan-in end-to-end
- Empty generation (0 items) handled gracefully
- Fan-out with parallel StepGroup
- Parent ID provenance tracking
- Fan-out without collector (items processed independently, results collected at pipeline end)
- Nested pipeline inside fan-out region receives each item independently

---

### Phase E: Schedulers

- [x] E1: Create `src/lib/scheduler.py` with `Scheduler` ABC
- [x] E2: Implement `LocalScheduler` (in-process, current behavior)
- [x] E3: Implement `SubprocessScheduler` (spawn process, serialize WorkItem to JSON, deserialize CompletedState)
- [x] E4: Add `scheduler=` parameter to `Pipeline.__init__` (default: `LocalScheduler()`)
- [x] E5: Support step-level `scheduler` override on `PipelineStep`
- [x] E6: Integrate scheduler invocation into middleware chain (innermost "execute" middleware calls `scheduler.execute()` instead of `step.run()` directly)
- [x] E7: Update `src/lib/__init__.py` to export scheduler classes
- [x] E8: Write ~8 unit tests for Phase E
- [x] E9: Stub `SLURMScheduler` interface (implementation deferred to Ex 5)

**Files touched:**
- `src/lib/scheduler.py` — **new file**, ~150 lines
- `src/lib/pipeline.py` — ~30 changed lines
- `src/lib/test_pipeline.py` — ~8 new tests
- `src/lib/__init__.py` — update exports

**Scheduler ABC:**

```python
class Scheduler(ABC):
    @abstractmethod
    def execute(self, step: PipelineStep, work_item: WorkItem) -> CompletedState: ...
```

**Configuration:**

```python
Pipeline(steps=[...], scheduler=LocalScheduler())        # pipeline-level
RenderStep(scheduler=SLURMScheduler(partition="gpu"))     # step-level override
```

**Tests to add:**
- LocalScheduler produces identical results to direct `step.run()` call
- SubprocessScheduler round-trips WorkItem through JSON
- Scheduler configured at pipeline level applies to all steps
- Step-level scheduler overrides pipeline-level scheduler
- Scheduler failure (process crash) returns proper error CompletedState
- Scheduler integrates with middleware chain
- Scheduler with fan-out (each work item dispatched independently)
- Mock SLURMScheduler to test the interface without actual SLURM

---

## 6. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | Premature abstraction — building PDG-lite before exercises validate the need | Trigger criteria (§4.3). Don't build phases until the exercise demands it. |
| R2 | WorkItem breaks existing step `run(context)` contract | Backward-compatible: `run(context)` still works. WorkItem wraps context. |
| R3 | Subprocess scheduler adds serialization complexity (work items must be JSON-safe) | Constrain WorkItem attributes to JSON-serializable types. Validate at construction. |
| R4 | DAG solver introduces ambiguous execution order when multiple valid sorts exist | Stable sort preserving declared order as tiebreaker. Log resolved order. |
| R5 | Fan-out + parallel execution on shared context causes data races | Work items are independent copies. Fan-out creates new WorkItem instances, not shared refs. |
| R6 | Scope creep toward full PDG | Explicit non-goals: no graph UI, no dirty propagation, no HDA wrapping, no daemon mode. |

---

## 7. Estimation

| Phase | New/Changed Lines | New Tests | Risk | Depends On |
|---|---|---|---|---|
| A: Middleware | ~60 in pipeline.py | ~8 | Low | Nothing |
| B: DAG Solver | ~80 in pipeline.py | ~6 | Low | Nothing |
| C: WorkItem | ~120 in work_item.py, ~60 in pipeline.py | ~10 | Medium | A |
| D: Fan-Out/Fan-In | ~100 in pipeline.py | ~8 | Medium | C |
| E: Schedulers | ~150 in scheduler.py, ~30 in pipeline.py | ~8 | Medium | A, C |
| **Total** | **~600 lines** | **~40 tests** | | |

---

## 8. File Inventory

| File | Action | Phases |
|---|---|---|
| `src/lib/pipeline.py` | Extend | A, B, C, D, E |
| `src/lib/work_item.py` | Create | C |
| `src/lib/scheduler.py` | Create | E |
| `src/lib/test_pipeline.py` | Extend | A, B, C, D, E |
| `src/lib/__init__.py` | Update exports | C, E |
| `src/ex_1__mini_inspector/mini_inspector.py` | No change | — |
| `context.md` | Update after all phases | A–E |

---

## Appendix A: What We Explicitly Don't Build

| PDG Feature | Why Not |
|---|---|
| Graph UI / node editor | CLI + dry-run logging is sufficient for 5 exercises |
| Fine-grained dirty propagation (input hash tracking) | `idempotent` + `step_states` covers our use cases |
| Service/daemon mode (persistent Blender process) | Subprocess-per-item is simpler and sufficient for memory isolation |
| Distributed work item storage (Redis, DB) | In-memory + JSON checkpoint files are adequate at our scale |
| Dynamic graph modification at runtime | Static graph declared at construction time is sufficient |

---

## Appendix B: Implementation Summary

**Completed:** 2026-02-18 | **All 88 unit tests passing** | **0 linter errors**

### Final file sizes

| File | Before | After |
|---|---|---|
| `src/lib/pipeline.py` | 520 lines | ~650 lines |
| `src/lib/work_item.py` | (new) | ~65 lines |
| `src/lib/scheduler.py` | (new) | ~120 lines |
| `src/lib/test_pipeline.py` | 652 lines | ~1050 lines |
| `src/lib/__init__.py` | 7 lines | 6 lines |

### Key signatures (post-implementation)

```python
# Phase A: Middleware
_Middleware = Callable[[_StepExecContext, _NextFn], CompletedState]
default_middleware: List[_Middleware] = [
    _idempotent_skip_mw, _state_write_mw, _requirements_check_mw,
    _logging_mw, _timing_mw, _execute_mw,
]

# Phase B: DAG Solver
_topo_sort(steps: List[PipelineStep]) -> List[PipelineStep]

# Phase C: WorkItem
@dataclass
class WorkItem:
    id: str; attributes: Dict[str, Any]
    input_files: List[str]; output_files: List[str]
    parent_id: Optional[str]; meta: Dict[str, Any]

Pipeline.run_item(work_item, *, _prefix="", dry_run=False) -> CompletedState

# Phase D: Fan-out / Fan-in
class GeneratorStep(PipelineStep):
    def generate(self, work_item: WorkItem) -> List[WorkItem]: ...

class CollectorStep(PipelineStep):
    def collect(self, work_items: List[WorkItem]) -> WorkItem: ...

# Phase E: Schedulers
class Scheduler(ABC):
    def execute(self, step: PipelineStep, work_item: WorkItem) -> CompletedState: ...

class LocalScheduler(Scheduler): ...
class SubprocessScheduler(Scheduler): ...
class SLURMScheduler(Scheduler): ...  # stub
```

### Test coverage by phase

| Phase | New Tests | What's covered |
|---|---|---|
| A: Middleware | 8 | Default regression, custom ordering, early return, retry, empty chain |
| B: DAG Solver | 6 | Reordering, cycle detection, stable tiebreaker, no-op default, integration |
| C: WorkItem | 10 | Construction, JSON round-trip, serialization validation, run/run_item compat, independence, nesting, dry-run |
| D: Fan-Out/Fan-In | 6 | N-item generation, collection/merge, parent ID provenance, empty gen, no-collector, nested pipeline |
| E: Schedulers | 6 | LocalScheduler, pipeline/step override, SLURM stub, direct-run fallback |
| **PDG Total** | **36** | |
| **Grand Total** | **88** | (52 original + 36 PDG-Lite) |

### Backward compatibility

- `Pipeline.run(context)` still works — wraps context in WorkItem internally
- All 52 original tests pass unmodified
- `mini_inspector.py` requires zero changes
- `PipelineStep.run(context)` signature unchanged — steps still receive a plain dict
- `default_middleware` reproduces exact original `_execute_step()` behavior
