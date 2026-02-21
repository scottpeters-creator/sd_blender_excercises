"""Pipeline primitives: CompletedState, PipelineStep, Pipeline, and StepGroup.

Composable pipeline library. Pipeline IS-A PipelineStep, enabling nesting
(pipeline-of-pipelines). StepGroup supports parallel execution via
ThreadPoolExecutor.

PDG-Lite extensions (Phase A–E):
  - Middleware chain for composable step orchestration (Phase A)
  - DAG solver for automatic step ordering (Phase B)
  - WorkItem support via run_item() (Phase C)
  - GeneratorStep / CollectorStep for fan-out / fan-in (Phase D)
  - Scheduler integration for pluggable execution backends (Phase E)

This module is dependency-free and Blender-agnostic.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import datetime
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CompletedState:
    success: bool
    timestamp: str
    duration_s: float
    provides: List[str] = field(default_factory=list)
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    signature: Optional[str] = None

    @staticmethod
    def now_iso() -> str:
        return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class ExitCode(IntEnum):
    OK = 0
    MISSING_REQUIREMENT = 1
    STEP_EXCEPTION = 2
    INVALID_RETURN = 3
    STEP_FAILED = 4


def exit_code_from(cs: CompletedState) -> ExitCode:
    """Map a CompletedState to an ExitCode suitable for sys.exit()."""
    if cs.success:
        return ExitCode.OK
    err = cs.error or {}
    msg = err.get("message", "")
    if "missing requirements" in msg:
        return ExitCode.MISSING_REQUIREMENT
    if err.get("type"):
        return ExitCode.STEP_EXCEPTION
    if "invalid CompletedState" in msg:
        return ExitCode.INVALID_RETURN
    return ExitCode.STEP_FAILED


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class PipelineStep(ABC):
    """Abstract base for pipeline steps.

    Subclasses implement `run(context)` and may optionally override
    `validate` and `rollback`.
    """

    name: str
    version: str
    requires: Set[str]
    provides: Set[str]
    idempotent: bool
    continue_on_error: bool

    def __init__(
        self,
        name: str,
        requires: Optional[Iterable[str]] = None,
        provides: Optional[Iterable[str]] = None,
        idempotent: bool = True,
        continue_on_error: bool = False,
        version: str = "0.0.0",
        scheduler: Any = None,
    ) -> None:
        self.name = name
        self.version = version
        self.requires = set(requires or [])
        self.provides = set(provides or [])
        self.idempotent = idempotent
        self.continue_on_error = continue_on_error
        self._scheduler = scheduler

    @abstractmethod
    def run(self, context: Dict[str, Any]) -> CompletedState:  # pragma: no cover
        raise NotImplementedError()

    def validate(self, context: Dict[str, Any]) -> bool:
        for key in self.provides:
            if key not in context:
                return False
        return True

    def rollback(self, context: Dict[str, Any]) -> None:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short_signature(value: Any) -> str:
    try:
        raw = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        raw = str(value).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def write_json_atomic(path: str, data: Any) -> None:
    path = str(path)
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix="pipeline_", dir=d, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        shutil.move(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Phase A: Middleware infrastructure
# ---------------------------------------------------------------------------

@dataclass
class _StepExecContext:
    """Carries metadata through the middleware chain."""
    step: PipelineStep
    context: Dict[str, Any]
    step_key: str
    force: bool


_NextFn = Callable[[], CompletedState]
_Middleware = Callable[[_StepExecContext, _NextFn], CompletedState]


def _idempotent_skip_mw(ctx: _StepExecContext, next_fn: _NextFn) -> CompletedState:
    """Short-circuit if this step already succeeded and is idempotent."""
    prev = ctx.context.get("step_states", {}).get(ctx.step_key)
    if prev and prev.get("success") and not ctx.force and ctx.step.idempotent:
        logger.info("SKIP  %s (idempotent, already succeeded)", ctx.step_key)
        return CompletedState(**{
            k: prev[k] for k in CompletedState.__dataclass_fields__
        })
    return next_fn()


def _requirements_check_mw(ctx: _StepExecContext, next_fn: _NextFn) -> CompletedState:
    """Short-circuit with failure if step requirements are missing from context."""
    missing = [r for r in ctx.step.requires if r not in ctx.context]
    if missing:
        logger.error("FAIL  %s — missing requirements: %s", ctx.step_key, missing)
        return CompletedState(
            success=False,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            error={"message": f"missing requirements: {missing}"},
        )
    return next_fn()


def _logging_mw(ctx: _StepExecContext, next_fn: _NextFn) -> CompletedState:
    """Log START before execution, OK or FAIL after."""
    logger.info("START %s", ctx.step_key)
    cs = next_fn()
    if cs.success:
        logger.info("OK    %s (%.3fs)", ctx.step_key, cs.duration_s)
    else:
        err = cs.error or {}
        err_type = err.get("type")
        msg = err.get("message", "")
        if err_type:
            logger.error(
                "FAIL  %s — %s: %s (%.3fs)",
                ctx.step_key, err_type, msg, cs.duration_s,
            )
        elif "validation failed" in msg:
            logger.error("FAIL  %s — validation failed (%.3fs)", ctx.step_key, cs.duration_s)
        elif "invalid CompletedState" in msg:
            logger.error("FAIL  %s — invalid return type (%.3fs)", ctx.step_key, cs.duration_s)
        else:
            logger.error("FAIL  %s (%.3fs)", ctx.step_key, cs.duration_s)
    return cs


def _timing_mw(ctx: _StepExecContext, next_fn: _NextFn) -> CompletedState:
    """Wrap execution with wall-clock timing."""
    start = time.time()
    cs = next_fn()
    cs.duration_s = time.time() - start
    return cs


def _state_write_mw(ctx: _StepExecContext, next_fn: _NextFn) -> CompletedState:
    """Write the final CompletedState to context['step_states']."""
    cs = next_fn()
    ctx.context.setdefault("step_states", {})
    ctx.context["step_states"][ctx.step_key] = dataclasses.asdict(cs)
    return cs


def _execute_mw(ctx: _StepExecContext, next_fn: _NextFn) -> CompletedState:
    """Core execution: try/except, return-type check, validate, rollback."""
    try:
        cs = next_fn()
    except Exception as exc:
        try:
            ctx.step.rollback(ctx.context)
        except Exception:
            pass
        return CompletedState(
            success=False,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            error={"type": type(exc).__name__, "message": str(exc)},
        )

    if not isinstance(cs, CompletedState):
        return CompletedState(
            success=False,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            error={"message": "invalid CompletedState returned"},
        )

    if cs.success and not ctx.step.validate(ctx.context):
        cs = CompletedState(
            success=False,
            timestamp=cs.timestamp,
            duration_s=cs.duration_s,
            provides=cs.provides,
            outputs=cs.outputs,
            error={
                "message": (
                    f"validation failed: provides keys "
                    f"{list(ctx.step.provides)} not all present in context"
                ),
            },
            meta=cs.meta,
            signature=cs.signature,
        )

    if not cs.success:
        try:
            ctx.step.rollback(ctx.context)
        except Exception:
            pass

    return cs


default_middleware: List[_Middleware] = [
    _idempotent_skip_mw,
    _state_write_mw,
    _requirements_check_mw,
    _logging_mw,
    _timing_mw,
    _execute_mw,
]


def _build_chain(
    middleware: List[_Middleware],
    ctx: _StepExecContext,
    core: _NextFn,
) -> _NextFn:
    """Build a callable chain: middleware[0] wraps middleware[1] wraps ... wraps core."""
    fn = core
    for mw in reversed(middleware):
        outer = fn
        fn = (lambda m, f: lambda: m(ctx, f))(mw, outer)
    return fn


# ---------------------------------------------------------------------------
# Phase B: DAG solver
# ---------------------------------------------------------------------------

def _topo_sort(steps: List[PipelineStep]) -> List[PipelineStep]:
    """Topological sort of steps based on requires/provides.

    Stable sort: declared order is the tiebreaker for independent steps.
    Raises ValueError on cycles.
    """
    n = len(steps)
    if n == 0:
        return []

    provider: Dict[str, int] = {}
    for i, step in enumerate(steps):
        for key in step.provides:
            provider[key] = i

    deps: List[Set[int]] = [set() for _ in range(n)]
    for i, step in enumerate(steps):
        for key in step.requires:
            if key in provider and provider[key] != i:
                deps[i].add(provider[key])

    in_degree = [len(d) for d in deps]
    result: List[int] = []
    remaining = set(range(n))

    while remaining:
        chosen = None
        for i in range(n):
            if i in remaining and in_degree[i] == 0:
                chosen = i
                break

        if chosen is None:
            cycle_nodes = [steps[i].name for i in sorted(remaining)]
            raise ValueError(f"Cycle detected among steps: {cycle_nodes}")

        result.append(chosen)
        remaining.discard(chosen)

        for i in remaining:
            if chosen in deps[i]:
                deps[i].discard(chosen)
                in_degree[i] -= 1

    return [steps[i] for i in result]


# ---------------------------------------------------------------------------
# Phase D: Fan-out / Fan-in step types
# ---------------------------------------------------------------------------

class GeneratorStep(PipelineStep):
    """Fan-out: produce N work items from 1 input work item.

    Subclasses implement generate(). The orchestrator calls generate() and
    runs downstream steps once per emitted work item.
    """

    @abstractmethod
    def generate(self, work_item: Any) -> List[Any]:  # pragma: no cover
        raise NotImplementedError()

    def run(self, context: Dict[str, Any]) -> CompletedState:
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=list(self.provides),
        )


class CollectorStep(PipelineStep):
    """Fan-in: merge N work items into 1.

    Subclasses implement collect(). The orchestrator passes all accumulated
    work items and expects a single merged work item back.
    """

    @abstractmethod
    def collect(self, work_items: List[Any]) -> Any:  # pragma: no cover
        raise NotImplementedError()

    def run(self, context: Dict[str, Any]) -> CompletedState:
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=list(self.provides),
        )


# ---------------------------------------------------------------------------
# Pipeline (composite: IS-A PipelineStep)
# ---------------------------------------------------------------------------

class Pipeline(PipelineStep):
    """Composable pipeline orchestrator.

    A Pipeline IS-A PipelineStep, so it can be nested inside another Pipeline.
    The orchestrator owns timing, step_states writes, and validation calls
    via a composable middleware chain.
    """

    steps: List[PipelineStep]
    force: bool

    def __init__(
        self,
        name: str = "pipeline",
        version: str = "1.0",
        steps: Optional[List[PipelineStep]] = None,
        requires: Optional[Iterable[str]] = None,
        provides: Optional[Iterable[str]] = None,
        idempotent: bool = True,
        continue_on_error: bool = False,
        force: bool = False,
        middleware: Optional[List[_Middleware]] = None,
        resolve_order: bool = False,
        scheduler: Any = None,
    ) -> None:
        self._raw_requires = set(requires) if requires is not None else None
        self._raw_provides = set(provides) if provides is not None else None
        super().__init__(
            name=name,
            requires=requires,
            provides=provides,
            idempotent=idempotent,
            continue_on_error=continue_on_error,
            version=version,
        )
        self.steps = list(steps or [])
        if resolve_order and self.steps:
            self.steps = _topo_sort(self.steps)
        self.force = force
        self._middleware = list(middleware) if middleware is not None else list(default_middleware)
        self._scheduler = scheduler
        if self._raw_requires is None or self._raw_provides is None:
            self._infer_requires_provides()

    def _infer_requires_provides(self) -> None:
        """Auto-infer requires/provides from child steps when not explicitly set."""
        all_provides: Set[str] = set()
        all_requires: Set[str] = set()
        for step in self.steps:
            all_requires |= step.requires
            all_provides |= step.provides
        if self._raw_requires is None:
            self.requires = all_requires - all_provides
        if self._raw_provides is None:
            self.provides = all_provides

    # -- Phase C: WorkItem-aware entry point --------------------------------

    def run_item(self, work_item: Any, *, _prefix: str = "", dry_run: bool = False) -> CompletedState:
        """Run the pipeline against a WorkItem.

        Steps receive work_item.attributes as their context dict.
        Fan-out/fan-in (GeneratorStep/CollectorStep) is handled transparently.
        """
        context = work_item.attributes
        context.setdefault("step_states", {})
        pipeline_start = time.time()
        force = self.force or context.get("_force", False)

        if dry_run:
            self._dry_run_report(context, _prefix or self.name)
            return CompletedState(
                success=True,
                timestamp=CompletedState.now_iso(),
                duration_s=0.0,
                provides=list(self.provides),
                meta={"dry_run": True},
            )

        current_items = [work_item]
        child_results: List[CompletedState] = []
        step_counter = context.get("_step_index", 0)

        for step in self.steps:
            if isinstance(step, GeneratorStep):
                new_items = []
                for item in current_items:
                    generated = step.generate(item)
                    for gi in generated:
                        if gi.parent_id is None:
                            gi.parent_id = item.id
                        gi.attributes["step_states"] = {}
                    new_items.extend(generated)
                if new_items:
                    current_items = new_items
                continue

            if isinstance(step, CollectorStep):
                collected = step.collect(current_items)
                context.update(collected.attributes)
                collected.attributes = context
                current_items = [collected]
                continue

            for item in current_items:
                step_counter += 1
                item.attributes["_step_index"] = step_counter
                cs = self._execute_step(
                    step, item.attributes, force=force,
                    prefix=_prefix or self.name,
                )
                child_results.append(cs)
                if not cs.success and not step.continue_on_error:
                    return CompletedState(
                        success=False,
                        timestamp=CompletedState.now_iso(),
                        duration_s=time.time() - pipeline_start,
                        provides=list(self.provides),
                        error=cs.error,
                        meta={"failed_step": step.name},
                    )

        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=time.time() - pipeline_start,
            provides=list(self.provides),
            meta={"steps_run": len(child_results)},
        )

    def run(self, context: Dict[str, Any], *, _prefix: str = "", dry_run: bool = False) -> CompletedState:
        """Run the pipeline against a plain context dict (backward-compatible).

        Internally wraps the context in a WorkItem and delegates to run_item().
        """
        try:
            from lib.work_item import WorkItem
            wi = WorkItem(id="default", attributes=context)
        except ImportError:
            wi = _SimpleWorkItem(id="default", attributes=context)
        return self.run_item(wi, _prefix=_prefix, dry_run=dry_run)

    def _execute_step(
        self,
        step: PipelineStep,
        context: Dict[str, Any],
        *,
        force: bool,
        prefix: str,
    ) -> CompletedState:
        """Run a single step through the middleware chain."""
        step_key = f"{prefix}.{step.name}" if prefix else step.name
        ctx = _StepExecContext(step=step, context=context, step_key=step_key, force=force)

        def core() -> CompletedState:
            sched = getattr(step, "_scheduler", None) or self._scheduler
            if sched is not None:
                try:
                    from lib.work_item import WorkItem
                    wi = WorkItem(id=step_key, attributes=context)
                except ImportError:
                    wi = _SimpleWorkItem(id=step_key, attributes=context)
                return sched.execute(step, wi)
            if isinstance(step, Pipeline):
                return step.run(context, _prefix=step_key)
            return step.run(context)

        return _build_chain(self._middleware, ctx, core)()

    def _dry_run_report(self, context: Dict[str, Any], prefix: str) -> None:
        """Log planned execution order without running anything."""
        logger.info("DRY-RUN plan for [%s]:", prefix)
        self._dry_run_walk(context, prefix, indent=1)

    def _dry_run_walk(self, context: Dict[str, Any], prefix: str, indent: int) -> None:
        pad = "  " * indent
        for step in self.steps:
            step_key = f"{prefix}.{step.name}"
            missing = [r for r in step.requires if r not in context]
            status = "READY" if not missing else f"BLOCKED (missing: {missing})"
            if isinstance(step, Pipeline):
                logger.info("%s[pipeline] %s — %s", pad, step_key, status)
                step._dry_run_walk(context, step_key, indent + 1)
            elif isinstance(step, StepGroup):
                mode = "parallel" if step.parallel else "sequential"
                logger.info("%s[group/%s] %s — %s", pad, mode, step_key, status)
                for child in step.steps:
                    child_key = f"{step_key}.{child.name}"
                    child_missing = [r for r in child.requires if r not in context]
                    child_status = "READY" if not child_missing else f"BLOCKED (missing: {child_missing})"
                    logger.info("%s  %s — %s", pad, child_key, child_status)
            else:
                logger.info("%s%s — %s", pad, step_key, status)


# ---------------------------------------------------------------------------
# Fallback WorkItem (used when work_item.py is not yet importable)
# ---------------------------------------------------------------------------

@dataclass
class _SimpleWorkItem:
    """Minimal stand-in so Pipeline.run(context) works without work_item.py."""
    id: str
    attributes: Dict[str, Any]
    parent_id: Optional[str] = None


# ---------------------------------------------------------------------------
# StepGroup (parallel execution support)
# ---------------------------------------------------------------------------

class StepGroup(PipelineStep):
    """A group of steps that can execute sequentially or in parallel.

    When parallel=True, child steps must have non-overlapping `provides` sets.
    """

    steps: List[PipelineStep]
    parallel: bool

    def __init__(
        self,
        name: str,
        steps: Optional[List[PipelineStep]] = None,
        parallel: bool = False,
        requires: Optional[Iterable[str]] = None,
        provides: Optional[Iterable[str]] = None,
        version: str = "0.0.0",
        continue_on_error: bool = False,
    ) -> None:
        self.steps = list(steps or [])
        self.parallel = parallel
        self._raw_requires = set(requires) if requires is not None else None
        self._raw_provides = set(provides) if provides is not None else None

        inferred_requires, inferred_provides = self._infer_contracts()
        super().__init__(
            name=name,
            requires=requires if requires is not None else inferred_requires,
            provides=provides if provides is not None else inferred_provides,
            idempotent=False,
            continue_on_error=continue_on_error,
            version=version,
        )

        if self.parallel:
            self._validate_no_provides_overlap()

    def _infer_contracts(self) -> tuple:
        all_provides: Set[str] = set()
        all_requires: Set[str] = set()
        for step in self.steps:
            all_requires |= step.requires
            all_provides |= step.provides
        external_requires = all_requires - all_provides
        return external_requires, all_provides

    def _validate_no_provides_overlap(self) -> None:
        seen: Dict[str, str] = {}
        for step in self.steps:
            for key in step.provides:
                if key in seen:
                    raise ValueError(
                        f"Parallel StepGroup '{self.name}': provides key '{key}' "
                        f"claimed by both '{seen[key]}' and '{step.name}'"
                    )
                seen[key] = step.name

    def run(self, context: Dict[str, Any]) -> CompletedState:
        start = time.time()
        if self.parallel:
            return self._run_parallel(context, start)
        return self._run_sequential(context, start)

    def _run_sequential(self, context: Dict[str, Any], start: float) -> CompletedState:
        for step in self.steps:
            step_start = time.time()
            try:
                cs = step.run(context)
            except Exception as exc:
                try:
                    step.rollback(context)
                except Exception:
                    pass
                return CompletedState(
                    success=False,
                    timestamp=CompletedState.now_iso(),
                    duration_s=time.time() - start,
                    error={"type": type(exc).__name__, "message": str(exc)},
                    meta={"failed_step": step.name},
                )
            if not isinstance(cs, CompletedState):
                return CompletedState(
                    success=False,
                    timestamp=CompletedState.now_iso(),
                    duration_s=time.time() - start,
                    error={"message": "invalid CompletedState returned"},
                    meta={"failed_step": step.name},
                )
            cs.duration_s = time.time() - step_start
            if not cs.success and not step.continue_on_error:
                return CompletedState(
                    success=False,
                    timestamp=CompletedState.now_iso(),
                    duration_s=time.time() - start,
                    error=cs.error,
                    meta={"failed_step": step.name},
                )
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=time.time() - start,
            provides=list(self.provides),
        )

    def _run_parallel(self, context: Dict[str, Any], start: float) -> CompletedState:
        errors: List[Dict[str, Any]] = []

        def _run_one(step: PipelineStep) -> CompletedState:
            step_start = time.time()
            try:
                cs = step.run(context)
            except Exception as exc:
                try:
                    step.rollback(context)
                except Exception:
                    pass
                return CompletedState(
                    success=False,
                    timestamp=CompletedState.now_iso(),
                    duration_s=time.time() - step_start,
                    error={"type": type(exc).__name__, "message": str(exc)},
                    meta={"failed_step": step.name},
                )
            if not isinstance(cs, CompletedState):
                return CompletedState(
                    success=False,
                    timestamp=CompletedState.now_iso(),
                    duration_s=time.time() - step_start,
                    error={"message": "invalid CompletedState returned"},
                    meta={"failed_step": step.name},
                )
            cs.duration_s = time.time() - step_start
            return cs

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.steps) or 1) as pool:
            futures = {pool.submit(_run_one, step): step for step in self.steps}
            for future in concurrent.futures.as_completed(futures):
                cs = future.result()
                step = futures[future]
                if not cs.success:
                    errors.append({"step": step.name, "error": cs.error})

        if errors:
            return CompletedState(
                success=False,
                timestamp=CompletedState.now_iso(),
                duration_s=time.time() - start,
                error={"message": "parallel step(s) failed", "details": errors},
            )
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=time.time() - start,
            provides=list(self.provides),
        )
