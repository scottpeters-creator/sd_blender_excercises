"""Comprehensive unit tests for the composable pipeline library.

All tests run without Blender (bpy). The pipeline library is Blender-agnostic.
Covers all five POR phases: A (foundation), B (return type), C (composite),
D (parallel groups), E (dry-run).
"""

from __future__ import annotations

import logging
import threading
import time
import unittest
from typing import Any, Dict

from lib.pipeline import (
    CompletedState,
    ExitCode,
    Pipeline,
    PipelineStep,
    StepGroup,
    _short_signature,
    exit_code_from,
    write_json_atomic,
)


# ---------------------------------------------------------------------------
# Reusable test step helpers
# ---------------------------------------------------------------------------

class SetKeyStep(PipelineStep):
    """Step that sets a context key to a value."""

    def __init__(self, name: str, key: str, value: Any, **kwargs: Any) -> None:
        self._key = key
        self._value = value
        super().__init__(name=name, provides=[key], **kwargs)

    def run(self, context: Dict[str, Any]) -> CompletedState:
        context[self._key] = self._value
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=[self._key],
        )


class FailingStep(PipelineStep):
    """Step that returns success=False."""

    def __init__(self, name: str = "failing_step", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)

    def run(self, context: Dict[str, Any]) -> CompletedState:
        return CompletedState(
            success=False,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            error={"message": "intentional failure"},
        )


class ExplodingStep(PipelineStep):
    """Step that raises an exception."""

    def __init__(self, name: str = "exploding_step", **kwargs: Any) -> None:
        self.rollback_called = False
        super().__init__(name=name, **kwargs)

    def run(self, context: Dict[str, Any]) -> CompletedState:
        raise RuntimeError("boom")

    def rollback(self, context: Dict[str, Any]) -> None:
        self.rollback_called = True


class BadReturnStep(PipelineStep):
    """Step that returns a non-CompletedState value."""

    def __init__(self, name: str = "bad_return") -> None:
        super().__init__(name=name)

    def run(self, context: Dict[str, Any]) -> CompletedState:
        return "not a CompletedState"  # type: ignore


class NoProvideStep(PipelineStep):
    """Step that claims provides=['x'] but never sets it on context."""

    def __init__(self) -> None:
        super().__init__(name="no_provide", provides=["x"])

    def run(self, context: Dict[str, Any]) -> CompletedState:
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=["x"],
        )


class SlowStep(PipelineStep):
    """Step that sleeps briefly, for testing orchestrator-owned timing."""

    def __init__(self, name: str = "slow", duration: float = 0.05, **kwargs: Any) -> None:
        self._duration = duration
        super().__init__(name=name, provides=["slow_done"], **kwargs)

    def run(self, context: Dict[str, Any]) -> CompletedState:
        time.sleep(self._duration)
        context["slow_done"] = True
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=["slow_done"],
        )


class ThreadRecordStep(PipelineStep):
    """Records the thread ID it ran on, for parallel testing."""

    def __init__(self, name: str, key: str) -> None:
        self._key = key
        super().__init__(name=name, provides=[key])

    def run(self, context: Dict[str, Any]) -> CompletedState:
        context[self._key] = threading.current_thread().ident
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=[self._key],
        )


# ===========================================================================
# Phase A: Foundation
# ===========================================================================

class TestVersionField(unittest.TestCase):
    def test_default_version(self):
        step = SetKeyStep("s", "k", 1)
        self.assertEqual(step.version, "0.0.0")

    def test_custom_version(self):
        step = SetKeyStep("s", "k", 1, version="2.1.0")
        self.assertEqual(step.version, "2.1.0")

    def test_pipeline_version(self):
        p = Pipeline(name="p", version="3.0")
        self.assertEqual(p.version, "3.0")


class TestOrchestratorOwnsTiming(unittest.TestCase):
    def test_duration_overwritten_by_orchestrator(self):
        p = Pipeline(name="p", steps=[SlowStep(name="slow", duration=0.05)])
        ctx: Dict[str, Any] = {}
        result = p.run(ctx)
        self.assertTrue(result.success)
        state = ctx["step_states"]["p.slow"]
        self.assertGreater(state["duration_s"], 0.01)


class TestOrchestratorOwnsStepStates(unittest.TestCase):
    def test_step_states_written_by_orchestrator(self):
        step = SetKeyStep("a", "key_a", 42)
        p = Pipeline(name="test", steps=[step])
        ctx: Dict[str, Any] = {}
        p.run(ctx)
        self.assertIn("test.a", ctx["step_states"])
        self.assertTrue(ctx["step_states"]["test.a"]["success"])

    def test_step_does_not_need_to_write_step_states(self):
        step = SetKeyStep("a", "key_a", 42)
        p = Pipeline(name="t", steps=[step])
        ctx: Dict[str, Any] = {}
        p.run(ctx)
        self.assertNotIn("a", ctx["step_states"])
        self.assertIn("t.a", ctx["step_states"])


class TestValidationCalled(unittest.TestCase):
    def test_validation_failure_treated_as_step_failure(self):
        step = NoProvideStep()
        p = Pipeline(name="vp", steps=[step])
        ctx: Dict[str, Any] = {}
        result = p.run(ctx)
        self.assertFalse(result.success)
        state = ctx["step_states"]["vp.no_provide"]
        self.assertFalse(state["success"])
        self.assertIn("validation failed", state["error"]["message"])

    def test_validation_passes_when_provides_set(self):
        step = SetKeyStep("s", "k", 1)
        p = Pipeline(name="vp", steps=[step])
        ctx: Dict[str, Any] = {}
        result = p.run(ctx)
        self.assertTrue(result.success)


class TestLogging(unittest.TestCase):
    def test_lifecycle_events_logged(self):
        step = SetKeyStep("a", "k", 1)
        p = Pipeline(name="log_test", steps=[step])
        with self.assertLogs("pipeline", level="INFO") as cm:
            p.run({})
        messages = " ".join(cm.output)
        self.assertIn("START", messages)
        self.assertIn("OK", messages)

    def test_failure_logged(self):
        step = FailingStep()
        p = Pipeline(name="flog", steps=[step])
        with self.assertLogs("pipeline", level="ERROR") as cm:
            p.run({})
        messages = " ".join(cm.output)
        self.assertIn("FAIL", messages)


class TestWriteJsonAtomicPrefix(unittest.TestCase):
    def test_atomic_write_roundtrip(self):
        import json
        import os
        import tempfile
        d = tempfile.mkdtemp()
        path = os.path.join(d, "out.json")
        write_json_atomic(path, {"hello": "world"})
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data, {"hello": "world"})
        os.remove(path)
        os.rmdir(d)


# ===========================================================================
# Phase B: Return Type
# ===========================================================================

class TestExitCode(unittest.TestCase):
    def test_ok(self):
        cs = CompletedState(success=True, timestamp="", duration_s=0.0)
        self.assertEqual(exit_code_from(cs), ExitCode.OK)
        self.assertEqual(int(exit_code_from(cs)), 0)

    def test_missing_requirement(self):
        cs = CompletedState(success=False, timestamp="", duration_s=0.0,
                            error={"message": "missing requirements: ['x']"})
        self.assertEqual(exit_code_from(cs), ExitCode.MISSING_REQUIREMENT)

    def test_step_exception(self):
        cs = CompletedState(success=False, timestamp="", duration_s=0.0,
                            error={"type": "RuntimeError", "message": "boom"})
        self.assertEqual(exit_code_from(cs), ExitCode.STEP_EXCEPTION)

    def test_invalid_return(self):
        cs = CompletedState(success=False, timestamp="", duration_s=0.0,
                            error={"message": "invalid CompletedState returned"})
        self.assertEqual(exit_code_from(cs), ExitCode.INVALID_RETURN)

    def test_step_failed(self):
        cs = CompletedState(success=False, timestamp="", duration_s=0.0,
                            error={"message": "intentional failure"})
        self.assertEqual(exit_code_from(cs), ExitCode.STEP_FAILED)


class TestPipelineReturnsCompletedState(unittest.TestCase):
    def test_success_returns_completed_state(self):
        p = Pipeline(name="p", steps=[SetKeyStep("a", "k", 1)])
        result = p.run({})
        self.assertIsInstance(result, CompletedState)
        self.assertTrue(result.success)

    def test_failure_returns_completed_state(self):
        p = Pipeline(name="p", steps=[FailingStep()])
        result = p.run({})
        self.assertIsInstance(result, CompletedState)
        self.assertFalse(result.success)


# ===========================================================================
# Phase C: Composite Pattern
# ===========================================================================

class TestPipelineIsAStep(unittest.TestCase):
    def test_isinstance(self):
        p = Pipeline(name="p")
        self.assertIsInstance(p, PipelineStep)

    def test_has_name_version(self):
        p = Pipeline(name="myp", version="2.0")
        self.assertEqual(p.name, "myp")
        self.assertEqual(p.version, "2.0")


class TestFlatPipeline(unittest.TestCase):
    def test_steps_run_in_order(self):
        order: list = []

        class OrderStep(PipelineStep):
            def __init__(self, n: str, key: str):
                super().__init__(name=n, provides=[key])
                self._key = key

            def run(self, context: Dict[str, Any]) -> CompletedState:
                order.append(self.name)
                context[self._key] = True
                return CompletedState(success=True, timestamp=CompletedState.now_iso(),
                                      duration_s=0.0, provides=[self._key])

        p = Pipeline(name="flat", steps=[
            OrderStep("first", "a"),
            OrderStep("second", "b"),
            OrderStep("third", "c"),
        ])
        result = p.run({})
        self.assertTrue(result.success)
        self.assertEqual(order, ["first", "second", "third"])

    def test_context_keys_populated(self):
        p = Pipeline(name="p", steps=[
            SetKeyStep("s1", "alpha", 10),
            SetKeyStep("s2", "beta", 20, requires=["alpha"]),
        ])
        ctx: Dict[str, Any] = {}
        p.run(ctx)
        self.assertEqual(ctx["alpha"], 10)
        self.assertEqual(ctx["beta"], 20)


class TestMissingRequirements(unittest.TestCase):
    def test_stops_on_missing(self):
        step = SetKeyStep("need_x", "out", 1, requires=["x"])
        p = Pipeline(name="p", steps=[step])
        result = p.run({})
        self.assertFalse(result.success)
        self.assertIn("missing requirements", result.error["message"])


class TestIdempotentSkip(unittest.TestCase):
    def test_skip_already_succeeded(self):
        call_count = 0

        class CountStep(PipelineStep):
            def __init__(self):
                super().__init__(name="counter", provides=["counted"], idempotent=True)

            def run(self, context: Dict[str, Any]) -> CompletedState:
                nonlocal call_count
                call_count += 1
                context["counted"] = True
                return CompletedState(success=True, timestamp=CompletedState.now_iso(),
                                      duration_s=0.0, provides=["counted"])

        p = Pipeline(name="p", steps=[CountStep()])
        ctx: Dict[str, Any] = {}
        p.run(ctx)
        self.assertEqual(call_count, 1)
        p.run(ctx)
        self.assertEqual(call_count, 1)

    def test_force_reruns(self):
        call_count = 0

        class CountStep(PipelineStep):
            def __init__(self):
                super().__init__(name="counter", provides=["counted"], idempotent=True)

            def run(self, context: Dict[str, Any]) -> CompletedState:
                nonlocal call_count
                call_count += 1
                context["counted"] = True
                return CompletedState(success=True, timestamp=CompletedState.now_iso(),
                                      duration_s=0.0, provides=["counted"])

        p = Pipeline(name="p", steps=[CountStep()], force=True)
        ctx: Dict[str, Any] = {}
        p.run(ctx)
        self.assertEqual(call_count, 1)
        p.run(ctx)
        self.assertEqual(call_count, 2)


class TestExceptionHandling(unittest.TestCase):
    def test_exception_recorded(self):
        step = ExplodingStep()
        p = Pipeline(name="p", steps=[step])
        result = p.run({})
        self.assertFalse(result.success)

    def test_rollback_called_on_exception(self):
        step = ExplodingStep()
        p = Pipeline(name="p", steps=[step])
        p.run({})
        self.assertTrue(step.rollback_called)

    def test_continue_on_error(self):
        exploder = ExplodingStep(name="boom", continue_on_error=True)
        after = SetKeyStep("after", "survived", True)
        p = Pipeline(name="p", steps=[exploder, after])
        result = p.run({})
        self.assertTrue(result.success)


class TestBadReturn(unittest.TestCase):
    def test_invalid_return_type(self):
        p = Pipeline(name="p", steps=[BadReturnStep()])
        result = p.run({})
        self.assertFalse(result.success)
        state = result.error or {}
        self.assertIn("invalid CompletedState", state.get("message", ""))


class TestNestedPipeline(unittest.TestCase):
    def test_pipeline_in_pipeline(self):
        inner = Pipeline(name="inner", steps=[
            SetKeyStep("a", "alpha", 1),
            SetKeyStep("b", "beta", 2),
        ])
        outer = Pipeline(name="outer", steps=[inner])
        ctx: Dict[str, Any] = {}
        result = outer.run(ctx)
        self.assertTrue(result.success)
        self.assertEqual(ctx["alpha"], 1)
        self.assertEqual(ctx["beta"], 2)

    def test_namespaced_step_states(self):
        inner = Pipeline(name="inner", steps=[
            SetKeyStep("a", "alpha", 1),
        ])
        outer = Pipeline(name="outer", steps=[inner])
        ctx: Dict[str, Any] = {}
        outer.run(ctx)
        self.assertIn("outer.inner", ctx["step_states"])
        self.assertIn("outer.inner.a", ctx["step_states"])

    def test_deep_nesting(self):
        step = SetKeyStep("leaf", "val", 42)
        level2 = Pipeline(name="l2", steps=[step])
        level1 = Pipeline(name="l1", steps=[level2])
        root = Pipeline(name="root", steps=[level1])
        ctx: Dict[str, Any] = {}
        result = root.run(ctx)
        self.assertTrue(result.success)
        self.assertEqual(ctx["val"], 42)
        self.assertIn("root.l1.l2.leaf", ctx["step_states"])


class TestAutoInference(unittest.TestCase):
    def test_provides_union(self):
        p = Pipeline(name="p", steps=[
            SetKeyStep("a", "x", 1),
            SetKeyStep("b", "y", 2),
        ])
        self.assertEqual(p.provides, {"x", "y"})

    def test_requires_external_only(self):
        s1 = SetKeyStep("a", "x", 1)
        s2 = SetKeyStep("b", "y", 2, requires=["x"])
        p = Pipeline(name="p", steps=[s1, s2])
        self.assertEqual(p.requires, set())

    def test_requires_not_internally_satisfied(self):
        s1 = SetKeyStep("a", "x", 1, requires=["external"])
        p = Pipeline(name="p", steps=[s1])
        self.assertEqual(p.requires, {"external"})

    def test_explicit_overrides_inference(self):
        p = Pipeline(name="p", steps=[SetKeyStep("a", "x", 1)],
                     requires=["forced_req"], provides=["forced_prov"])
        self.assertEqual(p.requires, {"forced_req"})
        self.assertEqual(p.provides, {"forced_prov"})


# ===========================================================================
# Phase D: Parallel Groups (StepGroup)
# ===========================================================================

class TestStepGroupSequential(unittest.TestCase):
    def test_runs_all_steps(self):
        g = StepGroup(name="g", steps=[
            SetKeyStep("a", "x", 1),
            SetKeyStep("b", "y", 2),
        ])
        ctx: Dict[str, Any] = {}
        cs = g.run(ctx)
        self.assertTrue(cs.success)
        self.assertEqual(ctx["x"], 1)
        self.assertEqual(ctx["y"], 2)

    def test_failure_stops(self):
        g = StepGroup(name="g", steps=[
            FailingStep(name="fail1"),
            SetKeyStep("after", "z", 99),
        ])
        ctx: Dict[str, Any] = {}
        cs = g.run(ctx)
        self.assertFalse(cs.success)
        self.assertNotIn("z", ctx)


class TestStepGroupParallel(unittest.TestCase):
    def test_parallel_runs_all(self):
        g = StepGroup(name="g", parallel=True, steps=[
            SetKeyStep("a", "x", 1),
            SetKeyStep("b", "y", 2),
            SetKeyStep("c", "z", 3),
        ])
        ctx: Dict[str, Any] = {}
        cs = g.run(ctx)
        self.assertTrue(cs.success)
        self.assertEqual(ctx["x"], 1)
        self.assertEqual(ctx["y"], 2)
        self.assertEqual(ctx["z"], 3)

    def test_parallel_failure(self):
        g = StepGroup(name="g", parallel=True, steps=[
            SetKeyStep("ok", "x", 1),
            FailingStep(name="bad"),
        ])
        ctx: Dict[str, Any] = {}
        cs = g.run(ctx)
        self.assertFalse(cs.success)

    def test_parallel_uses_threads(self):
        g = StepGroup(name="g", parallel=True, steps=[
            ThreadRecordStep("t1", "tid1"),
            ThreadRecordStep("t2", "tid2"),
        ])
        ctx: Dict[str, Any] = {}
        g.run(ctx)
        self.assertIn("tid1", ctx)
        self.assertIn("tid2", ctx)

    def test_provides_overlap_rejected(self):
        with self.assertRaises(ValueError) as cm:
            StepGroup(name="g", parallel=True, steps=[
                SetKeyStep("a", "x", 1),
                SetKeyStep("b", "x", 2),
            ])
        self.assertIn("provides key 'x'", str(cm.exception))

    def test_provides_overlap_ok_sequential(self):
        g = StepGroup(name="g", parallel=False, steps=[
            SetKeyStep("a", "x", 1),
            SetKeyStep("b", "x", 2),
        ])
        ctx: Dict[str, Any] = {}
        cs = g.run(ctx)
        self.assertTrue(cs.success)
        self.assertEqual(ctx["x"], 2)


class TestStepGroupInPipeline(unittest.TestCase):
    def test_group_as_pipeline_step(self):
        g = StepGroup(name="group", steps=[
            SetKeyStep("a", "x", 1),
            SetKeyStep("b", "y", 2),
        ])
        p = Pipeline(name="p", steps=[g])
        ctx: Dict[str, Any] = {}
        result = p.run(ctx)
        self.assertTrue(result.success)
        self.assertEqual(ctx["x"], 1)
        self.assertEqual(ctx["y"], 2)


class TestStepGroupAutoInference(unittest.TestCase):
    def test_inferred_provides(self):
        g = StepGroup(name="g", steps=[
            SetKeyStep("a", "x", 1),
            SetKeyStep("b", "y", 2),
        ])
        self.assertEqual(g.provides, {"x", "y"})

    def test_inferred_requires(self):
        g = StepGroup(name="g", steps=[
            SetKeyStep("a", "x", 1, requires=["external"]),
        ])
        self.assertEqual(g.requires, {"external"})


# ===========================================================================
# Phase E: Dry Run
# ===========================================================================

class TestDryRun(unittest.TestCase):
    def test_no_steps_executed(self):
        call_count = 0

        class Tracker(PipelineStep):
            def __init__(self):
                super().__init__(name="tracker", provides=["tracked"])

            def run(self, context: Dict[str, Any]) -> CompletedState:
                nonlocal call_count
                call_count += 1
                context["tracked"] = True
                return CompletedState(success=True, timestamp=CompletedState.now_iso(),
                                      duration_s=0.0, provides=["tracked"])

        p = Pipeline(name="p", steps=[Tracker()])
        result = p.run({}, dry_run=True)
        self.assertTrue(result.success)
        self.assertEqual(call_count, 0)
        self.assertTrue(result.meta.get("dry_run"))

    def test_dry_run_logs_plan(self):
        p = Pipeline(name="dryp", steps=[
            SetKeyStep("a", "x", 1),
            SetKeyStep("b", "y", 2, requires=["x"]),
        ])
        with self.assertLogs("pipeline", level="INFO") as cm:
            p.run({}, dry_run=True)
        messages = " ".join(cm.output)
        self.assertIn("DRY-RUN", messages)
        self.assertIn("dryp.a", messages)
        self.assertIn("dryp.b", messages)

    def test_dry_run_nested(self):
        inner = Pipeline(name="inner", steps=[SetKeyStep("x", "v", 1)])
        outer = Pipeline(name="outer", steps=[inner])
        with self.assertLogs("pipeline", level="INFO") as cm:
            outer.run({}, dry_run=True)
        messages = " ".join(cm.output)
        self.assertIn("outer.inner", messages)


# ===========================================================================
# Helpers
# ===========================================================================

class TestShortSignature(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(_short_signature({"a": 1}), _short_signature({"a": 1}))

    def test_different_values(self):
        self.assertNotEqual(_short_signature({"a": 1}), _short_signature({"a": 2}))


class TestCompletedStateNowIso(unittest.TestCase):
    def test_format(self):
        ts = CompletedState.now_iso()
        self.assertTrue(ts.endswith("Z"))
        self.assertIn("T", ts)


# ===========================================================================
# Phase A (PDG): Middleware Chain
# ===========================================================================

from lib.pipeline import (
    _StepExecContext,
    _build_chain,
    _execute_mw,
    _idempotent_skip_mw,
    _logging_mw,
    _requirements_check_mw,
    _state_write_mw,
    _timing_mw,
    GeneratorStep,
    CollectorStep,
    _topo_sort,
    default_middleware,
)
from lib.work_item import WorkItem


class TestMiddlewareDefaultRegression(unittest.TestCase):
    """Default middleware must reproduce the exact behavior of the original _execute_step."""

    def test_success_writes_step_states(self):
        p = Pipeline(name="mw", steps=[SetKeyStep("a", "k", 1)])
        ctx: Dict[str, Any] = {}
        result = p.run(ctx)
        self.assertTrue(result.success)
        self.assertIn("mw.a", ctx["step_states"])
        self.assertTrue(ctx["step_states"]["mw.a"]["success"])

    def test_failure_writes_step_states(self):
        p = Pipeline(name="mw", steps=[FailingStep()])
        ctx: Dict[str, Any] = {}
        p.run(ctx)
        self.assertIn("mw.failing_step", ctx["step_states"])
        self.assertFalse(ctx["step_states"]["mw.failing_step"]["success"])

    def test_exception_writes_step_states(self):
        p = Pipeline(name="mw", steps=[ExplodingStep()])
        ctx: Dict[str, Any] = {}
        p.run(ctx)
        self.assertIn("mw.exploding_step", ctx["step_states"])
        self.assertFalse(ctx["step_states"]["mw.exploding_step"]["success"])

    def test_missing_requirements_writes_step_states(self):
        step = SetKeyStep("need_x", "out", 1, requires=["x"])
        p = Pipeline(name="mw", steps=[step])
        ctx: Dict[str, Any] = {}
        p.run(ctx)
        self.assertIn("mw.need_x", ctx["step_states"])
        self.assertFalse(ctx["step_states"]["mw.need_x"]["success"])


class TestCustomMiddleware(unittest.TestCase):
    def test_custom_middleware_fires_in_order(self):
        order: list = []

        def mw_a(ctx, next_fn):
            order.append("a_before")
            cs = next_fn()
            order.append("a_after")
            return cs

        def mw_b(ctx, next_fn):
            order.append("b_before")
            cs = next_fn()
            order.append("b_after")
            return cs

        p = Pipeline(name="p", steps=[SetKeyStep("s", "k", 1)], middleware=[mw_a, mw_b])
        p.run({})
        self.assertEqual(order, ["a_before", "b_before", "b_after", "a_after"])

    def test_early_return_skips_downstream(self):
        def blocking_mw(ctx, next_fn):
            return CompletedState(
                success=True, timestamp=CompletedState.now_iso(),
                duration_s=0.0, meta={"blocked": True},
            )

        call_count = 0

        class CountStep(PipelineStep):
            def __init__(self):
                super().__init__(name="counter", provides=["c"])

            def run(self, context):
                nonlocal call_count
                call_count += 1
                context["c"] = True
                return CompletedState(success=True, timestamp=CompletedState.now_iso(),
                                      duration_s=0.0, provides=["c"])

        p = Pipeline(name="p", steps=[CountStep()], middleware=[blocking_mw])
        result = p.run({})
        self.assertTrue(result.success)
        self.assertEqual(call_count, 0)

    def test_retry_middleware(self):
        attempt = 0

        class FlakeyStep(PipelineStep):
            def __init__(self):
                super().__init__(name="flakey", provides=["done"])

            def run(self, context):
                nonlocal attempt
                attempt += 1
                if attempt < 3:
                    return CompletedState(success=False, timestamp=CompletedState.now_iso(),
                                          duration_s=0.0, error={"message": "transient"})
                context["done"] = True
                return CompletedState(success=True, timestamp=CompletedState.now_iso(),
                                      duration_s=0.0, provides=["done"])

        def retry_mw(ctx, next_fn):
            for _ in range(5):
                cs = next_fn()
                if cs.success:
                    return cs
            return cs

        p = Pipeline(name="p", steps=[FlakeyStep()], middleware=[retry_mw])
        result = p.run({})
        self.assertTrue(result.success)
        self.assertEqual(attempt, 3)

    def test_empty_middleware_runs_step_raw(self):
        p = Pipeline(name="p", steps=[SetKeyStep("s", "k", 42)], middleware=[])
        ctx: Dict[str, Any] = {}
        result = p.run(ctx)
        self.assertTrue(result.success)
        self.assertEqual(ctx["k"], 42)
        self.assertNotIn("p.s", ctx.get("step_states", {}))


# ===========================================================================
# Phase B (PDG): DAG Solver
# ===========================================================================

class TestTopoSort(unittest.TestCase):
    def test_reorders_based_on_dependencies(self):
        s_a = SetKeyStep("a", "x", 1)
        s_b = SetKeyStep("b", "y", 2, requires=["x"])
        p = Pipeline(name="p", steps=[s_b, s_a], resolve_order=True)
        self.assertEqual([s.name for s in p.steps], ["a", "b"])

    def test_preserves_declared_order_for_independent_steps(self):
        s_a = SetKeyStep("a", "x", 1)
        s_b = SetKeyStep("b", "y", 2)
        s_c = SetKeyStep("c", "z", 3)
        p = Pipeline(name="p", steps=[s_c, s_a, s_b], resolve_order=True)
        self.assertEqual([s.name for s in p.steps], ["c", "a", "b"])

    def test_cycle_detection(self):
        class CycA(PipelineStep):
            def __init__(self):
                super().__init__(name="cyc_a", requires=["y"], provides=["x"])
            def run(self, context):
                pass

        class CycB(PipelineStep):
            def __init__(self):
                super().__init__(name="cyc_b", requires=["x"], provides=["y"])
            def run(self, context):
                pass

        with self.assertRaises(ValueError) as cm:
            Pipeline(name="p", steps=[CycA(), CycB()], resolve_order=True)
        self.assertIn("Cycle detected", str(cm.exception))

    def test_resolve_order_false_is_noop(self):
        s_a = SetKeyStep("a", "x", 1)
        s_b = SetKeyStep("b", "y", 2, requires=["x"])
        p = Pipeline(name="p", steps=[s_b, s_a], resolve_order=False)
        self.assertEqual([s.name for s in p.steps], ["b", "a"])

    def test_no_requires_no_provides_keeps_order(self):
        class PlainStep(PipelineStep):
            def __init__(self, name):
                super().__init__(name=name)
            def run(self, context):
                return CompletedState(success=True, timestamp=CompletedState.now_iso(), duration_s=0.0)

        p = Pipeline(name="p", steps=[PlainStep("z"), PlainStep("a"), PlainStep("m")], resolve_order=True)
        self.assertEqual([s.name for s in p.steps], ["z", "a", "m"])

    def test_topo_sort_with_middleware(self):
        s_a = SetKeyStep("a", "x", 1)
        s_b = SetKeyStep("b", "y", 2, requires=["x"])
        p = Pipeline(name="p", steps=[s_b, s_a], resolve_order=True)
        ctx: Dict[str, Any] = {}
        result = p.run(ctx)
        self.assertTrue(result.success)
        self.assertEqual(ctx["x"], 1)
        self.assertEqual(ctx["y"], 2)


# ===========================================================================
# Phase C (PDG): WorkItem
# ===========================================================================

class TestWorkItemBasics(unittest.TestCase):
    def test_construction_and_defaults(self):
        wi = WorkItem(id="test", attributes={"a": 1})
        self.assertEqual(wi.id, "test")
        self.assertEqual(wi.attributes["a"], 1)
        self.assertEqual(wi.input_files, [])
        self.assertEqual(wi.output_files, [])
        self.assertIsNone(wi.parent_id)
        self.assertEqual(wi.meta, {})

    def test_to_json_from_json_roundtrip(self):
        wi = WorkItem(
            id="rt",
            attributes={"key": "value", "n": 42},
            input_files=["/a.glb"],
            output_files=["/b.json"],
            parent_id="parent",
            meta={"hint": True},
        )
        restored = WorkItem.from_json(wi.to_json())
        self.assertEqual(restored.id, wi.id)
        self.assertEqual(restored.attributes, wi.attributes)
        self.assertEqual(restored.input_files, wi.input_files)
        self.assertEqual(restored.output_files, wi.output_files)
        self.assertEqual(restored.parent_id, wi.parent_id)
        self.assertEqual(restored.meta, wi.meta)

    def test_validate_serializable_rejects_bad_types(self):
        wi = WorkItem(id="bad", attributes={"obj": object()})
        with self.assertRaises(ValueError):
            wi.validate_serializable()

    def test_validate_serializable_accepts_good_types(self):
        wi = WorkItem(id="ok", attributes={"s": "hi", "n": 1, "f": 1.5, "b": True, "none": None})
        wi.validate_serializable()

    def test_generate_id(self):
        id1 = WorkItem.generate_id("model")
        self.assertTrue(id1.startswith("model_"))
        self.assertEqual(len(id1), len("model_") + 8)
        id2 = WorkItem.generate_id()
        self.assertEqual(len(id2), 8)
        self.assertNotEqual(id1, id2)


class TestPipelineRunItem(unittest.TestCase):
    def test_run_context_still_works(self):
        p = Pipeline(name="p", steps=[SetKeyStep("a", "k", 1)])
        ctx: Dict[str, Any] = {}
        result = p.run(ctx)
        self.assertTrue(result.success)
        self.assertEqual(ctx["k"], 1)
        self.assertIn("p.a", ctx["step_states"])

    def test_run_item_with_work_item(self):
        p = Pipeline(name="p", steps=[SetKeyStep("a", "k", 99)])
        wi = WorkItem(id="item1", attributes={})
        result = p.run_item(wi)
        self.assertTrue(result.success)
        self.assertEqual(wi.attributes["k"], 99)
        self.assertIn("p.a", wi.attributes["step_states"])

    def test_multiple_work_items_independent(self):
        p = Pipeline(name="p", steps=[SetKeyStep("a", "k", 1)])
        wi1 = WorkItem(id="w1", attributes={})
        wi2 = WorkItem(id="w2", attributes={})
        p.run_item(wi1)
        p.run_item(wi2)
        self.assertEqual(wi1.attributes["k"], 1)
        self.assertEqual(wi2.attributes["k"], 1)
        self.assertIsNot(wi1.attributes, wi2.attributes)

    def test_nested_pipeline_with_work_item(self):
        inner = Pipeline(name="inner", steps=[SetKeyStep("a", "x", 10)])
        outer = Pipeline(name="outer", steps=[inner])
        wi = WorkItem(id="nested", attributes={})
        result = outer.run_item(wi)
        self.assertTrue(result.success)
        self.assertEqual(wi.attributes["x"], 10)

    def test_dry_run_via_run_item(self):
        call_count = 0

        class Tracker(PipelineStep):
            def __init__(self):
                super().__init__(name="t", provides=["tracked"])

            def run(self, context):
                nonlocal call_count
                call_count += 1
                context["tracked"] = True
                return CompletedState(success=True, timestamp=CompletedState.now_iso(),
                                      duration_s=0.0, provides=["tracked"])

        p = Pipeline(name="p", steps=[Tracker()])
        wi = WorkItem(id="dry", attributes={})
        result = p.run_item(wi, dry_run=True)
        self.assertTrue(result.success)
        self.assertEqual(call_count, 0)


# ===========================================================================
# Phase D (PDG): Fan-Out / Fan-In
# ===========================================================================

class TestFanOutFanIn(unittest.TestCase):
    def _make_counter_step(self):
        counts = {"runs": 0, "items": []}

        class Counter(PipelineStep):
            def __init__(self):
                super().__init__(name="counter", provides=["counted"])

            def run(self, context):
                counts["runs"] += 1
                counts["items"].append(context.get("item_id", "unknown"))
                context["counted"] = True
                return CompletedState(success=True, timestamp=CompletedState.now_iso(),
                                      duration_s=0.0, provides=["counted"])

        return Counter(), counts

    def test_generator_emits_n_items(self):
        class Splitter(GeneratorStep):
            def __init__(self):
                super().__init__(name="split", provides=[])

            def generate(self, work_item):
                return [
                    WorkItem(id=f"item_{i}", attributes={**work_item.attributes, "item_id": i})
                    for i in range(4)
                ]

        counter, counts = self._make_counter_step()
        p = Pipeline(name="p", steps=[Splitter(), counter])
        wi = WorkItem(id="root", attributes={})
        result = p.run_item(wi)
        self.assertTrue(result.success)
        self.assertEqual(counts["runs"], 4)

    def test_collector_merges_items(self):
        class Splitter(GeneratorStep):
            def __init__(self):
                super().__init__(name="split", provides=[])

            def generate(self, work_item):
                return [
                    WorkItem(id=f"i{n}", attributes={**work_item.attributes, "val": n})
                    for n in [10, 20, 30]
                ]

        class Summer(CollectorStep):
            def __init__(self):
                super().__init__(name="sum", provides=["total"])

            def collect(self, work_items):
                total = sum(wi.attributes.get("val", 0) for wi in work_items)
                merged_attrs = dict(work_items[0].attributes)
                merged_attrs["total"] = total
                return WorkItem(
                    id="merged",
                    attributes=merged_attrs,
                )

        class CheckTotal(PipelineStep):
            def __init__(self):
                super().__init__(name="check", requires=["total"], provides=["verified"])

            def run(self, context):
                context["verified"] = context["total"] == 60
                return CompletedState(success=True, timestamp=CompletedState.now_iso(),
                                      duration_s=0.0, provides=["verified"])

        p = Pipeline(name="p", steps=[Splitter(), Summer(), CheckTotal()])
        wi = WorkItem(id="root", attributes={})
        result = p.run_item(wi)
        self.assertTrue(result.success)
        self.assertTrue(wi.attributes.get("verified"))

    def test_fanout_sets_parent_id(self):
        class Splitter(GeneratorStep):
            def __init__(self):
                super().__init__(name="split", provides=[])

            def generate(self, work_item):
                return [WorkItem(id=f"child_{i}", attributes=dict(work_item.attributes)) for i in range(2)]

        parent_ids = []

        class Recorder(CollectorStep):
            def __init__(self):
                super().__init__(name="rec", provides=[])

            def collect(self, work_items):
                for wi in work_items:
                    parent_ids.append(wi.parent_id)
                return work_items[0]

        p = Pipeline(name="p", steps=[Splitter(), Recorder()])
        wi = WorkItem(id="root", attributes={})
        p.run_item(wi)
        self.assertEqual(parent_ids, ["root", "root"])

    def test_empty_generation(self):
        class EmptyGen(GeneratorStep):
            def __init__(self):
                super().__init__(name="empty", provides=[])

            def generate(self, work_item):
                return []

        counter, counts = self._make_counter_step()
        p = Pipeline(name="p", steps=[EmptyGen(), counter])
        wi = WorkItem(id="root", attributes={})
        result = p.run_item(wi)
        self.assertTrue(result.success)
        self.assertEqual(counts["runs"], 1)

    def test_fanout_without_collector(self):
        class Splitter(GeneratorStep):
            def __init__(self):
                super().__init__(name="split", provides=[])

            def generate(self, work_item):
                return [
                    WorkItem(id=f"i{n}", attributes={**work_item.attributes, "val": n})
                    for n in range(3)
                ]

        counter, counts = self._make_counter_step()
        p = Pipeline(name="p", steps=[Splitter(), counter])
        wi = WorkItem(id="root", attributes={})
        result = p.run_item(wi)
        self.assertTrue(result.success)
        self.assertEqual(counts["runs"], 3)

    def test_nested_pipeline_in_fanout(self):
        class Splitter(GeneratorStep):
            def __init__(self):
                super().__init__(name="split", provides=[])

            def generate(self, work_item):
                return [
                    WorkItem(id=f"i{n}", attributes={**work_item.attributes, "val": n})
                    for n in range(2)
                ]

        inner = Pipeline(name="inner", steps=[SetKeyStep("set", "processed", True)])
        p = Pipeline(name="p", steps=[Splitter(), inner])
        wi = WorkItem(id="root", attributes={})
        result = p.run_item(wi)
        self.assertTrue(result.success)


# ===========================================================================
# Phase E (PDG): Schedulers
# ===========================================================================

from lib.scheduler import (
    LocalScheduler,
    PoolScheduler,
    Scheduler,
    SLURMScheduler,
    _run_step_from_identity,
    _step_identity,
)


# -- Helpers ----------------------------------------------------------------

class TestStepIdentity(unittest.TestCase):
    def test_returns_module_and_class(self):
        step = SetKeyStep("s", "k", 1)
        mod, cls = _step_identity(step)
        self.assertIn("test_pipeline", mod)
        self.assertEqual(cls, "SetKeyStep")


class _NoArgStep(PipelineStep):
    """Step with no required constructor args — reconstructable in subprocesses."""

    def __init__(self):
        super().__init__(name="noarg", provides=["noarg_done"])

    def run(self, context):
        context["noarg_done"] = True
        return CompletedState(
            success=True, timestamp=CompletedState.now_iso(),
            duration_s=0.0, provides=["noarg_done"],
        )


class TestRunStepFromIdentity(unittest.TestCase):
    def test_roundtrip(self):
        step = _NoArgStep()
        mod, cls = _step_identity(step)
        wi = WorkItem(id="t", attributes={})
        data = _run_step_from_identity(mod, cls, wi.to_json())
        self.assertTrue(data["success"])


# -- LocalScheduler ---------------------------------------------------------

class TestLocalScheduler(unittest.TestCase):
    def test_produces_identical_result(self):
        step = SetKeyStep("s", "k", 42)
        wi = WorkItem(id="test", attributes={})
        sched = LocalScheduler()
        cs = sched.execute(step, wi)
        self.assertTrue(cs.success)
        self.assertEqual(wi.attributes["k"], 42)

    def test_pipeline_with_local_scheduler(self):
        p = Pipeline(name="p", steps=[SetKeyStep("a", "k", 7)], scheduler=LocalScheduler())
        ctx: Dict[str, Any] = {}
        result = p.run(ctx)
        self.assertTrue(result.success)
        self.assertEqual(ctx["k"], 7)


# -- Scheduler routing (pipeline integration) -------------------------------

class TestSchedulerRouting(unittest.TestCase):
    def test_step_scheduler_overrides_pipeline(self):
        calls: list = []

        class SpyScheduler(Scheduler):
            def __init__(self, name):
                self._name = name

            def execute(self, step, work_item):
                calls.append(self._name)
                return step.run(work_item.attributes)

        pipeline_sched = SpyScheduler("pipeline")
        step_sched = SpyScheduler("step")
        step_with = SetKeyStep("over", "k", 1, scheduler=step_sched)
        step_without = SetKeyStep("default", "j", 2)

        p = Pipeline(name="p", steps=[step_with, step_without], scheduler=pipeline_sched)
        ctx: Dict[str, Any] = {}
        result = p.run(ctx)
        self.assertTrue(result.success)
        self.assertIn("step", calls)
        self.assertIn("pipeline", calls)

    def test_no_scheduler_calls_run_directly(self):
        call_count = 0

        class DirectStep(PipelineStep):
            def __init__(self):
                super().__init__(name="direct", provides=["done"])

            def run(self, context):
                nonlocal call_count
                call_count += 1
                context["done"] = True
                return CompletedState(success=True, timestamp=CompletedState.now_iso(),
                                      duration_s=0.0, provides=["done"])

        p = Pipeline(name="p", steps=[DirectStep()])
        p.run({})
        self.assertEqual(call_count, 1)

    def test_scheduler_applies_to_composite_pipeline_step(self):
        calls: list = []

        class SpyScheduler(Scheduler):
            def execute(self, step, work_item):
                calls.append(type(step).__name__)
                return step.run(work_item.attributes)

        inner = Pipeline(name="inner", steps=[SetKeyStep("a", "k", 1)])
        outer = Pipeline(name="outer", steps=[inner], scheduler=SpyScheduler())
        ctx: Dict[str, Any] = {}
        result = outer.run(ctx)
        self.assertTrue(result.success)
        self.assertIn("Pipeline", calls)

    def test_no_scheduler_composite_runs_inline(self):
        inner = Pipeline(name="inner", steps=[SetKeyStep("a", "k", 1)])
        outer = Pipeline(name="outer", steps=[inner])
        ctx: Dict[str, Any] = {}
        result = outer.run(ctx)
        self.assertTrue(result.success)
        self.assertEqual(ctx["k"], 1)
        self.assertIn("outer.inner.a", ctx["step_states"])


# -- PoolScheduler ----------------------------------------------------------

class TestPoolScheduler(unittest.TestCase):
    def test_executes_step_and_returns_result(self):
        step = _NoArgStep()
        wi = WorkItem(id="test", attributes={})
        with PoolScheduler(workers=1) as sched:
            cs = sched.execute(step, wi)
        self.assertTrue(cs.success)

    def test_multiple_items_produce_independent_results(self):
        results = []
        with PoolScheduler(workers=2) as sched:
            for i in range(4):
                step = _NoArgStep()
                wi = WorkItem(id=f"item_{i}", attributes={})
                cs = sched.execute(step, wi)
                results.append(cs.success)
        self.assertTrue(all(results))

    def test_context_manager_cleanup(self):
        sched = PoolScheduler(workers=1)
        step = _NoArgStep()
        wi = WorkItem(id="test", attributes={})
        with sched:
            sched.execute(step, wi)
        self.assertIsNone(sched._pool)

    def test_pool_reuse(self):
        with PoolScheduler(workers=1) as sched:
            step = _NoArgStep()
            sched.execute(step, WorkItem(id="a", attributes={}))
            pool_id_1 = id(sched._pool)
            sched.execute(step, WorkItem(id="b", attributes={}))
            pool_id_2 = id(sched._pool)
        self.assertEqual(pool_id_1, pool_id_2)


# -- SLURMScheduler ---------------------------------------------------------

class TestSLURMScheduler(unittest.TestCase):
    def test_stores_config(self):
        sched = SLURMScheduler(partition="gpu", gpus=2, time_limit="12:00:00", mem="64GB")
        self.assertEqual(sched.partition, "gpu")
        self.assertEqual(sched.gpus, 2)
        self.assertEqual(sched.time_limit, "12:00:00")
        self.assertEqual(sched.mem, "64GB")

    def test_requires_submitit(self):
        sched = SLURMScheduler(partition="test")
        step = SetKeyStep("s", "k", 1)
        wi = WorkItem(id="test", attributes={})
        try:
            import submitit  # noqa: F401
            pass  # submitit installed — can't test the ImportError path
        except ImportError:
            with self.assertRaises(ImportError) as cm:
                sched.execute(step, wi)
            self.assertIn("submitit", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
