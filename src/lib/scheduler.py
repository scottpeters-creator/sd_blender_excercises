"""Scheduler abstraction for pipeline execution.

Schedulers decouple *what to do* from *where to do it*.

- LocalScheduler:      run step in-process (default, current behavior).
- PoolScheduler:       persistent pool of worker processes (amortized startup).
- SubprocessScheduler: spawn a fresh process per work item.
- SLURMScheduler:      submit work item as an HPC batch job via submitit.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import logging
import os
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Optional

from lib.pipeline import CompletedState, PipelineStep
from lib.work_item import WorkItem

logger = logging.getLogger("pipeline.scheduler")


# ---------------------------------------------------------------------------
# Shared serialization helpers
# ---------------------------------------------------------------------------

def _step_identity(step: PipelineStep) -> tuple:
    """Return (module_name, class_name) so the step can be reconstructed remotely."""
    return type(step).__module__, type(step).__name__


def _run_step_from_identity(
    step_module: str,
    step_class: str,
    work_item_json: str,
) -> dict:
    """Reconstruct a step in an isolated process, run it, return result as dict.

    This is the function that executes inside worker processes / subprocesses.
    It must be importable at the top level for pickling (multiprocessing).
    """
    mod = importlib.import_module(step_module)
    cls = getattr(mod, step_class)
    step = cls()
    wi = WorkItem.from_json(work_item_json)
    cs = step.run(wi.attributes)
    return dataclasses.asdict(cs)


# ---------------------------------------------------------------------------
# Scheduler ABC
# ---------------------------------------------------------------------------

class Scheduler(ABC):
    """Abstract execution backend for a pipeline step + work item."""

    @abstractmethod
    def execute(self, step: PipelineStep, work_item: WorkItem) -> CompletedState:
        """Run *step* against *work_item* and return the result."""
        raise NotImplementedError()  # pragma: no cover


# ---------------------------------------------------------------------------
# LocalScheduler
# ---------------------------------------------------------------------------

class LocalScheduler(Scheduler):
    """Run the step in the current process (same as calling step.run directly)."""

    def execute(self, step: PipelineStep, work_item: WorkItem) -> CompletedState:
        return step.run(work_item.attributes)


# ---------------------------------------------------------------------------
# SubprocessScheduler
# ---------------------------------------------------------------------------

class SubprocessScheduler(Scheduler):
    """Spawn a fresh Python process per work item.

    Serializes the WorkItem to a temp JSON file, invokes a runner script,
    and deserializes the CompletedState from an output JSON file.

    Context mutations made by the step in the subprocess are *not*
    propagated back.  Only the CompletedState return value survives.
    """

    def __init__(
        self,
        python: str = sys.executable,
        timeout: Optional[float] = None,
    ) -> None:
        self.python = python
        self.timeout = timeout

    def execute(self, step: PipelineStep, work_item: WorkItem) -> CompletedState:
        in_path = out_path = None
        try:
            fd_in, in_path = tempfile.mkstemp(suffix=".json", prefix="wi_in_")
            with os.fdopen(fd_in, "w") as f_in:
                f_in.write(work_item.to_json())

            out_path = in_path.replace("wi_in_", "wi_out_")
            step_module, step_class = _step_identity(step)

            runner = (
                "import json, sys, os, dataclasses, importlib\n"
                "from lib.work_item import WorkItem\n"
                "from lib.pipeline import CompletedState\n"
                f"mod = importlib.import_module('{step_module}')\n"
                f"cls = getattr(mod, '{step_class}')\n"
                "step = cls()\n"
                f"wi = WorkItem.from_json(open(r'{in_path}').read())\n"
                "cs = step.run(wi.attributes)\n"
                f"with open(r'{out_path}', 'w') as f:\n"
                "    json.dump(dataclasses.asdict(cs), f)\n"
            )

            result = subprocess.run(
                [self.python, "-c", runner],
                capture_output=True, text=True,
                timeout=self.timeout,
            )

            if result.returncode != 0:
                return CompletedState(
                    success=False,
                    timestamp=CompletedState.now_iso(),
                    duration_s=0.0,
                    error={
                        "type": "SubprocessError",
                        "message": result.stderr.strip() or f"exit code {result.returncode}",
                    },
                )

            with open(out_path) as f_out:
                data = json.load(f_out)
            return CompletedState(**data)

        except subprocess.TimeoutExpired:
            return CompletedState(
                success=False,
                timestamp=CompletedState.now_iso(),
                duration_s=0.0,
                error={
                    "type": "TimeoutError",
                    "message": f"Subprocess timed out after {self.timeout}s",
                },
            )
        except Exception as exc:
            return CompletedState(
                success=False,
                timestamp=CompletedState.now_iso(),
                duration_s=0.0,
                error={"type": type(exc).__name__, "message": str(exc)},
            )
        finally:
            for path in (in_path, out_path):
                if path:
                    try:
                        os.remove(path)
                    except OSError:
                        pass


# ---------------------------------------------------------------------------
# PoolScheduler
# ---------------------------------------------------------------------------

class PoolScheduler(Scheduler):
    """Persistent pool of worker processes for amortized process isolation.

    Each worker imports bpy (or any heavy dependency) once at startup.
    Work items are dispatched to available workers, giving full memory
    isolation without per-item process startup cost.

    Uses ``multiprocessing.get_context("spawn")`` — safe for bpy and
    other C extensions that are not fork-safe.

    Args:
        workers:  Number of worker processes (default: 4).
        timeout:  Per-item timeout in seconds (default: None = no timeout).
    """

    def __init__(
        self,
        workers: int = 4,
        timeout: Optional[float] = None,
    ) -> None:
        self.workers = workers
        self.timeout = timeout
        self._pool = None

    def _get_pool(self):
        if self._pool is None:
            import multiprocessing
            ctx = multiprocessing.get_context("spawn")
            self._pool = ctx.Pool(processes=self.workers)
            logger.info("PoolScheduler: spawned %d worker(s)", self.workers)
        return self._pool

    def execute(self, step: PipelineStep, work_item: WorkItem) -> CompletedState:
        pool = self._get_pool()
        step_module, step_class = _step_identity(step)
        wi_json = work_item.to_json()

        try:
            async_result = pool.apply_async(
                _run_step_from_identity,
                args=(step_module, step_class, wi_json),
            )
            data = async_result.get(timeout=self.timeout)
            return CompletedState(**data)
        except Exception as exc:
            err_type = type(exc).__name__
            if "TimeoutError" in err_type or "timeout" in str(exc).lower():
                return CompletedState(
                    success=False,
                    timestamp=CompletedState.now_iso(),
                    duration_s=0.0,
                    error={
                        "type": "TimeoutError",
                        "message": f"Pool worker timed out after {self.timeout}s",
                    },
                )
            return CompletedState(
                success=False,
                timestamp=CompletedState.now_iso(),
                duration_s=0.0,
                error={"type": err_type, "message": str(exc)},
            )

    def shutdown(self) -> None:
        """Terminate worker processes and release resources."""
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
            logger.info("PoolScheduler: workers shut down")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()

    def __del__(self):
        self.shutdown()


# ---------------------------------------------------------------------------
# SLURMScheduler
# ---------------------------------------------------------------------------

class SLURMScheduler(Scheduler):
    """Submit work item as an HPC batch job via submitit.

    Requires ``pip install submitit``.  Each call to ``execute()`` submits
    an sbatch job and blocks until the job completes (or times out).

    Args:
        partition:   SLURM partition name.
        gpus:        GPUs per node.
        time_limit:  Wall-clock time limit (e.g., ``"2:00:00"``).
        mem:         Memory per node (e.g., ``"32GB"``).
        log_dir:     Directory for SLURM log files.
        **kwargs:    Extra parameters forwarded to ``submitit.AutoExecutor.update_parameters()``.
    """

    def __init__(
        self,
        partition: str = "default",
        gpus: int = 0,
        time_limit: str = "1:00:00",
        mem: str = "16GB",
        log_dir: str = "slurm_logs",
        **kwargs: Any,
    ) -> None:
        self.partition = partition
        self.gpus = gpus
        self.time_limit = time_limit
        self.mem = mem
        self.log_dir = log_dir
        self.extra = kwargs
        self._executor = None

    def _get_executor(self):
        if self._executor is None:
            try:
                import submitit
            except ImportError as exc:
                raise ImportError(
                    "SLURMScheduler requires the 'submitit' package. "
                    "Install it with: pip install submitit"
                ) from exc

            os.makedirs(self.log_dir, exist_ok=True)
            self._executor = submitit.AutoExecutor(folder=self.log_dir)
            self._executor.update_parameters(
                slurm_partition=self.partition,
                slurm_gpus_per_node=self.gpus,
                slurm_time=self.time_limit,
                slurm_mem=self.mem,
                **self.extra,
            )
            logger.info(
                "SLURMScheduler: executor ready (partition=%s, gpus=%d, time=%s)",
                self.partition, self.gpus, self.time_limit,
            )
        return self._executor

    def execute(self, step: PipelineStep, work_item: WorkItem) -> CompletedState:
        executor = self._get_executor()
        step_module, step_class = _step_identity(step)
        wi_json = work_item.to_json()

        try:
            job = executor.submit(_run_step_from_identity, step_module, step_class, wi_json)
            logger.info("SLURMScheduler: submitted job %s for work item '%s'", job.job_id, work_item.id)
            data = job.result()
            return CompletedState(**data)
        except Exception as exc:
            return CompletedState(
                success=False,
                timestamp=CompletedState.now_iso(),
                duration_s=0.0,
                error={"type": type(exc).__name__, "message": str(exc)},
            )
