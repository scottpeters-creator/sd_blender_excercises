# Plan of Record — Exercise 1: mini_inspector

**Status:** Complete
**Completed:** 2026-02-18

## Overview

Pipeline-driven mini inspector that analyzes 3D model files (.glb, .gltf, .obj, .fbx, .stl, .blend) and produces JSON metadata reports containing geometry counts, topology breakdown, world-space bounding box, and material names.

Repository path: `blender_excercises/src/ex_1__mini_inspector/`

## Deliverables

| File | Purpose |
|---|---|
| `mini_inspector.py` | Steps, pipeline composition, batch runner, CLI |
| `__init__.py` | Package init |
| `Plan_of_Record.md` | This file |

## What Was Implemented

### Pipeline structure

```
mini_inspector_e2e (single file)
├── scene_prep
│   ├── PrepareSceneStep        → scene.reset_scene()
│   └── ImportModelStep         → io.import_model(path)
├── analysis
│   ├── CollectMeshesStep       → mesh.collect_meshes()
│   └── ComputeCountsStep      → mesh.compute_geometry_counts()
│                                 + mesh.compute_topology()
│                                 + mesh.compute_world_bbox()
├── reporting
│   ├── AssembleReportStep      → assembles JSON report + mesh.get_material_names()
│   └── WriteJSONStep           → write_json_atomic() with ensure_directory()
└── cleanup
    └── CleanupSceneStep        → scene.reset_scene()
```

### Batch mode

```
mini_inspector_batch
├── GrepModelsStep (GeneratorStep)  → scans dir, emits 1 WorkItem per file
└── inspect_model (Pipeline)        → the full e2e pipeline above, runs depth-first per item
```

`GrepModelsStep` is a `GeneratorStep` that fan-outs a directory into N WorkItems. The inner `inspect_model` pipeline is a single composite step so that the entire pipeline runs to completion for each model before starting the next — this is critical because Blender's `bpy` scene is global mutable state.

### CLI

```bash
# Single file
python -m ex_1__mini_inspector.mini_inspector model.glb report.json

# Batch — all .glb files in a directory
python -m ex_1__mini_inspector.mini_inspector /path/to/models/ /path/to/reports/ --type glb

# Batch — first 30 only
python -m ex_1__mini_inspector.mini_inspector /path/to/models/ /path/to/reports/ --type glb --limit 30

# Dry-run (preview plan without executing)
python -m ex_1__mini_inspector.mini_inspector /path/to/models/ /path/to/reports/ --type glb --limit 5 --dry-run

# Force re-run (ignore idempotent skip)
python -m ex_1__mini_inspector.mini_inspector model.glb report.json --force
```

### Output format

Per-model subdirectory with `report.json`:

```
reports/
├── 000074a334c541878360457c672b6c2e/
│   └── report.json
├── 0000ecca9a234cae994be239f6fec552/
│   └── report.json
└── ...
```

Report schema:

```json
{
  "filename": "000074a334c541878360457c672b6c2e.glb",
  "geometry": {"vertices": 8454, "faces": 14884},
  "topology": {"triangles": 14884, "quads": 0, "ngons": 0},
  "bounding_box": {
    "center": [-0.0138, -0.025, 0.0378],
    "dimensions": [0.3129, 0.2448, 0.0756],
    "min": [-0.1702, -0.1474, 0.0],
    "max": [0.1427, 0.0974, 0.0756]
  },
  "materials": ["sh_flipflop"],
  "mesh_count": 1
}
```

### Input data

Validated against 300 objaverse .glb files at `/Users/scott.peters/dev/source/scott_data/objaverse/`.
Performance: ~2 seconds per model (including scene reset and cleanup).

## Framework features used

- `Pipeline` composition (4 nested sub-pipelines)
- `GeneratorStep` fan-out for batch processing
- `WorkItem` for per-model independent state
- `OutputNamer` for deterministic per-model subdirectory output paths
- `ensure_directory()` for automatic output directory creation
- `write_json_atomic()` for crash-safe JSON output
- Idempotent skip (steps not re-run on second invocation)
- Dry-run mode
- `exit_code_from()` for CLI exit codes

## Lessons learned

1. **Depth-first execution for bpy.** Fan-out with bpy-dependent steps must wrap the entire per-model pipeline as a single composite `Pipeline` step. Breadth-first (running step N for all items before step N+1) corrupts the scene because `bpy` is global mutable state.

2. **Deferred bpy imports.** All `from lib.bpy.* import ...` calls are inside `run()` methods, not at module top level. This lets the module be imported for pipeline composition and unit testing without bpy present.

3. **`provides` validation.** Every step that declares `provides=["key"]` must actually set `context["key"]` in `run()`. The orchestrator's validation middleware catches missing keys. (We hit this with `WriteJSONStep` initially.)

4. **`OutputNamer` for batch output.** Per-model subdirectories (`{output_dir}/{stem}/report.json`) scale better than flat naming across all exercises, especially when multiple output files per model are needed (Ex 2: 4 renders + metadata).

## Dependencies

- `lib.pipeline` — Pipeline, PipelineStep, GeneratorStep, CompletedState, etc.
- `lib.work_item` — WorkItem
- `lib.naming` — OutputNamer, ensure_directory
- `lib.bpy.scene` — reset_scene
- `lib.bpy.io` — import_model
- `lib.bpy.mesh` — collect_meshes, compute_geometry_counts, compute_topology, compute_world_bbox, get_material_names

## What this exercise exports for reuse

Other exercises can import and embed:

```python
from ex_1__mini_inspector.mini_inspector import (
    scene_prep,        # Pipeline: PrepareSceneStep + ImportModelStep
    analysis,          # Pipeline: CollectMeshesStep + ComputeCountsStep
    reporting,         # Pipeline: AssembleReportStep + WriteJSONStep
    cleanup,           # Pipeline: CleanupSceneStep
    build_e2e_pipeline,    # full single-file pipeline
    build_batch_pipeline,  # full batch pipeline with GrepModelsStep
    GrepModelsStep,        # reusable directory scanner
)
```
