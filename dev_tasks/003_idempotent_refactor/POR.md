# Plan of Record — 003: Pipeline Steps Categorical Refactor

**Status:** Not Started
**Created:** 2026-02-19
**Depends on:** Ex 3 completion

## Overview

Reorganize `lib/pipeline_steps/` from a flat module layout into categorical subfolders. Each subfolder groups steps by what they do, not which exercise uses them. This makes the step library browsable and composable — a new exercise picks from the catalog rather than reading exercise-specific code.

## Current Structure (flat)

```
lib/pipeline_steps/
├── blender_step.py        ← base class
├── __init__.py
├── scene.py               ← PrepareScene, MergeMeshes, Cleanup
├── io.py                  ← OpenBlend, ImportModel, GrepModels
├── analysis.py            ← CollectMeshes, ComputeCounts
├── camera.py              ← SetupCamera
├── lighting.py            ← SetupEnvironmentLight
├── render.py              ← ConfigureRenderer, ConfigureVideoOutput, RenderAnimation, RenderTextured, etc.
├── reporting.py           ← WriteJSON, WriteMetadata
└── volume.py              ← BBoxVolume, GeomVolume, VolumeDiff, ScatterPoints, OrderPoints
```

## Proposed Structure (categorical)

```
lib/pipeline_steps/
├── blender_step.py            ← base class (stays at root)
├── __init__.py                ← re-exports for backward compatibility
├── geom/                      ← geometry operations
│   ├── __init__.py
│   ├── scene.py               ← MergeMeshes, ExtractSingleSided
│   ├── volume.py              ← BBoxVolume, GeomVolume, VolumeDiff, ScatterPoints, OrderPoints
│   ├── mesh_repair.py         ← SingleSidedToIsoMesh, future: remesh, decimate
│   └── analysis.py            ← CollectMeshes, ComputeCounts
├── anim/                      ← animation operations
│   ├── __init__.py
│   └── camera.py              ← SetupCamera, future: orbit, path constraints
├── scene/                     ← scene lifecycle and I/O
│   ├── __init__.py
│   ├── io.py                  ← OpenBlend, ImportModel, GrepModels
│   └── lifecycle.py           ← PrepareScene, CleanupScene
├── render/                    ← render operations
│   ├── __init__.py
│   └── render.py              ← ConfigureRenderer, ConfigureVideoOutput, RenderAnimation,
│                                  RenderTextured, RenderNormal, RenderDepth, RenderEdge
├── reporting/                 ← output/reporting
│   ├── __init__.py
│   └── reporting.py           ← WriteJSON, WriteMetadata
└── lighting/                  ← lighting setup
    ├── __init__.py
    └── lighting.py            ← SetupEnvironmentLight
```

## Category Descriptions

| Category | Purpose | Houdini Analogy |
|---|---|---|
| `geom/` | Geometry manipulation: merge, repair, volume ops, analysis | SOP context |
| `anim/` | Animation: camera setup, keyframing, constraints | CHOP/OBJ context |
| `scene/` | Scene lifecycle: open, import, prepare, cleanup | File I/O + scene management |
| `render/` | Render configuration and execution | ROP context |
| `reporting/` | Output file writing (JSON, metadata) | Post-processing |
| `lighting/` | Light setup, environment | Part of SOP/OBJ |

## Backward Compatibility

The `__init__.py` at the `pipeline_steps/` root will re-export all step classes from their new locations, so existing imports in Ex 1 and Ex 2 continue to work:

```python
# These still work after the refactor:
from lib.pipeline_steps.scene import PrepareSceneStep, CleanupSceneStep
from lib.pipeline_steps.io import ImportModelStep, GrepModelsStep
from lib.pipeline_steps.render import ConfigureRendererStep, RenderTexturedStep
```

The `__init__.py` maps old module paths to new locations. No exercise code changes needed.

## Implementation Order

1. Create the subfolder structure with `__init__.py` files
2. Move existing steps to their categorical homes
3. Update `lib/pipeline_steps/__init__.py` for backward-compatible re-exports
4. Verify Ex 1, Ex 2, Ex 3 still work (imports resolve correctly)
5. Run unit tests (96) + all three exercises
