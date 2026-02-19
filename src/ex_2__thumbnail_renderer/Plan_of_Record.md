# Plan of Record — Exercise 2: Thumbnail Rendering with Multiple Modalities

**Status:** In Progress (functional, needs visual tuning)
**Created:** 2026-02-18

## Overview

Pipeline-driven thumbnail renderer that imports a 3D model and produces four 512x512 PNG images (textured, normal, depth, edge) plus a `metadata.json` with camera intrinsics and bounding box.

Repository path: `blender_excercises/src/ex_2__thumbnail_renderer/`

## What Has Been Implemented

### Pipeline structure

```
thumbnail_renderer_e2e
├── scene_prep
│   ├── PrepareSceneStep       → scene.reset_scene()
│   └── ImportModelStep        → io.import_model()
├── camera_setup
│   └── SetupCameraStep        → camera.create_camera() + camera.frame_objects()
├── lighting
│   └── SetupEnvironmentLightStep → white environment world shader
├── render
│   ├── ConfigureRendererStep  → render.configure_cycles() + render.enable_gpu()
│   ├── RenderTexturedStep     → original materials, standard render
│   ├── RenderNormalStep       → normal shader material override
│   ├── RenderDepthStep        → Z-depth shader material override
│   └── RenderEdgeStep         → Freestyle line-art + white material override
├── metadata
│   └── WriteMetadataStep      → camera intrinsics + bounding box JSON
└── cleanup
    └── CleanupSceneStep       → scene.reset_scene()
```

All steps are imported from `lib/pipeline_steps/` — the exercise script is pure composition + CLI.

### Rendering approach

Each modality is rendered as a **separate Cycles render** with material overrides:

- **Textured** — standard render with original materials + environment light
- **Normal** — GeometryNode → Emission shader override on all meshes
- **Depth** — CameraData Z Depth → MapRange (near=white, far=black) → Emission override
- **Edge** — Freestyle enabled, white Emission override, `as_render_pass` for isolation

This approach was chosen over the multi-pass compositor method because Blender 5.0 broke the `CompositorNodeOutputFile` API (`base_path` → `directory`, `CompositorNodeComposite` removed, node-level format locked to EXR). Four separate renders at 512x512 complete in ~1.7 seconds total, which is acceptable.

### Output format

```
{output_dir}/{model_stem}/
├── render_textured.png    (512x512)
├── render_normal.png      (512x512)
├── render_depth.png       (512x512, grayscale)
├── render_edge.png        (512x512, black lines on white)
└── metadata.json
```

### CLI

```bash
# Single file
python -m ex_2__thumbnail_renderer.render_task model.glb output_dir/

# Batch
python -m ex_2__thumbnail_renderer.render_task /path/to/models/ /path/to/renders/ --type glb --limit 10

# Custom samples
python -m ex_2__thumbnail_renderer.render_task model.glb output_dir/ --samples 64
```

### Verified working

- Single-file render produces all 5 output files
- Batch mode with GrepModelsStep fan-out works correctly (depth-first per model)
- Environment light properly illuminates the textured render
- macOS CPU fallback works (GPU rendering on Linux/HPC via enable_gpu())

## Blender 5.0 API Issues Encountered

| Issue | Resolution |
|---|---|
| `scene.node_tree` removed | Use `scene.compositing_node_group` + create via `bpy.data.node_groups.new()` |
| `CompositorNodeComposite` removed | Not needed — File Output nodes are the only output mechanism |
| `CompositorNodeOutputFile.base_path` removed | Use `directory` and `file_name` properties |
| Node-level `format.file_format` locked to EXR | Switched to 4 separate renders with material overrides |
| `scene.use_nodes` deprecated (removal in 6.0) | Added `_ensure_compositor_tree()` compat helper |
| `lineset.linestyle` is None after creation | Create linestyle explicitly via `bpy.data.linestyles.new()` |
| `Material.use_nodes` deprecated (removal in 6.0) | Still works, emits warning — monitor for Blender 6.0 |

## Remaining Work

- Visual quality tuning (render samples, camera angle, environment light strength)
- Run batch on 1k models and upload results
- README.md with method explanation

## Steps Used (all from `lib/pipeline_steps/`)

| Module | Step | Source |
|---|---|---|
| `scene.py` | PrepareSceneStep, CleanupSceneStep | Shared |
| `io.py` | ImportModelStep, GrepModelsStep | Shared |
| `camera.py` | SetupCameraStep | Shared |
| `lighting.py` | SetupEnvironmentLightStep | Shared |
| `render.py` | ConfigureRendererStep, RenderTexturedStep, RenderNormalStep, RenderDepthStep, RenderEdgeStep | Shared |
| `reporting.py` | WriteMetadataStep | Shared |
