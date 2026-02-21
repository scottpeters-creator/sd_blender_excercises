# Plan of Record — Exercise 3: Procedural Camera Path Generation and Animation

**Status:** In Progress — v6 (normalize scene, perturbed grid, SDF-voxel-matched spacing)
**Created:** 2026-02-18
**v1 Complete:** 2026-02-18 (basic pipeline, BVH raycasting)
**v2 Complete:** 2026-02-19 (SDF negative space, SOP-style granular steps)
**v3-v4:** 2026-02-19 (single-sided repair iterations — superseded by v5)
**v5 Complete:** 2026-02-20 (all spatial ops GN-native: vol scatter + SDF cull)
**v6 In Progress:** 2026-02-20 (normalize scene, perturbed grid, SDF-voxel-matched spacing)

Repository path: `blender_exercises/src/ex_3__proc_camera/`
Context doc: `blender_exercises/context.md` (read this first for framework reference)

---

## Current State

**Working pipeline (v6):** 17-step pipeline. Loads a scene, normalizes unit scale and transforms, builds an iso shell via GN scatter, creates a bbox, scatters a dense candidate grid (spacing matched to SDF voxel resolution) with `Interior Band Width = 1.0` for crisp volume fill, applies seeded random jitter to break lattice bias, culls against the iso shell via GN SDF, then builds a spline path, camera, animation, and renders MP4.

**Design principles:**
- No Python geometry loops. All spatial operations run as GN-native evaluations.
- Candidate grid spacing matches the SDF cull voxel resolution (`diag / 200`).
- Seeded jitter (stratified sampling) breaks lattice bias for ML training while preserving spatial coverage and reproducibility.
- Scene is normalized before any geometry work (unit scale, transforms, modifiers, normals).

**What's solved:**
- `NormalizeSceneStep`: sets `scale_length=1.0`, makes multi-user meshes single, applies transforms/modifiers, fixes negative-scale normals
- GN-native iso mesh: `Distribute Points on Faces → Points to SDF Grid → Grid to Mesh`
- GN-native volume scatter: `Mesh to Volume → Distribute Points in Volume (GRID)` with `Interior Band Width = 1.0`
- GN-native seeded jitter: `Random Value (Vector) → Set Position (Offset)` — deterministic per seed
- GN-native SDF cull: `Mesh to SDF Grid → Sample Grid → Delete Geometry`
- All GN trees auto-laid-out left-to-right via `@auto_layout` decorator
- Live GN modifiers preserved in `.blend` snapshots for introspection
- SDF cull rate ~23% with correct iso shell parameters (1M candidates → 815K surviving)

**What needs work:**
- Points outside the building structure (above roof, beside exterior walls) survive the SDF cull — need interior shell extraction (see Next Steps)
- GN modifier outputs are buried in modifier stacks — not visible as separate objects in outliner, not screenshottable (see Next Steps: BlenderStep artifacts)
- Trajectory ordering (walkthrough vs orbit vs wander) not yet implemented — currently uses angular sort around Z
- The `sample_surface_points` function (for camera aim targets) is slow (~18s) because it triangulates all scene meshes in Python

---

## Next Steps

### 1. BlenderStep Artifact Generation and Screenshot Capture — COMPLETE

Implemented in `blender_step.py`. Every `BlenderStep` now runs three post-execute hooks: `_generate_artifact` (numbered outliner object), `_take_screenshot` (Eevee render framed by debug camera with auto clip planes), `_save_snapshot` (numbered `.blend`). Subclasses override `get_artifact(context)` to return their output object. See `blender_step.py` module docstring for full documentation.

### 2. SDF Attribute Pipeline (Exterior Point Elimination) — v7

**Problem:** The v6 monolithic `CullBySdfWeightStep` computed SDF sampling, normalization, randomization, and culling in one GN tree. Intermediate values were discarded — impossible to debug or visualize. The SDF was sampled twice (once for hard cull, once for weight cull).

**Solution:** Decompose into Houdini-style attribute steps. Each step reads an attribute, transforms it, and stores the result as a named attribute on the points. The SDF is sampled once and cached. Every intermediate value is inspectable.

**Pipeline steps (replacing the old monolithic CullBySdfWeightStep):**

```
Step 8:  CullPointsByMeshStep    → hard cull (SDF < 0, keep outside)
Step 9:  SampleSdfStep           → sample SDF at each point, store as "sdf_distance" float attribute
Step 10: NormalizeSdfStep        → normalize to 0-1, store as "sdf_normalized" attribute
Step 11: RandomizeWeightStep     → weight = (1 - normalized) × seeded noise, store as "weight" attribute
Step 12: ColorizeBySdfStep       → map weight → vertex color attribute for debug visualization
Step 13: CullByWeightStep        → delete points where weight < threshold
```

**GN tree per step:**

- **`build_sample_sdf_tree`**: `Position → Sample Grid (from shell SDF) → Store Named Attribute("sdf_distance")`
- **`build_normalize_attr_tree`**: `Named Attribute("sdf_distance") → Clamp(min=0) → ÷ Attribute Statistic(max) → Store Named Attribute("sdf_normalized")`
- **`build_randomize_weight_tree`**: `Named Attribute("sdf_normalized") → Subtract from 1.0 → × Random Value(seeded) → Store Named Attribute("weight")`
- **`build_colorize_attr_tree`**: `Named Attribute("weight") → ColorRamp(blue→green→red) → Store Named Attribute("color", type=FLOAT_COLOR)`
- **`build_cull_by_attr_tree`**: `Named Attribute("weight") → Compare(< threshold) → Delete Geometry`

**Key properties:**
- SDF sampled once (step 9), cached as point attribute, never re-evaluated
- Every intermediate value inspectable in Blender's spreadsheet editor
- Colorization (step 12) works because the attribute is stored — downstream Instance on Points inherits it
- Each step has its own artifact, screenshot, and snapshot
- Steps are independently skippable/replaceable

**Named attributes on points after each step:**

| After step | Attributes on points |
|---|---|
| Step 8 (hard cull) | position only |
| Step 9 (sample SDF) | position, `sdf_distance` |
| Step 10 (normalize) | position, `sdf_distance`, `sdf_normalized` |
| Step 11 (randomize) | position, `sdf_distance`, `sdf_normalized`, `weight` |
| Step 12 (colorize) | position, `sdf_distance`, `sdf_normalized`, `weight`, `color` |
| Step 13 (cull) | same attributes, fewer points |

### 3. Interior Shell Extraction (Future)

Alternative approach to exterior elimination using mesh connectivity separation. May complement or replace the SDF attribute pipeline for scenes where distance-based weighting isn't sufficient. See `GeometryNodeInputMeshIsland` (Island Index, Island Count) for GN-side labeling, with Python bmesh for bbox volume sorting.

### 4. Trajectory Modes

Not yet designed. Current: angular sort around Z axis. Planned: walkthrough, orbit, wander.

---

## Current File Inventory

| File | What it contains |
|---|---|
| `src/ex_3__proc_camera/proc_camera.py` | Exercise-specific steps (PointsToSplineStep, CreateCameraStep, CurveAnimStep, WriteCameraLogStep) + pipeline composition + CLI |
| `src/ex_3__proc_camera/__init__.py` | Empty package init |
| `src/lib/pipeline_steps/blender_step.py` | `BlenderStep` base class — auto scene snapshots, future: artifact generation + screenshots |
| `src/lib/pipeline_steps/io.py` | `OpenBlendStep` (shared), `ImportModelStep`, `GrepModelsStep` |
| `src/lib/pipeline_steps/scene.py` | `NormalizeSceneStep`, `MergeMeshesStep` (shared), `PrepareSceneStep`, `CleanupSceneStep` |
| `src/lib/pipeline_steps/volume.py` | All volume/SDF/scatter steps (shared): `IsoMeshFromMeshStep`, `CreateBBoxMeshStep`, `ScatterPointsInMeshStep`, `PerturbPointsStep`, `CullPointsByMeshStep`, `OrderPointsStep`, plus legacy steps |
| `src/lib/pipeline_steps/render.py` | `ConfigureRendererStep`, `ConfigureVideoOutputStep`, `RenderAnimationStep` |
| `src/lib/bpy/geometry_nodes.py` | GN tree builders: `build_scatter_to_iso_mesh_tree`, `build_sdf_cull_tree`, `build_vol_scatter_tree`, `build_perturb_tree`, plus others. Auto-layout: `layout_node_tree`, `auto_layout` decorator. Object helpers: `join_scene_meshes`, `create_aabb_box`, `scene_aabb`, `apply_gn_modifier`, `evaluated_mesh_copy`, `read_points_from_evaluated`, `cleanup_temp_data`. |
| `src/lib/bpy/spatial.py` | `sort_by_angle_around_z`, `sample_surface_points` |
| `src/lib/bpy/animation.py` | `create_bezier_path`, `constrain_to_path`, `create_linear_keyframes` |
| `src/lib/bpy/camera.py` | `create_camera`, `frame_objects`, `add_track_to_location` |
| `src/lib/bpy/render.py` | `configure_video_output` (with Blender 5.0 `media_type` fix), `render_animation` |

---

## Pipeline Structure (v6 — Current)

```
proc_camera_e2e (version 6.1, 17 steps)
├── 1.  OpenBlendStep              → scene_loaded
├── 2.  NormalizeSceneStep         → scene_normalized (scale_length=1.0, apply transforms/modifiers, fix normals)
├── 3.  MergeMeshesStep            → merged_mesh
├── 4.  IsoMeshFromMeshStep        → iso_shell (GN: scatter → Points to SDF Grid → Grid to Mesh)
├── 5.  CreateBBoxMeshStep         → bbox_mesh
├── 6.  ScatterPointsInMeshStep    → camera_candidates (GN: Mesh to Volume → Distribute Pts in Vol, spacing=diag/200)
├── 7.  PerturbPointsStep          → camera_candidates_perturbed (GN: seeded Random Value → Set Position offset)
├── 8.  CullPointsByMeshStep       → camera_candidates_culled (GN: SDF cull against iso_shell)
├── 9.  OrderPointsStep            → ordered_candidates
├── 10. PointsToSplineStep         → spline_curve, control_points
├── 11. CreateCameraStep           → camera_object, aim_targets
├── 12. CurveAnimStep              → animation_configured
├── 13. ConfigureRendererStep      → renderer_ready
├── 14. ConfigureVideoOutputStep   → video_configured
├── 15. RenderAnimationStep        → render_path
├── 16. WriteCameraLogStep         → log_written
└── 17. CleanupSceneStep           → cleaned
```

---

## Blender 5.0 API Lessons

1. **`ImageFormatSettings.media_type`** must be set to `"VIDEO"` before `file_format = "FFMPEG"`. Use `hasattr` guard.
2. **GN modifier geometry inputs** — use `GeometryNodeObjectInfo` with `inputs["Object"].default_value = obj` for secondary geometry. First input receives host object mesh automatically.
3. **`Distribute Points in Volume`** crashes with integer overflow on large complex scenes. Works fine for simple bounding volumes (bbox). Use appropriate voxel sizes to avoid overflow.
4. **Poisson Disk on large meshes** OOMs. Use per-object scatter or random mode with appropriate density.
5. **GN node socket discovery**: `ng.nodes.new('NodeType')` then inspect `.inputs` / `.outputs`.
6. **Depsgraph evaluation**: `obj.evaluated_get(bpy.context.evaluated_depsgraph_get())` for GN modifier output.
7. **Temp collection pattern**: `_proc_cam` collection at Scene Collection root for all pipeline objects.
8. **`ConfigureRendererStep` requires override**: `ConfigureRendererStep(requires={"animation_configured"})`.
9. **Live GN modifiers in snapshots**: Keep the modifier live so users can inspect/tweak the GN tree in the `.blend` snapshot. Also create an evaluated copy as a separate artifact for outliner visibility.
10. **Ray-parity inside/outside test** is unreliable for multi-shell meshes. Use SDF-based (`Mesh to SDF Grid → Sample Grid`, negative = inside) or normal dot product method instead.
11. **GN scatter parameters for Evermotion scene**: `density=0.010, radius=20.0, voxel=15.0` (in Blender internal units). These produce a connected iso shell (~57K verts) with 23% SDF cull rate.
12. **Blender 5.0 GN menu sockets**: Node modes (e.g., `Mesh to Volume` Resolution Mode, `Distribute Points in Volume` Mode) are `MENU`-type input sockets, not Python properties. Set via `node.inputs["Mode"].default_value = "Grid"`. Valid values for `Mesh to Volume` Resolution Mode: `"Amount"`, `"Size"`.
13. **`Mesh to Volume` Interior Band Width**: The minimum search radius inward from the surface, NOT a gradient width. Use a small value (1.0) for a crisp solid fill of closed meshes. Large values cause the density gradient to spread, and points in the center fall below the `Distribute Points in Volume` threshold.
14. **GN PointCloud output not readable via Python depsgraph**: `Distribute Points in Volume` outputs a PointCloud. When a GN modifier on a Mesh object produces PointCloud geometry, `eval_obj.data.vertices` is empty and `data.attributes["position"]` returns 0 items. Fix: append `Instance on Points` (with a single-vertex Mesh Line) + `Realize Instances` at the end of the GN tree to convert PointCloud → Mesh vertices.
15. **GN auto-layout**: All `build_*` tree builders use the `@auto_layout` decorator which runs `layout_node_tree()` after construction. Topological sort positions nodes left-to-right with consistent spacing. New builders should always use this decorator.
16. **Candidate grid spacing = SDF voxel size** (`diag / 200`). Matching the grid to the SDF resolution ensures no wall boundary falls between two grid points undetected. Seeded jitter (`PerturbPointsStep`) then breaks the lattice bias for ML training.
17. **`scale_length` affects GN dimension sockets**: Blender stores GN socket values in internal BU. The UI displays `value * scale_length` with the unit suffix. With `scale_length=0.01`, setting `default_value=0.2` displays as `0.002 m`. Fix: normalize `scale_length=1.0` early in the pipeline via `NormalizeSceneStep` so code values = UI values. Do NOT rescale geometry — only change the display setting.
18. **Multi-user meshes block `transform_apply`**: Blender refuses to apply transforms on objects sharing mesh data. Fix: `obj.data = obj.data.copy()` to make single-user before applying.
19. **GN cannot create scene objects**: Geometry Nodes operate on geometry within a single object — they cannot create new objects in the outliner. To make GN output visible as a separate inspectable object, the pipeline step must read the evaluated data back to Python and create a new object via `bpy.data.objects.new()`.
20. **`GeometryNodeInputMeshIsland`**: Outputs `Island Index` (int per face) and `Island Count` for connectivity separation. Available in Blender 5.0. Useful for separating inner/outer shells of iso meshes.
21. **No custom Python GN nodes**: Blender does not support running arbitrary Python inside a GN tree. The GN engine is compiled C++. For logic that GN can't express (e.g., sort islands by bbox volume), use Python for analysis and feed results back as GN parameters or selections.

---

## CullPointsByMeshStep Methods

The step supports three culling methods:

| Method | How it works | Speed | Accuracy |
|---|---|---|---|
| `"raycast"` | BVH ray-parity (+Z ray, count hits) | Python loop, moderate | Unreliable for multi-shell |
| `"normal"` | BVH nearest surface + normal dot product | Python loop, moderate | Good for manifold meshes |
| `"sdf"` | GN: Mesh to SDF Grid → Sample Grid → Delete Geometry | GN native, fast | Best — works for any manifold mesh |

---

## CLI

```bash
python -m ex_3__proc_camera.proc_camera scene.blend output/ --mode spline
python -m ex_3__proc_camera.proc_camera scene.blend output/ --mode point --threshold 0.3
python -m ex_3__proc_camera.proc_camera scene.blend output/ --mode spline --seed 42 --frames 10
python -m ex_3__proc_camera.proc_camera scene.blend output/ --mode spline --dry-run
```

---

## Version History

### v1 (2026-02-18) — Basic Pipeline
- Monolithic GenerateCameraPathStep with random AABB sampling + BVH raycasting

### v2 (2026-02-19) — SDF Negative Space + SOP-Style Steps
- 15 granular steps, SDF boolean (box - geo), Blender 5.0 GN
- Problem: SDF holes from single-sided walls

### v3 (2026-02-19) — Single-Sided Surface Repair
- ExtractSingleSided + SingleSidedToIsoMesh + per-face scatter + KDTree relaxation
- Problem: slow Python scatter (170s), SDF boolean still leaky

### v4 (2026-02-19) — Simplified Pipeline (iso whole scene)
- Removed SDF boolean entirely. Scatter on all faces → iso shell → cull by BVH
- Problem: random scatter quality poor, ray-parity cull unreliable on multi-shell

### v5 (2026-02-20) — All-GN-Native Spatial Pipeline
- All spatial ops are GN-native — no Python geometry loops
- Step 3 (iso mesh): GN `Distribute Points on Faces → Points to SDF Grid → Grid to Mesh`
- Step 5 (vol scatter): GN `Mesh to Volume → Distribute Points in Volume (GRID)`
- Step 6 (SDF cull): GN `Mesh to SDF Grid → Sample Grid → Delete Geometry`
- All GN trees auto-laid-out via `@auto_layout` decorator
- Iso mesh params fixed: `density=0.010, radius=20.0, voxel=15.0`

### v6 (2026-02-20) — Normalize Scene + Perturbed Grid (Current)
- New `NormalizeSceneStep` (step 2): sets `scale_length=1.0` (display only, no geometry rescaling), makes multi-user meshes single, applies transforms/modifiers, fixes negative-scale normals
- `Mesh to Volume` Interior Band Width fixed to `1.0` — small value gives crisp solid fill (large values spread the density gradient, causing center dropout)
- Candidate grid spacing matched to SDF cull voxel size (`diag / 200`) — 1M candidates filling full bbox
- New `PerturbPointsStep`: GN-native seeded jitter (`Random Value → Set Position`) breaks lattice bias for ML training
- Pipeline: 17 steps, v6.1, deterministic via `--seed`
- SDF cull rate ~23% with correct iso shell parameters
- Next: BlenderStep artifact generation + screenshots, interior shell extraction for exterior point elimination, trajectory modes
