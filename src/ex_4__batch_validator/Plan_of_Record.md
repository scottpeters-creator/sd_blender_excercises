# Plan of Record — Exercise 4: Batch Validator & S3 Integration

## Overview

Purpose: implement a resilient CLI tool that recursively lists `.glb` objects from an S3 prefix, downloads and analyzes them using standalone `bpy`, and uploads per-file JSON reports back to S3. The tool must handle pagination, failures, and memory management so it can run reliably over thousands of assets.

Repository path: `blender_excercises/src/ex_4__batch_validator`

## Scope
- Implement `batch_validator.py` runnable in a standard Python environment (with `bpy` accessible) that:
  - Lists `.glb` objects under a provided S3 `bucket` and `prefix` using `boto3` with pagination.
  - Downloads files (serial or small-batch mode), processes them through `bpy` to extract geometry stats, uploads JSON reports to S3, and removes local files.
  - Robustly handles corrupt files by uploading an error JSON and continuing.

## Objectives / Success Criteria
- Proper pagination of S3 listings and honoring `--limit` argument.
- Reports uploaded back to S3 at same path with `_report.json` suffix for each `.glb` processed.
- Memory-safe Blender processing: explicit clearing of `bpy.data` (meshes, materials, textures) between files and/or `bpy.ops.wm.read_homefile(use_empty=True)`.

## Requirements & Constraints
- Language: Python 3.10+.
- Libraries: `boto3`, `argparse`, `bpy` (standalone), and stdlib (`logging`, `json`, `os`, `sys`, `tempfile`).
- Execution: CLI on a machine with `bpy` available (either Blender Python or `bpy` wheel) and AWS credentials available in environment or instance role.

## Implementation Approach (high level)
1. Discovery
   - Use `boto3.client('s3').get_paginator('list_objects_v2')` to iterate objects under prefix and filter for `.glb` suffix.
2. Processing strategy
   - Default mode: Serial (download → process → upload → delete) to minimize local disk usage.
   - Optional `--batch-size` parameter for small caching workflows (e.g., 10 files) if the operator prefers throughput over minimal disk IO.
3. Analysis
   - For each file: import via `bpy.ops.import_scene.gltf` (or OBJ), compute vertices, faces, material names, and AABB dimensions.
   - After processing, explicitly remove `bpy.data.meshes[:]`, `bpy.data.materials[:]`, `bpy.data.images[:]`, and call `bpy.ops.wm.read_homefile(use_empty=True)` when appropriate.
4. Upload
   - Put JSON to S3 path replacing `.glb` with `_{uid}_report.json` or `{basename}_report.json`.
5. Error handling
   - Wrap analysis in try/except; on exception, upload error JSON with `status: error` and continue.

## Deliverables
1. `batch_validator.py` — main script.
2. `README.md` — usage, architecture choice (serial vs batch), and memory strategy explanation.
3. `Plan_of_Record.md` — this file.

## Timeline & Milestones (suggested)
- Day 0: POR and scaffolding.
- Day 1: Implement S3 listing and serial processing flow; run small-scale test (10 files).
- Day 2: Harden memory cleanup and error handling; test on 100 files.
- Day 3: Add optional batch mode and finalize README.

## Testing & Validation
- Functional test: run with `--limit 5` and inspect S3 for report JSONs.
- Fault injection: process a known-bad file and verify error report upload and continued processing.
- Long-run test: run with `--limit 200` and monitor memory usage.

## Risks & Mitigations
- Risk: Blender memory leaks across iterations. Mitigation: aggressively clear `bpy.data` and reload empty file; optionally restart process worker periodically.
- Risk: Network or S3 rate limits. Mitigation: implement retry/backoff for S3 operations and expose concurrency knobs.

## Implementation Notes / APIs
- S3 listing: `paginator = s3.get_paginator('list_objects_v2')` then `for page in paginator.paginate(Bucket=bucket, Prefix=prefix):`.
- Download to `tempfile.NamedTemporaryFile(delete=False)` then pass path to Blender importers.
- JSON upload: `s3.put_object(Bucket=bucket, Key=report_key, Body=json.dumps(payload).encode('utf-8'))`.

## Next Steps
1. Confirm POR. I will scaffold `batch_validator.py` and `README.md` next upon confirmation.
