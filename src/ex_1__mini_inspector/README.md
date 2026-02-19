# mini_inspector — README

Purpose: A small headless Blender script that inspects a single 3D model and emits a JSON report summarizing geometry and bounding-box dimensions.

Quick start

1) Using the Blender executable (recommended):

```bash
blender --background --python mini_inspector.py -- input_model.glb output.json
```

2) If you prefer to run `bpy` as a Python module, ensure the `bpy` wheel matches your Python version and Blender build.

Output

The output JSON will follow this structure:

```json
{
  "filename": "model.glb",
  "geometry": { "vertices": 507, "faces": 500 },
  "topology": { "triangles": 0, "quads": 500, "ngons": 0 },
  "dimensions": { "x": 2.73, "y": 1.96, "z": 1.54 }
}
```

Notes & Tips

- If `bpy` is not importable from your system Python, run the script with the Blender binary (method shown above). This avoids `bpy` wheel compatibility issues.
- The script is intentionally minimal and suitable as a building block for Exercise 4's batch processing.
- For batch usage, remember to clear Blender data between files to avoid memory growth:

```python
import bpy
bpy.ops.wm.read_homefile(use_empty=True)
```

Next steps

- Implement additional validation checks (materials list, per-mesh reports) as needed.
- Integrate this script into a robust batch runner that downloads from S3 and uploads JSON reports.
