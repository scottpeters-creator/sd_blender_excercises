# Onboarding for 3D Data Engineering

by @Vlad Stojanovic  (vladeta.stojanovic@stability.ai)

# Introduction

Welcome to the Data Team! As you have probably guessed by now, you will be working on various 3D data related tasks. This usually involves ingressing various 3D data sources, pre or pos-processing ingressed data, or augmenting and/or generating completely new 3D data. We work closely with the 3D research team on various projects, so we are all always available and ready to accomodate their data needs. 

Important note: This guide assumes that your work laptop will be a Macbook Pro running the latest version of macOS. It also assumes you have access to GPU nodes on the HPC cluster. 

# Development Practices & Info

## Code/Dev Environments

We use Visual Studio Code for development, alongside Python as our main programming language. In addition, we use Blender (both as a stand alone DCC tool and as a Python module). 

Important note: For managing Python environments, I recommend using miniconda: [https://www.anaconda.com/docs/getting-started/miniconda/main](https://www.anaconda.com/docs/getting-started/miniconda/main) 

## Essential 3D Software

I recommend you install the following 3D software tools:

- Blender (usually good to have 2 or 3 version installed - I recommend 4.0, 4.5 and the latest 5.0+)
- MeshLab (good for various mesh analysis tasks)
- GIMP for image editing/inspection (in case you don't want to deal with Adobe Creative Cloud)

## Local vs Cluster Development Environments

Once you have everything setup locally, you should be able to use VS code to setup and create projects. You need to decide on a strategy, project by project, if the solution you are developing will first be developed and used locally (e.g., from you work laptop), or if it will be developed/ported to run on the cluster. 

Note about Python environments: It is recommended that you have one general "development” environment that you can use to test things with quickly. However, for any project that requires more specialised libraries/modules, it is recommended that you create a new, separate conda environment. 

# Getting Familiar with Blender

Ok, now we get the main part of this on-boarding document - working with Blender. There are two approaches when working with Blender - stand alone as a DCC, and as a Python module. 

## Blender as a Stand Alone DCC

You can run Blender as a normal app, but if you want to be able to see any kind of debug output (especially form Python scripts that run within Blender), you should start Blender via the command line after navigating to the `/Applications/` folder:

`./{Blender_app_name}.app/Contents/MacOS/Blender`

This will then launch Blender and allow you to see any outputs from Python scripts that you run within Blender. 

## Blender as a Python Module

A more complex, but flexible way, of using Blender is as a Python stand-alone module ([https://docs.blender.org/api/current/info_advanced_blender_as_bpy.html](https://docs.blender.org/api/current/info_advanced_blender_as_bpy.html)) 

You can install Blender via PIP and use it directly in Python. This gives you the advantage of being able to create and orchestrate very complex rendering/data processing pipelines on the HPC cluster. 

Note: Take care when installing BPY that you select the appropriate Python version, since specific BPY version are compatible only with specific versions of Python. See this guide for more details: [https://en.bioerrorlog.work/entry/pip-install-bpy](https://en.bioerrorlog.work/entry/pip-install-bpy) 

# Blender Exercises

The following five exercises are meant to get you up to speed with using Blender, especially for tasks that we use it for in our team. 

***It is recommended that in addition to these exercises, you educate yourself further on using Blender as a DCC in your own time.*** 

The following exercises should be implemented using the Blender BPY Python module, and kept and updated as internal projects on our GitHub. Exercises 1-3 can however first be prototyped in Blender - before being implemented as complete Python scripts that are executed outside of the Blender application environment. 

The timeline for implementing these tasks (and submitting them for feedback), is up to you, but it should generally not take longer than 2 weeks. 

## Exercise 1 - 3D Data Analysis

**Objective:** Create a lightweight Python script using the Blender API (`bpy`) to extract basic metadata from a 3D model and save it as a JSON report.

**Role Context:** As a 3D Data Engineer, you will often need to automate the validation of thousands of 3D assets. This task simulates the foundation of that pipeline.

### 1. Functional Requirements

The script must perform the following actions:

1. **Load:** Import a single `.glb`file into a Blender scene. You can use the existing Objaverse dataset for this: `s3://mod3d-west/objaverse-complete/glbs/` 
2. **Clean:** Ensure the scene is empty before importing (delete default cube/camera/lights).
3. **Analyze:** Extract the following data points from the mesh:
    - **Vertex Count:** Total number of vertices.
    - **Face Count:** Total number of polygons.
    - **Topology Check:** Count how many faces are **Triangles** (3 vertices) vs. **Quads** (4 vertices).
    - **Dimensions:** The X, Y, and Z dimensions of the object's bounding box (world space).
4. **Export:** Save the results to a local JSON file.

### 2. Technical Constraints

- **Language:** Python 3.10+
- **Dependencies:** `bpy` (Blender Python API) and standard libraries (`json`, `sys`, `os`) only.
- **Execution:** The script must run in **background mode** (headless) without opening the Blender GUI.
    - *Example command:* `blender --background --python mini_inspector.py -- input.obj output.json`
    
    OR
    
    It must run as part of stand-alone Python script that uses Blender BPY as a module. 
    

### 3. Expected Output Format

Your script should generate a JSON file with this exact structure:

```jsx
{
  "filename": "monkey.obj",
  "geometry": {
    "vertices": 507,
    "faces": 500
  },
  "topology": {
    "triangles": 0,
    "quads": 500,
    "ngons": 0
  },
  "dimensions": {
    "x": 2.73,
    "y": 1.96,
    "z": 1.54
  }
}
```

### 4. Deliverables

1. `mini_inspector.py`: The source code.
2. `README.md`: Short instructions on how to install `bpy` (or use the Blender executable) and run the script.
3. Run this script for 1k models - upload and save the generated results to the following bucket: `s3://mod3d-west/temp/scott_temp/ex1/` 
4. A report on the implementation approach and results on Notion. 

## Exercise 2 - Thumbnail Rendering with Multiple Modalities

**Objective:** Write a Python script using the Blender API (`bpy`) that imports a 3D object and renders it into four specific visual "modalities" (Textured, Normal, Depth, and Edge), accompanied by a metadata file.

**Role Context:** In machine learning and asset management pipelines, we often need "ground truth" data. A 3D Data Engineer must know how to programmatically switch render engines, manipulate shader trees, or use compositing nodes to extract this data.

### 1. Functional Requirements

The script must perform the following actions:

1. **Setup:** Initialize a scene, remove default objects, and set up a camera pointing at the center `(0,0,0)`.
2. **Import:** Load a single `.glb` or `.obj` file. Again you can use the objaverse dataset for this: `s3://mod3d-west/objaverse-complete/glbs/`
3. **Render Modalities:** Generate four separate $512 \times 512$ PNG images for the object:
    - **Textured:** The object as it looks with its original materials (Albedo/Lit).
    - **Normals:** A color-coded map representing surface orientation (World or Camera space).
    - **Depth:** A grayscale map representing distance from the camera (Black = Close, White = Far).
    - **Edges:** A line-art or contour render (black lines on white background, or similar).
4. **Metadata:** Save a `metadata.json` file containing camera intrinsics and the bounding box of the object.

IMPORTANT NOTE: Each loaded object must be fully in the camera frame. So you will need to figure out a way to adjust the object to fit the current camera frame, OR fit the camera view frame to the object. 

### 2. Technical Constraints

- **Language:** Python 3.10+
- **Dependencies:** `bpy` and standard libraries only.
- **Execution:** Must run headless (background mode).
    - *Command:* `blender --background --python render_task.py -- input.glb output_dir/`
    - OR
        
        It must run as part of stand-alone Python script that uses Blender BPY as a module. 
        
- **Methodology:** You are free to use any valid Blender method to achieve the visual styles (e.g., Compositing Nodes, Material Overrides, or switching to the Workbench Engine). It is recommended you use the Cycles rendering engine, especially for rendering edge maps.

### 3. Expected Output Format

**File Structure:**

```jsx
object_thumbnails_output_dir/
├── render_textured.png
├── render_normal.png
├── render_depth.png
├── render_edge.png
└── metadata.json
```

**JSON Structure:**

```jsx
{
  "source_file": "chair.glb",
  "render_resolution": [512, 512],
  "bounding_box": {
    "center": [0.0, 0.5, 0.0],
    "dimensions": [1.2, 2.0, 1.2]
  },
  "camera": {
    "focal_length": 50.0,
    "position": [0.0, -5.0, 2.0],
    "look_at": [0.0, 0.0, 0.0]
  }
}
```

### 4. Deliverables

1. `render_task.py`: The source code.
2. `README.md`: Brief explanation of the method used to generate the "Depth" and "Edge" passes (e.g., "I used the Workbench engine for edges and a Normalized Z-Pass in the compositor for depth").
3. Run this script for 1k models - upload and save the generated results to the following bucket: `s3://mod3d-west/temp/scott_temp/ex2/` 
4. A report on the implementation approach and results on Notion. 

## Exercise 3 - Procedural Camera Path Generation and Animation

**Objective:** Create a Python script using `bpy` that loads a 3D scene and generates a rendered video file (640x480 resolution, 30 fps) featuring a procedurally generated camera flythrough.

**Role Context:** A 3D Data Engineer must understand 3D coordinate systems, Euler angles vs. Quaternions, and animation methods to generate synthetic training data for computer vision models.

### 1. Functional Requirements

The script must accept a modelfile (representing a scene from the dataset) and perform the following:

1. **Scene Setup:** Check the Evermotion dataset and choose an apporpriate indoor scene: `s3://mod3d-west/evermotion/` 
2. **Camera Modes:** Implement two distinct functions to generate camera movement keyframes over a 150-frame timeline:
    - **Mode A (Random Spline):** Generate 5 random coordinate points within the scene bounds. Create a c**urve (Bezier or other)** from these points and constrain the camera to follow this path smoothly, always looking slightly ahead or at a fixed target.
    - **Mode B (Point-to-Point Interpolation):** Pick 3 distinct random positions. Animate the camera moving linearly between them, but implement a **"Look At" constraint** so the camera always stays focused on the center of the scene `(0,0,0)` regardless of its position.
3. **Rendering:**
    - Set the engine to Cycles.
    - Set resolution to **640 x 480**.
    - Configure the output to compile directly to an **MP4 video file** (using Blender's FFmpeg output settings).

### 2. Technical Constraints

- **Language:** Python 3.10+
- **Dependencies:** `bpy`, `mathutils`, `random` (if using BPY as a stand-alone Python module, you can use the `logger` library).
- **Execution:** Headless (Background mode).
    - *Command:* `blender input_scene.blend --background --python auto_cameraman.py -- --mode spline --output render.mp4`
    - OR
        
        It must run as part of stand-alone Python script that uses Blender BPY as a module. 
        

### 3. Expected Output

- **Video File:** A playable `.mp4` file showing the camera moving through the scene.
- **Console Output (save as text file):** A log confirming which mode was selected and the coordinates of the generated control points.

**Example Console Output:**

Plaintext

```jsx
[INFO] Mode: Random Spline
[INFO] Generated Control Points:
  P1: <Vector (2.1, 5.0, 1.2)>
  P2: <Vector (-3.2, 4.1, 2.0)>
  ...
[INFO] Rendering 150 frames to 'output_spline.mp4'...
```

### 4. Deliverables

1. `proc_camera.py`: The source code.
2. `README.md`: Instructions on how to switch between the two modes (e.g., via a command line argument or a variable at the top of the script).
3. Run this script for 1k models - upload and save the generated results to the following bucket: `s3://mod3d-west/temp/scott_temp/ex3/` 
4. A report on the implementation approach and results on Notion. 

 

## Exercise 4 - Data Processing Integration with AWS and S3

**Objective:** Create a Python CLI tool that recursively scans an S3 bucket for `.glb` files, processes them in a configurable loop, analyzes their geometry using `bpy` (standalone), and uploads JSON metadata back to S3.

**Role Context:** Real pipelines process thousands of assets. A "for loop" that crashes after file #50 because of memory leaks or network timeouts is useless. This task tests your ability to write **resilient, long-running worker scripts**.

### 1. Functional Requirements

The script must accept arguments for `bucket_name`, `prefix`, and `limit`.

1. **Discovery (Recursive S3 Listing):**
    - Use `boto3` to list all objects under a given prefix ending in `.glb`. Once again, you can use the Objaverse dataset: `s3://mod3d-west/objaverse-complete/glbs/`
    - Implement **Pagination**: The script must be able to handle cases where there are more than 1,000 files (the default S3 list limit).
2. **Processing Strategy (The Architectural Decision):**
    - **The Problem:** You are running this on a machine with limited disk space.
    - **The Task:** Implement a loop that handles the lifecycle of the file. You must decide (and document) whether you:
        - **Stream/Serial:** Download → Process → Upload JSON → **Delete Local File** → Next.
        - **Batch/Cache:** Download chunk of 10 → Process → Upload → Clear Cache.
    - *Constraint:* The script **must** clean up local files after processing. It cannot just fill up the `/tmp` directory.
3. **Analysis (BPY):**
    - Import the `.glb`.
    - Extract: **Vertex Count**, **Face Count**, **Material Names**, and **Dimensions (AABB)**.
    - **Memory Management:** Explicitly clear `bpy.data` blocks (meshes, materials, textures) after each file. *Blender does not garbage collect automatically in a script loop.*
4. **Upload (S3 Put):**
    - Upload a JSON report to the same S3 path, replacing `.glb` with `{glb_filename}_report.json`. Make sure you use the UID (i.e., name of the glb) for the json
    - **Error Handling:** If a file is corrupt (BPY fails to import), upload a JSON with `"status": "error"` and the exception message, then **continue** to the next file.

### 2. Technical Constraints

- **Language:** Python 3.10+
- **Libraries:** `bpy` (standalone module), `boto3`, `argparse`.
- **Execution:** Standard Python environment.
- **Resilience:** The script must not crash on a single bad file. Make sure you implement try/catch methods in your functions that handle the bulk processing.

### 3. Expected Output

**Console Output:**

```jsx
[INFO] Scanning 's3://my-test-bucket/assets/'... found 204 files.
[INFO] Processing limit set to 100.

[1/100] processing 'assets/chair.glb'...
   > Downloaded (4.2 MB)
   > Stats: 1,500 verts, 2 materials.
   > Report uploaded to 'assets/chair_report.json'.
   > Cleanup: Local file deleted. BPY memory cleared.

[2/100] processing 'assets/broken_file.glb'...
   > Downloaded (0.1 MB)
   > ERROR: Python: OSERROR: Invalid header.
   > Uploaded error report. Continuing...
...
[DONE] Processed 100 files. 1 Error.
```

**JSON Report (Success):**

```jsx
{
  "file": "assets/chair.glb",
  "status": "success",
  "stats": {
    "vertices": 1500,
    "faces": 1450,
    "dimensions": {"x": 1.0, "y": 1.0, "z": 2.0}
  }
}
```

**JSON Report (Failure):**

```jsx
{
  "file": "assets/broken_file.glb",
  "status": "error",
  "error_message": "bpy_struct: item.attr = val: AttributeError: 'NoneType' object has no attribute..."
}
```

### 4. Deliverables

1. **`batch_validator.py`**: The source code.
2. **`README.md`**:
    - **Architecture Choice:** Explain why you chose Serial vs. Batch processing.
    - **Memory Strategy:** Explain how you ensured Blender doesn't run out of RAM (e.g., "I used `bpy.ops.wm.read_homefile(use_empty=True)` to reset the state").
3. Run this script for 2k models - upload and save the generated results to the following bucket: `s3://mod3d-west/temp/scott_temp/ex4/` 
4. A report on the implementation approach and results on Notion. 

## Exercise 5 - Implementing a Simple Data Processing Pipeline on the HPC Cluster

**Objective:** Create a Python-based batch processing pipeline that runs on a High-Performance Computing (HPC) cluster. You must process 10 human scan files using the standalone `bpy` module, filtering out animated subjects and rendering the rest on an NVIDIA H100 GPU.

**Role Context:** When tasked with generating a lot of new training data, we don't render on our Macbooks usually. We submit jobs to GPU-enabled nodes on the HPC. You must demonstrate that you understand how to wrap Python logic inside a scheduler (SLURM) and handle data validation (skipping bad assets) programmatically.

### 1. The Scenario

You are given a folder containing  `.blend` files of humans, you need to render 10 of them. The humans subject are contained here: `s3://mod3d-west/renderpeople-motion4d/processed_human_test/` 

- **The Problem:** Some of these scans are "dirty"—they contain animation data where the subject drifts or walks away. We only want **static** subjects.
- **The Goal:** Generate a clean dataset of 640x480 orbiting shots (RGB + Depth) for all static subjects.

### 2. Functional Requirements

### A. Python Driver (`process_batch.py`)

Write a script that runs inside Blender (headless) and performs the following for **each** of the 20 files:

1. **Load:** Open the `.blend` file.
2. **Filter (Crucial):** Check if the subject is moving.
    - I.e. implement a class function called `_is_subject_animated` (or access `object.animation_data` directly).
    - **Logic:** If the subject moves/walks, **abort** processing for this file and log: `"Skipping {filename}: Subject is animated."`
3. **Normalize:**
    - If the subject is static, force their location to the World Origin `(0, 0, 0)` to ensure the camera orbit works correctly.
4. **Configure Renderer:**
    - Instantiate `RenderFlythroughHuman`.
    - **Resolution:** 640 x 480.
    - **Modalities:** Enable output for **Frames** (RGB) and **Depth Maps** (32-bit EXR).
    - **Video:** Disable video generation (`create_animation=False`)—we only want the raw frame sequence.
5. **Implement Orbit Animation:**
    - Implement an animation function that orbits the camera around the subject. Render out 360 frames for this animation.

### B. SLURM Submission Script

Write a Bash script to submit this job to the cluster. It must:

1. **Resources:** Request 1 Node with **1 NVIDIA H100 GPU**.
2. **Walltime:** Set a limit of 12 hours.
3. **Environment:** Load the Blender module (e.g., `module load blender/4.0`).
4. **Command:** Execute the Python driver in background mode.

---

### 4. Expected Deliverables

1. `process_batch.py`

- Iterates through the file list.
- Implements the "Skip if moving" logic.
- Centers the subject.
- Performs the orbit animation
- Calls the renderer for each frame on animation
- Saves each frame in two different modalities: Standard and Dept

2. `submit_jobs.sh`

- Standard SBATCH directives.
- Correct GPU partition requests.
1. Run this script for 10 human models - upload and save the generated results to the following bucket: `s3://mod3d-west/temp/scott_temp/ex5/` 
2. A report on the implementation approach and results on Notion. 

---

### 5. Technical Constraints & Hints

- **Depth Maps:** Rendered depthmaps 32-bit EXR output.

### Example Directory Structure (Mental Model)

```jsx
/project/
├── process_human_batch.py     
├── submit_jobs.sh        
└── /data/
    ├── human_001.blend   # (Static - Keep)
    ├── human_002.blend   # (Walking - Skip)
    └── ...
```

## Additional Notes for Tasks

- Make sure you explicitly enable GPU usage when running the scripts on the H100 GPU nodes. You can use the following code as reference:

```jsx
@staticmethod
    def _enable_gpu() -> None:
        logger.info("Setting rendering device...")

        # Handle macOS fallback
        if sys.platform == "darwin":
            logger.info("No GPU support on macOS; using CPU.")
            bpy.context.scene.cycles.device = "CPU"
            # Note: CPU threads already configured in _configure_threading_environment
            return

        # Log CUDA_VISIBLE_DEVICES env var
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        logger.info(f"CUDA_VISIBLE_DEVICES = '{cuda_visible}'")

        # Ensure Cycles addon is available
        if "cycles" not in bpy.context.preferences.addons:
            logger.error("Cycles addon is not enabled.")
            bpy.context.scene.cycles.device = "CPU"
            return

        bpy.context.scene.cycles.device = "GPU"
        render_prefs = bpy.context.preferences.addons["cycles"].preferences

        # Try OPTIX first (best for H100), then fallback to CUDA
        for gpu_type in ["OPTIX", "CUDA"]:
            render_prefs.compute_device_type = gpu_type
            render_prefs.refresh_devices()

            available = [d for d in render_prefs.devices if d.type == gpu_type]
            if not available:
                logger.info(f"No {gpu_type} devices found.")
                continue

            logger.info(f"{gpu_type} devices detected:")
            gpu_count = 0
            for d in available:
                d.use = True
                gpu_count += 1
                logger.info(f"  Enabled: {d.name} (Type: {d.type})")

            bpy.context.scene.cycles.device = "GPU"
            logger.info(f"Using compute device type: {gpu_type} with {gpu_count} GPU(s)")
            break
        else:
            logger.warning("No supported GPU devices found. Falling back to CPU.")
            bpy.context.scene.cycles.device = "CPU"
            return

        # Final device status log
        logger.debug("Final device configuration:")
        for i, d in enumerate(render_prefs.devices):
            logger.debug(f"  [{i}] {d.name} | Type: {d.type} | Enabled: {d.use}")
```

- Below is an example script for setting up a GPU node on the [gpu.hpc.stability.ai](http://gpu.hpc.stability.ai) cluster. This is sbatch - usually you would spin this out and then SSH into into.

```jsx
#!/bin/bash

# Simple SLURM Node Setup and SSH Configuration
# Usage: ./gpu_node_setup.sh

set -e

echo "=== SLURM Node Setup ==="

# Submit job using sbatch instead of srun for better control (use either data or sd3 accounts)
JOB_ID=$(sbatch --parsable \
    --account=data \
    --partition=shared \
    --nodes=1 \
    --mem=250G \
    --gpus=1 \
    --job-name=svcrf-job \
    --time=12:00:00 \
    --wrap="sleep 43200")  # Sleep for 12 hours

echo "✅ Job submitted successfully!"
echo "Job ID: $JOB_ID"

# Wait for job to start
echo "Waiting for job to start..."
while true; do
    JOB_STATE=$(squeue -j "$JOB_ID" --noheader --format="%T" 2>/dev/null || echo "UNKNOWN")
    
    if [ "$JOB_STATE" = "RUNNING" ]; then
        echo "✅ Job is now running!"
        break
    elif [ "$JOB_STATE" = "PENDING" ]; then
        echo "Job is pending... (waiting for resources)"
        sleep 5
    else
        echo "❌ Job state: $JOB_STATE"
        echo "Job may have failed. Check with: squeue -j $JOB_ID"
        exit 1
    fi
done

# Get node information
NODE=$(squeue -j "$JOB_ID" --noheader --format="%N")
echo "Job allocated to node: $NODE"

echo
echo "=== Setting up SSH ==="
echo "Running: stablessh -t -j $JOB_ID"

if stablessh -t -j "$JOB_ID"; then
    echo
    echo "✅ Setup complete!"
    echo
    echo "=== Summary ==="
    echo "Job ID: $JOB_ID"
    echo "Node: $NODE" 
    echo "SSH config has been set up by stablessh"
    echo
    echo "To cancel job: scancel $JOB_ID"
    echo "To check status: squeue -j $JOB_ID"
else
    echo "❌ stablessh failed"
    echo "You may need to cancel the job: scancel $JOB_ID"
    exit 1
fi
```

# Additional Resources

[Data Engineering Team V2](https://www.notion.so/Data-Engineering-Team-V2-2be61cdcd196803b93f8e06daef95195?pvs=21)