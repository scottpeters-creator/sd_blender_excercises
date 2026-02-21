"""Volume/SDF steps: mesh repair, AABB box, geometry SDF, boolean difference, scatter, order.

SOP-style granular steps for volumetric spatial operations. Each step
produces visible geometry in the Blender scene, inspectable via the
.blend snapshot saved by BlenderStep.

Uses Blender 5.0 Geometry Nodes (Mesh to SDF Grid, SDF Grid Boolean,
Grid to Mesh, Points to SDF Grid).
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any, Dict, List, Tuple

from lib.pipeline import CompletedState
from lib.pipeline_steps.blender_step import BlenderStep

logger = logging.getLogger("pipeline.steps.volume")


class ExtractSingleSidedStep(BlenderStep):
    """Detect non-manifold faces in the scene and extract them.

    Iterates all scene mesh objects, finds faces with boundary edges
    (edges shared by only one face), and copies those faces into a
    separate mesh object for isosurface repair.

    The output mesh in the ``_proc_cam`` collection shows exactly which
    faces were flagged as non-manifold — open it in the snapshot to verify.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="extract_single_sided",
            requires=["scene_loaded"],
            provides=["single_sided_mesh"],
            **kwargs,
        )

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        import bpy
        import bmesh
        from lib.bpy.geometry_nodes import _link_to_temp_collection, _PREFIX

        scene = bpy.context.scene
        depsgraph = bpy.context.evaluated_depsgraph_get()

        collected_verts = []
        collected_faces = []
        vert_offset = 0

        for obj in scene.objects:
            if obj.type != "MESH" or obj.name.startswith(_PREFIX):
                continue

            eval_obj = obj.evaluated_get(depsgraph)
            mesh = eval_obj.to_mesh()
            bm = bmesh.new()
            bm.from_mesh(mesh)
            bm.transform(obj.matrix_world)

            boundary_face_indices = set()
            for edge in bm.edges:
                if len(edge.link_faces) == 1:
                    boundary_face_indices.add(edge.link_faces[0].index)

            if boundary_face_indices:
                used_vert_map = {}
                for face in bm.faces:
                    if face.index in boundary_face_indices:
                        face_vert_indices = []
                        for v in face.verts:
                            if v.index not in used_vert_map:
                                used_vert_map[v.index] = len(collected_verts)
                                collected_verts.append(tuple(v.co))
                            face_vert_indices.append(used_vert_map[v.index])
                        collected_faces.append(face_vert_indices)

            bm.free()
            eval_obj.to_mesh_clear()

        if not collected_faces:
            logger.info("No single-sided faces found — all geometry is manifold")
            mesh_data = bpy.data.meshes.new(f"{_PREFIX}single_sided")
            ss_obj = bpy.data.objects.new(f"{_PREFIX}single_sided", mesh_data)
            _link_to_temp_collection(ss_obj)
            context["single_sided_mesh"] = ss_obj.name
            return CompletedState(
                success=True,
                timestamp=CompletedState.now_iso(),
                duration_s=0.0,
                provides=["single_sided_mesh"],
                outputs={"num_faces": 0, "num_verts": 0},
            )

        mesh_data = bpy.data.meshes.new(f"{_PREFIX}single_sided")
        mesh_data.from_pydata(collected_verts, [], collected_faces)
        mesh_data.update()

        ss_obj = bpy.data.objects.new(f"{_PREFIX}single_sided", mesh_data)
        _link_to_temp_collection(ss_obj)

        logger.info(
            "Extracted %d single-sided faces (%d verts) from scene",
            len(collected_faces), len(collected_verts),
        )

        context["single_sided_mesh"] = ss_obj.name
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=["single_sided_mesh"],
            outputs={"num_faces": len(collected_faces), "num_verts": len(collected_verts)},
        )


class SingleSidedToIsoMeshStep(BlenderStep):
    """Convert single-sided faces to a watertight proxy mesh via isosurface.

    Pipeline:
    1. Per-face random scatter (fast, O(n))
    2. KDTree relaxation (Lloyd's algorithm, K=8 neighbors, 4 iterations)
    3. Points to SDF Grid + Grid to Mesh via Geometry Nodes

    The relaxation step evens out point spacing — eliminating the gaps
    and clumps from random scatter — without the cost of Poisson Disk
    on large meshes. All parameters are derived from the scene AABB.
    """

    def __init__(self, relax_iterations: int = 4, **kwargs: Any) -> None:
        super().__init__(
            name="single_sided_to_iso_mesh",
            requires=["single_sided_mesh"],
            provides=["iso_proxy_mesh"],
            **kwargs,
        )
        self._relax_iterations = relax_iterations

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        import bpy
        from lib.bpy.geometry_nodes import (
            scene_aabb, build_points_to_iso_mesh_tree,
            apply_gn_modifier, evaluated_mesh_copy,
            _link_to_temp_collection, _PREFIX,
        )

        ss_name = context["single_sided_mesh"]
        ss_obj = bpy.data.objects[ss_name]
        mesh_data = ss_obj.data

        if len(mesh_data.polygons) == 0:
            logger.info("No single-sided faces to repair — creating empty proxy")
            proxy_mesh = bpy.data.meshes.new(f"{_PREFIX}iso_proxy")
            proxy_obj = bpy.data.objects.new(f"{_PREFIX}iso_proxy", proxy_mesh)
            _link_to_temp_collection(proxy_obj)
            context["iso_proxy_mesh"] = proxy_obj.name
            return CompletedState(
                success=True,
                timestamp=CompletedState.now_iso(),
                duration_s=0.0,
                provides=["iso_proxy_mesh"],
                outputs={"num_points": 0, "num_faces": 0},
            )

        all_scene_meshes = [
            o for o in bpy.context.scene.objects
            if o.type == "MESH" and not o.name.startswith(_PREFIX)
        ]
        bbox_min, bbox_max = scene_aabb(all_scene_meshes)
        diag = (bbox_max - bbox_min).length
        voxel_size = diag / 400.0
        radius = voxel_size * 1.5
        overlap_factor = 0.045

        scatter_points = _per_face_scatter(mesh_data, radius, overlap_factor)
        logger.info(
            "Scattered %d points on %d faces (radius=%.4f, voxel=%.4f)",
            len(scatter_points), len(mesh_data.polygons), radius, voxel_size,
        )

        if not scatter_points:
            proxy_mesh = bpy.data.meshes.new(f"{_PREFIX}iso_proxy")
            proxy_obj = bpy.data.objects.new(f"{_PREFIX}iso_proxy", proxy_mesh)
            _link_to_temp_collection(proxy_obj)
            context["iso_proxy_mesh"] = proxy_obj.name
            return CompletedState(
                success=True,
                timestamp=CompletedState.now_iso(),
                duration_s=0.0,
                provides=["iso_proxy_mesh"],
                outputs={"num_points": 0},
            )

        relaxed_points = _relax_points_kdtree(
            scatter_points,
            iterations=self._relax_iterations,
            k_neighbors=8,
            damping=0.3,
        )
        logger.info("Relaxed %d points (%d iterations)", len(relaxed_points), self._relax_iterations)

        pts_mesh = bpy.data.meshes.new(f"{_PREFIX}scatter_pts")
        pts_mesh.from_pydata(relaxed_points, [], [])
        pts_mesh.update()
        pts_obj = bpy.data.objects.new(f"{_PREFIX}scatter_pts", pts_mesh)
        _link_to_temp_collection(pts_obj)

        ng = build_points_to_iso_mesh_tree(radius=radius, voxel_size=voxel_size)
        apply_gn_modifier(pts_obj, ng, name="IsoMesh")

        proxy_obj = evaluated_mesh_copy(pts_obj, name="iso_proxy")

        proxy_verts = len(proxy_obj.data.vertices)
        proxy_faces = len(proxy_obj.data.polygons)
        logger.info("IsoMesh proxy: %d verts, %d faces", proxy_verts, proxy_faces)

        context["iso_proxy_mesh"] = proxy_obj.name
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=["iso_proxy_mesh"],
            outputs={
                "num_scatter_points": len(relaxed_points),
                "proxy_verts": proxy_verts,
                "proxy_faces": proxy_faces,
                "relax_iterations": self._relax_iterations,
            },
        )


class DeleteGeometryStep(BlenderStep):
    """Delete faces from a target mesh that match a reference mesh.

    Generic geometry deletion step. Finds faces in the target mesh whose
    vertices match positions in the reference mesh (within tolerance),
    and deletes them. Alternatively, if ``mode="boundary"``, deletes all
    faces with at least one boundary (non-manifold) edge.

    The target mesh is modified in-place (it's a pipeline temp copy,
    not the original scene geometry). The result is stored under a new
    context key.

    Constructor args:
        target_key: context key for the mesh to modify (default: "merged_mesh")
        output_key: context key for the result (default: "merged_mesh_clean")
        mode: "boundary" (delete non-manifold faces) or "reference"
              (delete faces matching a reference mesh)
        reference_key: context key for the reference mesh (used in "reference" mode)
    """

    def __init__(
        self,
        target_key: str = "merged_mesh",
        output_key: str = "merged_mesh_clean",
        mode: str = "boundary",
        reference_key: str = "",
        **kwargs: Any,
    ) -> None:
        requires = [target_key]
        if mode == "reference" and reference_key:
            requires.append(reference_key)
        super().__init__(
            name="delete_geometry",
            requires=requires,
            provides=[output_key],
            **kwargs,
        )
        self._target_key = target_key
        self._output_key = output_key
        self._mode = mode
        self._reference_key = reference_key

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        import bpy
        import bmesh
        from lib.bpy.geometry_nodes import _link_to_temp_collection, _PREFIX

        target_name = context[self._target_key]
        target_obj = bpy.data.objects[target_name]

        bm = bmesh.new()
        bm.from_mesh(target_obj.data)

        if self._mode == "boundary":
            boundary_face_indices = set()
            for edge in bm.edges:
                if len(edge.link_faces) == 1:
                    boundary_face_indices.add(edge.link_faces[0].index)
            faces_to_delete = [f for f in bm.faces if f.index in boundary_face_indices]
        else:
            faces_to_delete = []

        deleted_count = len(faces_to_delete)
        if faces_to_delete:
            bmesh.ops.delete(bm, geom=faces_to_delete, context="FACES")

        result_mesh = bpy.data.meshes.new(f"{_PREFIX}deleted")
        bm.to_mesh(result_mesh)
        bm.free()

        result_obj = bpy.data.objects.new(f"{_PREFIX}deleted", result_mesh)
        _link_to_temp_collection(result_obj)

        logger.info(
            "DeleteGeometry: deleted %d faces from '%s' -> '%s' (%d verts, %d faces)",
            deleted_count, target_name, result_obj.name,
            len(result_mesh.vertices), len(result_mesh.polygons),
        )

        context[self._output_key] = result_obj.name
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=[self._output_key],
            outputs={
                "deleted_faces": deleted_count,
                "result_verts": len(result_mesh.vertices),
                "result_faces": len(result_mesh.polygons),
            },
        )


class AppendGeometryStep(BlenderStep):
    """Append one mesh into another.

    Generic geometry merge step. Copies all vertices and faces from the
    source mesh into the target mesh. Both are pipeline temp objects —
    original scene geometry is never modified.

    Constructor args:
        target_key: context key for the mesh to append into
        source_key: context key for the mesh to append from
        output_key: context key for the result
    """

    def __init__(
        self,
        target_key: str = "merged_mesh_clean",
        source_key: str = "iso_proxy_mesh",
        output_key: str = "merged_mesh_final",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="append_geometry",
            requires=[target_key, source_key],
            provides=[output_key],
            **kwargs,
        )
        self._target_key = target_key
        self._source_key = source_key
        self._output_key = output_key

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        import bpy
        import bmesh
        from lib.bpy.geometry_nodes import _link_to_temp_collection, _PREFIX

        target_name = context[self._target_key]
        source_name = context[self._source_key]

        target_obj = bpy.data.objects[target_name]
        source_obj = bpy.data.objects.get(source_name)

        bm = bmesh.new()
        bm.from_mesh(target_obj.data)

        if source_obj and source_obj.data and len(source_obj.data.vertices) > 0:
            src_bm = bmesh.new()
            src_bm.from_mesh(source_obj.data)

            offset = len(bm.verts)
            for v in src_bm.verts:
                bm.verts.new(v.co)
            bm.verts.ensure_lookup_table()
            for f in src_bm.faces:
                try:
                    bm.faces.new([bm.verts[offset + v.index] for v in f.verts])
                except ValueError:
                    pass
            appended_verts = len(src_bm.verts)
            src_bm.free()
        else:
            appended_verts = 0

        result_mesh = bpy.data.meshes.new(f"{_PREFIX}appended")
        bm.to_mesh(result_mesh)
        bm.free()

        result_obj = bpy.data.objects.new(f"{_PREFIX}appended", result_mesh)
        _link_to_temp_collection(result_obj)

        logger.info(
            "AppendGeometry: appended %d verts from '%s' into '%s' -> '%s' (%d verts, %d faces)",
            appended_verts, source_name, target_name, result_obj.name,
            len(result_mesh.vertices), len(result_mesh.polygons),
        )

        context[self._output_key] = result_obj.name
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=[self._output_key],
            outputs={
                "appended_verts": appended_verts,
                "result_verts": len(result_mesh.vertices),
                "result_faces": len(result_mesh.polygons),
            },
        )


class BBoxVolumeStep(BlenderStep):
    """Create an AABB box mesh and compute its SDF volume.

    Creates a box mesh at the scene AABB bounds (with slight padding),
    then applies a Mesh to SDF Grid GN modifier. The resulting SDF
    visualization (Grid to Mesh at threshold=0) is visible in the snapshot.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="bbox_volume",
            requires=["merged_mesh_final"],
            provides=["bbox_sdf"],
            **kwargs,
        )

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        import bpy
        from lib.bpy.geometry_nodes import (
            scene_aabb, create_aabb_box, build_mesh_to_sdf_tree, apply_gn_modifier,
        )

        merged_name = context["merged_mesh_final"]
        merged_obj = bpy.data.objects[merged_name]

        all_meshes = [o for o in bpy.context.scene.objects if o.type == "MESH" and not o.name.startswith("_proc_cam_")]
        bbox_min, bbox_max = scene_aabb(all_meshes)
        diag = (bbox_max - bbox_min).length
        voxel_size = diag / 200.0

        inset = voxel_size * 1.5
        from mathutils import Vector
        inset_min = bbox_min + Vector((inset, inset, inset))
        inset_max = bbox_max - Vector((inset, inset, inset))
        box_obj = create_aabb_box(inset_min, inset_max, padding=0.0)

        ng = build_mesh_to_sdf_tree(voxel_size)
        apply_gn_modifier(box_obj, ng, name="BBoxSDF")

        context["bbox_sdf"] = box_obj.name
        context["_bbox_min"] = tuple(bbox_min)
        context["_bbox_max"] = tuple(bbox_max)
        context["_voxel_size"] = voxel_size
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=["bbox_sdf"],
            outputs={"box_obj": box_obj.name, "voxel_size": round(voxel_size, 4)},
        )


class MeshToSDFMeshStep(BlenderStep):
    """Convert a mesh to a unified SDF mesh via Mesh to SDF Grid + Grid to Mesh.

    Generic step: takes any mesh (potentially with multiple disconnected
    shells), converts to an SDF volume, then extracts the iso-surface
    back as a single unified mesh. This merges all shells into one
    continuous SDF field.

    The voxel size is derived from the scene AABB diagonal.

    Constructor args:
        input_key: context key for the input mesh object name
        output_key: context key for the output unified mesh object name
    """

    def __init__(
        self,
        input_key: str = "iso_shell",
        output_key: str = "unified_shell",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="mesh_to_sdf_mesh",
            requires=[input_key],
            provides=[output_key],
            **kwargs,
        )
        self._input_key = input_key
        self._output_key = output_key

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        import bpy
        from lib.bpy.geometry_nodes import (
            scene_aabb, build_mesh_to_sdf_tree, apply_gn_modifier,
            evaluated_mesh_copy, _PREFIX,
        )

        mesh_name = context[self._input_key]
        mesh_obj = bpy.data.objects[mesh_name]

        all_scene = [
            o for o in bpy.context.scene.objects
            if o.type == "MESH" and not o.name.startswith(_PREFIX)
        ]
        bbox_min, bbox_max = scene_aabb(all_scene)
        diag = (bbox_max - bbox_min).length
        voxel_size = diag / 200.0

        ng = build_mesh_to_sdf_tree(voxel_size)
        apply_gn_modifier(mesh_obj, ng, name="UnifySDF")

        unified_obj = evaluated_mesh_copy(mesh_obj, name="unified_shell")

        num_verts = len(unified_obj.data.vertices)
        num_faces = len(unified_obj.data.polygons)
        logger.info(
            "MeshToSDFMesh: '%s' -> '%s' (%d verts, %d faces)",
            mesh_name, unified_obj.name, num_verts, num_faces,
        )

        context[self._output_key] = unified_obj.name
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=[self._output_key],
            outputs={"num_verts": num_verts, "num_faces": num_faces},
        )


class GeomVolumeStep(BlenderStep):
    """Compute the SDF volume of the merged scene geometry.

    Applies a Mesh to SDF Grid GN modifier to the merged mesh object.
    The SDF visualization is visible in the snapshot.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="geom_volume",
            requires=["merged_mesh_final"],
            provides=["geom_sdf"],
            **kwargs,
        )

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        import bpy
        from lib.bpy.geometry_nodes import build_mesh_to_sdf_tree, apply_gn_modifier

        merged_name = context["merged_mesh_final"]
        merged_obj = bpy.data.objects[merged_name]
        voxel_size = context.get("_voxel_size", 10.0)

        ng = build_mesh_to_sdf_tree(voxel_size)
        apply_gn_modifier(merged_obj, ng, name="GeomSDF")

        context["geom_sdf"] = merged_obj.name
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=["geom_sdf"],
            outputs={"geom_obj": merged_obj.name},
        )


class VolumeDiffStep(BlenderStep):
    """Compute the SDF boolean difference: AABB box minus scene geometry.

    The result is the negative space — the navigable air volume.
    Uses SDF Grid Boolean (DIFFERENCE) via Geometry Nodes.
    The boundary mesh (Grid to Mesh at threshold=0) is visible in the snapshot.

    Also computes estimated_max_sdf = min(AABB dimensions) / 2 for
    threshold normalization in the scatter step.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="volume_diff",
            requires=["bbox_sdf", "geom_sdf"],
            provides=["neg_space_volume", "max_sdf_value"],
            **kwargs,
        )

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        import bpy
        from mathutils import Vector
        from lib.bpy.geometry_nodes import (
            build_sdf_boolean_tree, evaluated_mesh_copy,
        )

        geom_name = context["geom_sdf"]
        bbox_name = context["bbox_sdf"]
        geom_obj = bpy.data.objects[geom_name]
        bbox_obj = bpy.data.objects[bbox_name]

        for mod in list(geom_obj.modifiers):
            geom_obj.modifiers.remove(mod)

        voxel_size = context.get("_voxel_size", 10.0)
        ng = build_sdf_boolean_tree(voxel_size, bbox_obj)
        mod = geom_obj.modifiers.new(name="VolumeDiff", type="NODES")
        mod.node_group = ng

        neg_space_obj = evaluated_mesh_copy(geom_obj, name="neg_space")

        bbox_min = Vector(context.get("_bbox_min", (0, 0, 0)))
        bbox_max = Vector(context.get("_bbox_max", (1, 1, 1)))
        dims = bbox_max - bbox_min
        estimated_max_sdf = min(dims.x, dims.y, dims.z) / 2.0

        vert_count = len(neg_space_obj.data.vertices)
        logger.info(
            "Negative space: %d verts, estimated_max_sdf=%.2f",
            vert_count, estimated_max_sdf,
        )

        context["neg_space_volume"] = neg_space_obj.name
        context["max_sdf_value"] = estimated_max_sdf
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=["neg_space_volume", "max_sdf_value"],
            outputs={
                "neg_space_obj": neg_space_obj.name,
                "neg_space_verts": vert_count,
                "max_sdf": round(estimated_max_sdf, 4),
            },
        )


class ScatterPointsStep(BlenderStep):
    """Scatter points inside the negative space boundary mesh.

    Uses BVH ray-parity sampling: generates grid points inside the neg
    space mesh's bounding box, tests each with a ray-cast inside/outside
    check, and keeps only interior points. Scale-agnostic — spacing is
    derived from the estimated max SDF value.

    Reads ``sdf_threshold`` from context (default 0.5) to control how
    many grid cells are generated (lower threshold = denser grid = more
    candidates).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="scatter_points",
            requires=["neg_space_volume", "max_sdf_value"],
            provides=["camera_candidates"],
            **kwargs,
        )

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        import bpy
        from mathutils import Vector
        from mathutils.bvhtree import BVHTree
        import bmesh

        neg_space_name = context["neg_space_volume"]
        neg_space_obj = bpy.data.objects[neg_space_name]
        max_sdf = context["max_sdf_value"]
        threshold_frac = context.get("sdf_threshold", 0.5)

        spacing = max(max_sdf * (0.05 + 0.2 * threshold_frac), 1.0)

        mesh_data = neg_space_obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh_data)
        bvh = BVHTree.FromBMesh(bm)
        bm.free()

        coords = [Vector(v.co) for v in mesh_data.vertices]
        if not coords:
            return CompletedState(
                success=False,
                timestamp=CompletedState.now_iso(),
                duration_s=0.0,
                error={"message": "negative space mesh has no vertices"},
            )

        xs = [v.x for v in coords]
        ys = [v.y for v in coords]
        zs = [v.z for v in coords]
        bb_min = Vector((min(xs), min(ys), min(zs)))
        bb_max = Vector((max(xs), max(ys), max(zs)))

        logger.info(
            "Scatter grid: spacing=%.2f, bbox=(%.1f,%.1f,%.1f)-(%.1f,%.1f,%.1f)",
            spacing, bb_min.x, bb_min.y, bb_min.z, bb_max.x, bb_max.y, bb_max.z,
        )

        candidates = []
        x = bb_min.x
        while x <= bb_max.x:
            y = bb_min.y
            while y <= bb_max.y:
                z = bb_min.z
                while z <= bb_max.z:
                    pt = Vector((x, y, z))
                    if _is_inside_mesh(pt, bvh):
                        candidates.append((x, y, z))
                    z += spacing
                y += spacing
            x += spacing

        if len(candidates) < 2:
            return CompletedState(
                success=False,
                timestamp=CompletedState.now_iso(),
                duration_s=0.0,
                error={
                    "message": f"Only {len(candidates)} points inside neg space "
                    f"(need >= 2). Try lowering --threshold.",
                },
            )

        from lib.bpy.geometry_nodes import _link_to_temp_collection, _PREFIX
        pts_mesh = bpy.data.meshes.new(f"{_PREFIX}candidates")
        pts_mesh.from_pydata(candidates, [], [])
        pts_mesh.update()
        pts_obj = bpy.data.objects.new(f"{_PREFIX}candidates", pts_mesh)
        _link_to_temp_collection(pts_obj)

        logger.info("Scattered %d camera candidates (spacing=%.2f)", len(candidates), spacing)
        context["camera_candidates"] = candidates
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=["camera_candidates"],
            outputs={"num_candidates": len(candidates), "spacing": round(spacing, 2)},
        )


class CullPointsByMeshStep(BlenderStep):
    """Remove points that are inside (or outside) a reference mesh.

    .. note:: ``get_artifact`` returns the ``_proc_cam_culled`` object
       created during execution for outliner visibility.

    Generic culling step with two methods:

    - ``"raycast"``: BVH ray-parity (odd hit count = inside). Works for
      single enclosing shells. Unreliable for multiple disconnected shells.

    - ``"normal"``: Nearest-surface normal dot product. Finds the closest
      point on the mesh surface, then checks if the vector from surface
      to candidate aligns with the surface normal. Positive dot = outside
      (same side as normal). Negative dot = inside (opposite side).
      Works correctly for multiple disconnected manifold shells.

    - ``"sdf"``: Geometry Nodes native. Converts the reference mesh to an
      SDF Grid, samples each point's SDF value, deletes points where
      SDF < 0 (inside). Fastest method — runs entirely in the GN engine.
      Works correctly for any manifold mesh (single or multi-shell).

    Constructor args:
        points_key: context key for the point list
        mesh_key: context key for the reference mesh object name
        output_key: context key for the culled point list
        keep: "inside" or "outside"
        method: "raycast", "normal", or "sdf"
    """

    def __init__(
        self,
        points_key: str = "camera_candidates",
        mesh_key: str = "geom_sdf",
        output_key: str = "camera_candidates_culled",
        keep: str = "outside",
        method: str = "normal",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="cull_points_by_mesh",
            requires=[points_key, mesh_key],
            provides=[output_key],
            **kwargs,
        )
        self._points_key = points_key
        self._mesh_key = mesh_key
        self._output_key = output_key
        self._keep = keep
        self._method = method

    def get_artifact(self, context):
        import bpy
        from lib.bpy.geometry_nodes import _PREFIX
        return bpy.data.objects.get(f"{_PREFIX}culled")

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        import bpy
        from lib.bpy.geometry_nodes import _link_to_temp_collection, _PREFIX

        points = context[self._points_key]
        mesh_name = context[self._mesh_key]
        mesh_obj = bpy.data.objects[mesh_name]

        keep_inside = self._keep == "inside"

        if self._method == "sdf":
            from lib.bpy.geometry_nodes import (
                build_sdf_cull_tree, scene_aabb, apply_gn_modifier,
                read_points_from_evaluated,
            )

            all_scene = [
                o for o in bpy.context.scene.objects
                if o.type == "MESH" and not o.name.startswith(_PREFIX)
            ]
            bbox_min, bbox_max = scene_aabb(all_scene)
            diag = (bbox_max - bbox_min).length
            voxel_size = diag / 200.0

            pts_mesh_in = bpy.data.meshes.new(f"{_PREFIX}cull_input")
            pts_mesh_in.from_pydata(points, [], [])
            pts_mesh_in.update()
            pts_obj_in = bpy.data.objects.new(f"{_PREFIX}cull_input", pts_mesh_in)
            _link_to_temp_collection(pts_obj_in)

            if keep_inside:
                logger.warning("SDF method with keep=inside not yet supported, falling back to keep=outside")

            ng = build_sdf_cull_tree(mesh_obj, voxel_size)
            apply_gn_modifier(pts_obj_in, ng, name="SDFCull")

            culled = read_points_from_evaluated(pts_obj_in)

        else:
            import bmesh
            from mathutils import Vector
            from mathutils.bvhtree import BVHTree

            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_obj = mesh_obj.evaluated_get(depsgraph)

            bm = bmesh.new()
            bm.from_mesh(eval_obj.data)
            bvh = BVHTree.FromBMesh(bm)
            bm.free()

            culled = []
            if self._method == "normal":
                for p in points:
                    pt = Vector(p)
                    location, normal, index, dist = bvh.find_nearest(pt)
                    if location is None:
                        culled.append(p)
                        continue
                    to_point = (pt - location).normalized()
                    dot = to_point.dot(normal)
                    is_outside = dot > 0
                    if (keep_inside and not is_outside) or (not keep_inside and is_outside):
                        culled.append(p)
            else:
                for p in points:
                    inside = _is_inside_mesh(Vector(p), bvh)
                    if (keep_inside and inside) or (not keep_inside and not inside):
                        culled.append(p)

        pts_mesh = bpy.data.meshes.new(f"{_PREFIX}culled")
        pts_mesh.from_pydata(culled, [], [])
        pts_mesh.update()
        pts_obj = bpy.data.objects.new(f"{_PREFIX}culled", pts_mesh)
        _link_to_temp_collection(pts_obj)

        logger.info(
            "CullPointsByMesh: %d -> %d points (keep=%s, method=%s, mesh='%s')",
            len(points), len(culled), self._keep, self._method, mesh_name,
        )

        context[self._output_key] = culled
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=[self._output_key],
            outputs={
                "input_count": len(points),
                "output_count": len(culled),
                "keep": self._keep,
            },
        )


class IsoMeshFromMeshStep(BlenderStep):
    """Build an iso-surface mesh from a mesh using GN-native scatter.

    Single GN evaluation: Distribute Points on Faces (random mode) ->
    Points to SDF Grid -> Grid to Mesh. Runs entirely in the GN engine.

    The density and radius are derived from the scene AABB diagonal.

    Constructor args:
        input_key: context key for the input mesh object
        output_key: context key for the output iso mesh object
    """

    def __init__(
        self,
        input_key: str = "merged_mesh",
        output_key: str = "iso_shell",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="iso_mesh_from_mesh",
            requires=[input_key],
            provides=[output_key],
            **kwargs,
        )
        self._input_key = input_key
        self._output_key = output_key

    def get_artifact(self, context):
        import bpy
        name = context.get(self._output_key)
        return bpy.data.objects.get(name) if name else None

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        import bpy
        from lib.bpy.geometry_nodes import (
            build_scatter_to_iso_mesh_tree,
            apply_gn_modifier, _PREFIX,
        )

        mesh_name = context[self._input_key]
        mesh_obj = bpy.data.objects[mesh_name]

        density = 0.010
        radius = 20.0
        voxel_size = 15.0

        logger.info(
            "IsoMeshFromMesh: GN scatter+iso (density=%.3f, radius=%.2f, voxel=%.2f)",
            density, radius, voxel_size,
        )

        ng = build_scatter_to_iso_mesh_tree(
            radius=radius,
            voxel_size=voxel_size,
            density=density,
            method="RANDOM",
        )
        apply_gn_modifier(mesh_obj, ng, name="ScatterIso")

        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = mesh_obj.evaluated_get(depsgraph)
        num_verts = len(eval_obj.data.vertices)
        num_faces = len(eval_obj.data.polygons)
        logger.info("IsoMesh: %d verts, %d faces (live GN modifier)", num_verts, num_faces)

        context[self._output_key] = mesh_obj.name
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=[self._output_key],
            outputs={
                "density": density,
                "radius": radius,
                "voxel_size": voxel_size,
                "num_verts": num_verts,
                "num_faces": num_faces,
            },
        )


class OrderPointsStep(BlenderStep):
    """Sort points by angle around the Z axis.

    Generic point-ordering step. Produces a coherent sweep path
    (clockwise) instead of random jumps.

    Constructor args:
        input_key: context key for the input point list
        output_key: context key for the ordered output
    """

    def __init__(
        self,
        input_key: str = "camera_candidates",
        output_key: str = "ordered_candidates",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="order_points",
            requires=[input_key],
            provides=[output_key],
            **kwargs,
        )
        self._input_key = input_key
        self._output_key = output_key

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        from lib.bpy.spatial import sort_by_angle_around_z

        candidates = context[self._input_key]
        ordered = sort_by_angle_around_z(candidates)

        context[self._output_key] = ordered
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=[self._output_key],
            outputs={"num_ordered": len(ordered)},
        )


class PerturbPointsStep(BlenderStep):
    """Jitter points by a seeded random offset using GN.

    Applies stratified jitter to break lattice bias from regular grids
    while preserving spatial coverage. Each point is offset by a uniform
    random vector in [-jitter_radius, +jitter_radius] per axis.

    Deterministic: same seed produces identical jitter every run.

    Constructor args:
        input_key: context key for the input point list
        output_key: context key for the jittered output
        multiplier: jitter radius as a fraction of grid spacing
            (default 0.4 — keeps points within their Voronoi cell)
    """

    def __init__(
        self,
        input_key: str = "camera_candidates",
        output_key: str = "camera_candidates_perturbed",
        multiplier: float = 0.4,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="perturb_points",
            requires=[input_key],
            provides=[output_key],
            **kwargs,
        )
        self._input_key = input_key
        self._output_key = output_key
        self._multiplier = multiplier

    def get_artifact(self, context):
        import bpy
        from lib.bpy.geometry_nodes import _PREFIX
        return bpy.data.objects.get(f"{_PREFIX}perturb_input")

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        import bpy
        from lib.bpy.geometry_nodes import (
            build_perturb_tree, apply_gn_modifier,
            read_points_from_evaluated, _link_to_temp_collection, _PREFIX,
        )

        points = context[self._input_key]
        spacing = context.get("_scatter_spacing", 1.0)
        seed = context.get("camera_seed", 0) or 0
        jitter_radius = spacing * self._multiplier

        pts_mesh = bpy.data.meshes.new(f"{_PREFIX}perturb_input")
        pts_mesh.from_pydata(points, [], [])
        pts_mesh.update()
        pts_obj = bpy.data.objects.new(f"{_PREFIX}perturb_input", pts_mesh)
        _link_to_temp_collection(pts_obj)

        ng = build_perturb_tree(jitter_radius=jitter_radius, seed=seed)
        apply_gn_modifier(pts_obj, ng, name="Perturb")

        perturbed = read_points_from_evaluated(pts_obj)

        logger.info(
            "PerturbPoints: %d points, jitter_radius=%.2f (spacing=%.2f * %.1f), seed=%d",
            len(perturbed), jitter_radius, spacing, self._multiplier, seed,
        )

        context[self._output_key] = perturbed
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=[self._output_key],
            outputs={
                "num_points": len(perturbed),
                "jitter_radius": round(jitter_radius, 4),
                "seed": seed,
            },
        )


class SampleSdfStep(BlenderStep):
    """Sample SDF at each point and store as a named attribute.

    Creates a shared Blender object from the point list and applies a
    GN modifier that samples the SDF of the reference shell. The SDF
    value is stored as "sdf_distance" on each point. Subsequent
    attribute steps stack modifiers on the same object.
    """

    def __init__(
        self,
        points_key: str = "camera_candidates_culled",
        shell_key: str = "iso_shell",
        obj_key: str = "_sdf_pts_obj",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="sample_sdf",
            requires=[points_key, shell_key],
            provides=[obj_key],
            **kwargs,
        )
        self._points_key = points_key
        self._shell_key = shell_key
        self._obj_key = obj_key

    def get_artifact(self, context):
        import bpy
        from lib.bpy.geometry_nodes import _PREFIX
        return bpy.data.objects.get(f"{_PREFIX}sdf_pts")

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        import bpy
        from lib.bpy.geometry_nodes import (
            build_sample_sdf_tree, scene_aabb, apply_gn_modifier,
            _link_to_temp_collection, _PREFIX,
        )

        points = context[self._points_key]
        shell_obj = bpy.data.objects[context[self._shell_key]]

        all_scene = [
            o for o in bpy.context.scene.objects
            if o.type == "MESH" and not o.name.startswith(_PREFIX)
        ]
        bbox_min, bbox_max = scene_aabb(all_scene)
        voxel_size = (bbox_max - bbox_min).length / 200.0

        pts_mesh = bpy.data.meshes.new(f"{_PREFIX}sdf_pts")
        pts_mesh.from_pydata(points, [], [])
        pts_mesh.update()
        pts_obj = bpy.data.objects.new(f"{_PREFIX}sdf_pts", pts_mesh)
        _link_to_temp_collection(pts_obj)

        ng = build_sample_sdf_tree(shell_obj, voxel_size)
        apply_gn_modifier(pts_obj, ng, name="SampleSDF")

        context[self._obj_key] = pts_obj.name
        logger.info("SampleSdf: stored sdf_distance on %d points", len(points))
        return CompletedState(
            success=True, timestamp=CompletedState.now_iso(), duration_s=0.0,
            provides=[self._obj_key],
            outputs={"num_points": len(points)},
        )


class NormalizeSdfStep(BlenderStep):
    """Normalize the sdf_distance attribute to 0-1 and store as sdf_normalized."""

    def __init__(self, obj_key: str = "_sdf_pts_obj", **kwargs: Any) -> None:
        super().__init__(
            name="normalize_sdf", requires=[obj_key], provides=[], **kwargs,
        )
        self._obj_key = obj_key

    def get_artifact(self, context):
        import bpy
        return bpy.data.objects.get(context.get(self._obj_key, ""))

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        import bpy
        from lib.bpy.geometry_nodes import build_normalize_attr_tree, apply_gn_modifier

        pts_obj = bpy.data.objects[context[self._obj_key]]
        ng = build_normalize_attr_tree()
        apply_gn_modifier(pts_obj, ng, name="NormalizeSDF")

        logger.info("NormalizeSdf: stored sdf_normalized")
        return CompletedState(
            success=True, timestamp=CompletedState.now_iso(), duration_s=0.0,
            provides=[], outputs={},
        )


class RandomizeWeightStep(BlenderStep):
    """Compute weight = (1 - sdf_normalized) × seeded noise, store as weight attr."""

    def __init__(self, obj_key: str = "_sdf_pts_obj", **kwargs: Any) -> None:
        super().__init__(
            name="randomize_weight", requires=[obj_key], provides=[], **kwargs,
        )
        self._obj_key = obj_key

    def get_artifact(self, context):
        import bpy
        return bpy.data.objects.get(context.get(self._obj_key, ""))

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        import bpy
        from lib.bpy.geometry_nodes import build_randomize_weight_tree, apply_gn_modifier

        pts_obj = bpy.data.objects[context[self._obj_key]]
        seed = context.get("camera_seed", 0) or 0
        ng = build_randomize_weight_tree(seed=seed)
        apply_gn_modifier(pts_obj, ng, name="RandomizeWeight")

        logger.info("RandomizeWeight: stored weight (seed=%d)", seed)
        return CompletedState(
            success=True, timestamp=CompletedState.now_iso(), duration_s=0.0,
            provides=[], outputs={"seed": seed},
        )


class ColorizeBySdfStep(BlenderStep):
    """Map weight attribute to a vertex color for debug visualization."""

    def __init__(self, obj_key: str = "_sdf_pts_obj", **kwargs: Any) -> None:
        super().__init__(
            name="colorize_by_sdf", requires=[obj_key], provides=[], **kwargs,
        )
        self._obj_key = obj_key

    def get_artifact(self, context):
        import bpy
        return bpy.data.objects.get(context.get(self._obj_key, ""))

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        import bpy
        from lib.bpy.geometry_nodes import build_colorize_attr_tree, apply_gn_modifier

        pts_obj = bpy.data.objects[context[self._obj_key]]
        ng = build_colorize_attr_tree()
        apply_gn_modifier(pts_obj, ng, name="Colorize")

        logger.info("ColorizeBySdf: stored color attribute")
        return CompletedState(
            success=True, timestamp=CompletedState.now_iso(), duration_s=0.0,
            provides=[], outputs={},
        )


class CullByWeightStep(BlenderStep):
    """Delete points where weight attribute < threshold, output surviving points."""

    def __init__(
        self,
        obj_key: str = "_sdf_pts_obj",
        output_key: str = "camera_candidates_weighted",
        threshold: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="cull_by_weight",
            requires=[obj_key],
            provides=[output_key],
            **kwargs,
        )
        self._obj_key = obj_key
        self._output_key = output_key
        self._threshold = threshold

    def get_artifact(self, context):
        import bpy
        from lib.bpy.geometry_nodes import _PREFIX
        return bpy.data.objects.get(f"{_PREFIX}sdf_pts")

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        import bpy
        from lib.bpy.geometry_nodes import (
            build_cull_by_attr_tree, apply_gn_modifier,
            read_points_from_evaluated, _link_to_temp_collection, _PREFIX,
        )

        pts_obj = bpy.data.objects[context[self._obj_key]]
        ng = build_cull_by_attr_tree(threshold=self._threshold)
        apply_gn_modifier(pts_obj, ng, name="CullByWeight")

        culled = read_points_from_evaluated(pts_obj)

        result_mesh = bpy.data.meshes.new(f"{_PREFIX}wt_culled")
        result_mesh.from_pydata(culled, [], [])
        result_mesh.update()
        result_obj = bpy.data.objects.new(f"{_PREFIX}wt_culled", result_mesh)
        _link_to_temp_collection(result_obj)

        input_count = len(context.get("camera_candidates_culled", []))
        logger.info(
            "CullByWeight: %d -> %d points (threshold=%.2f)",
            input_count, len(culled), self._threshold,
        )

        context[self._output_key] = culled
        return CompletedState(
            success=True, timestamp=CompletedState.now_iso(), duration_s=0.0,
            provides=[self._output_key],
            outputs={
                "input_count": input_count,
                "output_count": len(culled),
                "threshold": self._threshold,
            },
        )


class ScatterOnSurfaceStep(BlenderStep):
    """Scatter points on all faces of a mesh.

    Generic surface scatter step. Per-face adaptive density with
    KDTree relaxation for even spacing. The radius and point density
    are derived from the scene AABB diagonal.

    Constructor args:
        mesh_key: context key for the input mesh object
        output_key: context key for the output point list
        relax_iterations: number of Lloyd relaxation passes
    """

    def __init__(
        self,
        mesh_key: str = "merged_mesh",
        output_key: str = "scatter_points",
        relax_iterations: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="scatter_on_surface",
            requires=[mesh_key],
            provides=[output_key],
            **kwargs,
        )
        self._mesh_key = mesh_key
        self._output_key = output_key
        self._relax_iterations = relax_iterations

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        import bpy
        from lib.bpy.geometry_nodes import scene_aabb, _link_to_temp_collection, _PREFIX

        mesh_name = context[self._mesh_key]
        mesh_obj = bpy.data.objects[mesh_name]

        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = mesh_obj.evaluated_get(depsgraph)
        mesh_data = eval_obj.data

        all_scene_meshes = [
            o for o in bpy.context.scene.objects
            if o.type == "MESH" and not o.name.startswith(_PREFIX)
        ]
        bbox_min, bbox_max = scene_aabb(all_scene_meshes)
        diag = (bbox_max - bbox_min).length
        voxel_size = diag / 200.0
        radius = voxel_size * 2.0
        overlap_factor = 0.7

        points = _per_face_scatter(mesh_data, radius, overlap_factor)
        logger.info("Surface scatter: %d points on %d faces", len(points), len(mesh_data.polygons))

        if points and self._relax_iterations > 0:
            points = _relax_points_kdtree(points, iterations=self._relax_iterations)
            logger.info("Relaxed %d points (%d iterations)", len(points), self._relax_iterations)

        pts_mesh = bpy.data.meshes.new(f"{_PREFIX}surface_pts")
        pts_mesh.from_pydata(points, [], [])
        pts_mesh.update()
        pts_obj = bpy.data.objects.new(f"{_PREFIX}surface_pts", pts_mesh)
        _link_to_temp_collection(pts_obj)

        context[self._output_key] = points
        context["_iso_voxel_size"] = voxel_size
        context["_iso_radius"] = radius
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=[self._output_key],
            outputs={"num_points": len(points), "radius": round(radius, 4)},
        )


class IsoMeshFromPointsStep(BlenderStep):
    """Convert a point cloud to a watertight iso-surface mesh.

    Generic step: takes a list of points, creates a mesh object from them,
    applies Points to SDF Grid + Grid to Mesh via GN, and outputs the
    resulting watertight shell.

    Constructor args:
        points_key: context key for the input point list
        output_key: context key for the output mesh object name
    """

    def __init__(
        self,
        points_key: str = "scatter_points",
        output_key: str = "iso_mesh",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="iso_mesh_from_points",
            requires=[points_key],
            provides=[output_key],
            **kwargs,
        )
        self._points_key = points_key
        self._output_key = output_key

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        import bpy
        from lib.bpy.geometry_nodes import (
            build_points_to_iso_mesh_tree, apply_gn_modifier,
            evaluated_mesh_copy, _link_to_temp_collection, _PREFIX,
        )

        points = context[self._points_key]
        voxel_size = context.get("_iso_voxel_size", 10.0)
        radius = context.get("_iso_radius", voxel_size * 1.5)

        if not points:
            empty_mesh = bpy.data.meshes.new(f"{_PREFIX}iso_empty")
            empty_obj = bpy.data.objects.new(f"{_PREFIX}iso_empty", empty_mesh)
            _link_to_temp_collection(empty_obj)
            context[self._output_key] = empty_obj.name
            return CompletedState(
                success=True,
                timestamp=CompletedState.now_iso(),
                duration_s=0.0,
                provides=[self._output_key],
                outputs={"num_verts": 0},
            )

        pts_mesh = bpy.data.meshes.new(f"{_PREFIX}iso_input")
        pts_mesh.from_pydata(points, [], [])
        pts_mesh.update()
        pts_obj = bpy.data.objects.new(f"{_PREFIX}iso_input", pts_mesh)
        _link_to_temp_collection(pts_obj)

        ng = build_points_to_iso_mesh_tree(radius=radius, voxel_size=voxel_size)
        apply_gn_modifier(pts_obj, ng, name="IsoMesh")

        iso_obj = evaluated_mesh_copy(pts_obj, name="iso_shell")

        num_verts = len(iso_obj.data.vertices)
        num_faces = len(iso_obj.data.polygons)
        logger.info("IsoMesh: %d verts, %d faces", num_verts, num_faces)

        context[self._output_key] = iso_obj.name
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=[self._output_key],
            outputs={"num_verts": num_verts, "num_faces": num_faces},
        )


class CreateBBoxMeshStep(BlenderStep):
    """Create an AABB bounding box as a simple mesh object.

    No SDF conversion — just the box mesh for use as a scatter boundary
    or culling reference.

    Constructor args:
        output_key: context key for the output mesh object name
    """

    def __init__(
        self,
        output_key: str = "bbox_mesh",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="create_bbox_mesh",
            requires=["scene_loaded"],
            provides=[output_key],
            **kwargs,
        )
        self._output_key = output_key

    def get_artifact(self, context):
        import bpy
        name = context.get(self._output_key)
        return bpy.data.objects.get(name) if name else None

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        import bpy
        from lib.bpy.geometry_nodes import scene_aabb, create_aabb_box, _PREFIX

        all_meshes = [
            o for o in bpy.context.scene.objects
            if o.type == "MESH" and not o.name.startswith(_PREFIX)
        ]
        bbox_min, bbox_max = scene_aabb(all_meshes)

        box_obj = create_aabb_box(bbox_min, bbox_max, padding=0.0)

        context[self._output_key] = box_obj.name
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=[self._output_key],
            outputs={"box_obj": box_obj.name},
        )


class ScatterPointsInMeshStep(BlenderStep):
    """Scatter a grid of points inside a mesh boundary via GN.

    Uses Mesh to Volume -> Distribute Points in Volume (GRID mode) entirely
    within the Geometry Nodes engine. No Python geometry loops.

    Constructor args:
        mesh_key: context key for the boundary mesh
        output_key: context key for the output point list
    """

    def __init__(
        self,
        mesh_key: str = "bbox_mesh",
        output_key: str = "camera_candidates",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="scatter_points_in_mesh",
            requires=[mesh_key],
            provides=[output_key],
            **kwargs,
        )
        self._mesh_key = mesh_key
        self._output_key = output_key

    def get_artifact(self, context):
        import bpy
        name = context.get(self._mesh_key)
        return bpy.data.objects.get(name) if name else None

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        import bpy
        from lib.bpy.geometry_nodes import (
            scene_aabb, build_vol_scatter_tree, apply_gn_modifier,
            read_points_from_evaluated, _PREFIX,
        )

        mesh_name = context[self._mesh_key]
        mesh_obj = bpy.data.objects[mesh_name]

        all_scene = [
            o for o in bpy.context.scene.objects
            if o.type == "MESH" and not o.name.startswith(_PREFIX)
        ]
        bbox_min, bbox_max = scene_aabb(all_scene)
        dims = bbox_max - bbox_min
        min_dim = min(dims.x, dims.y, dims.z)
        diag = (bbox_max - bbox_min).length
        spacing = diag / 200.0

        voxel_size = min(spacing, min_dim / 4.0)
        interior_band = 1.0

        context["_scatter_spacing"] = spacing

        ng = build_vol_scatter_tree(
            voxel_size=voxel_size,
            spacing=spacing,
            interior_band=interior_band,
        )
        apply_gn_modifier(mesh_obj, ng, name="VolScatter")

        candidates = read_points_from_evaluated(mesh_obj)

        if len(candidates) < 2:
            return CompletedState(
                success=False,
                timestamp=CompletedState.now_iso(),
                duration_s=0.0,
                error={
                    "message": f"Only {len(candidates)} points (need >= 2). "
                    "Try lowering --threshold.",
                },
            )

        logger.info(
            "Scattered %d points in mesh '%s' via GN VolScatter (spacing=%.2f)",
            len(candidates), mesh_name, spacing,
        )
        context[self._output_key] = candidates
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=[self._output_key],
            outputs={"num_candidates": len(candidates), "spacing": round(spacing, 2)},
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _relax_points_kdtree(
    points: List[Tuple[float, float, float]],
    iterations: int = 4,
    k_neighbors: int = 8,
    damping: float = 0.3,
) -> List[Tuple[float, float, float]]:
    """Lloyd's relaxation via KDTree nearest-neighbor averaging.

    Each point moves toward the centroid of its K nearest neighbors,
    scaled by the damping factor. Repeated iterations produce an
    increasingly even distribution.
    """
    from mathutils.kdtree import KDTree

    pts = [list(p) for p in points]
    n = len(pts)
    if n < 2:
        return points

    for iteration in range(iterations):
        kd = KDTree(n)
        for i, p in enumerate(pts):
            kd.insert(p, i)
        kd.balance()

        new_pts = []
        for i, p in enumerate(pts):
            neighbors = kd.find_n(p, k_neighbors + 1)
            cx, cy, cz = 0.0, 0.0, 0.0
            count = 0
            for co, _idx, _dist in neighbors:
                if _idx != i:
                    cx += co[0]
                    cy += co[1]
                    cz += co[2]
                    count += 1
            if count > 0:
                cx /= count
                cy /= count
                cz /= count
                new_pts.append([
                    p[0] + (cx - p[0]) * damping,
                    p[1] + (cy - p[1]) * damping,
                    p[2] + (cz - p[2]) * damping,
                ])
            else:
                new_pts.append(list(p))
        pts = new_pts

    return [tuple(p) for p in pts]


def _per_face_scatter(
    mesh_data,
    radius: float,
    overlap_factor: float = 0.7,
) -> List[Tuple[float, float, float]]:
    """Scatter points on mesh faces with per-face adaptive density.

    For each polygon, compute its area and scatter enough points to
    guarantee coverage at the given radius with overlap.
    """
    from mathutils import Vector

    points = []
    disc_area = math.pi * radius * radius * overlap_factor

    for poly in mesh_data.polygons:
        face_area = poly.area
        if face_area <= 0:
            continue

        n_points = max(1, math.ceil(face_area / disc_area))

        verts = [Vector(mesh_data.vertices[vi].co) for vi in poly.vertices]

        if len(verts) < 3:
            continue

        v0 = verts[0]
        for _ in range(n_points):
            tri_idx = random.randint(0, len(verts) - 3) if len(verts) > 3 else 0
            v1 = verts[tri_idx + 1]
            v2 = verts[tri_idx + 2]

            r1, r2 = random.random(), random.random()
            if r1 + r2 > 1.0:
                r1, r2 = 1.0 - r1, 1.0 - r2
            pt = v0 + r1 * (v1 - v0) + r2 * (v2 - v0)
            points.append((pt.x, pt.y, pt.z))

    return points


def _is_inside_mesh(point, bvh, direction=None):
    """Ray-parity inside/outside test: odd hit count = inside."""
    from mathutils import Vector
    if direction is None:
        direction = Vector((0, 0, 1))
    hits = 0
    origin = point.copy()
    max_dist = 1e7
    nudge = 1e-4
    while True:
        hit_loc, _normal, _index, dist = bvh.ray_cast(origin, direction, max_dist)
        if hit_loc is None:
            break
        hits += 1
        origin = hit_loc + direction * nudge
    return hits % 2 == 1
