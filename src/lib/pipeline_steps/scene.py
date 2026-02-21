"""Scene lifecycle steps: prepare, normalize, merge, and cleanup."""

from __future__ import annotations

import logging
from typing import Any, Dict

from lib.pipeline import CompletedState
from lib.pipeline_steps.blender_step import BlenderStep

logger = logging.getLogger("pipeline.steps.scene")


class NormalizeSceneStep(BlenderStep):
    """Normalize scene settings so downstream steps get predictable geometry.

    1. Set unit scale_length to 1.0 so code values = UI display values.
    2. Apply all object transforms (location, rotation, scale) so
       matrix_world is identity for every mesh object.
    3. Apply all pending modifiers so mesh data matches viewport.
    4. Recalculate normals on objects that had negative scale (flipped).

    This step is destructive to the scene — it modifies objects in place.
    Run it early (right after loading) before any pipeline geometry work.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="normalize_scene",
            requires=["scene_loaded"],
            provides=["scene_normalized"],
            **kwargs,
        )

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        import bpy

        scene = bpy.context.scene
        old_scale = scene.unit_settings.scale_length

        mesh_objects = [o for o in scene.objects if o.type == "MESH"]

        scene.unit_settings.scale_length = 1.0

        neg_scale_objs = [
            o for o in mesh_objects
            if o.scale.x < 0 or o.scale.y < 0 or o.scale.z < 0
        ]
        non_unit_transforms = [
            o for o in mesh_objects
            if (abs(o.scale.x - 1.0) > 0.001
                or abs(o.scale.y - 1.0) > 0.001
                or abs(o.scale.z - 1.0) > 0.001
                or abs(o.rotation_euler.x) > 0.001
                or abs(o.rotation_euler.y) > 0.001
                or abs(o.rotation_euler.z) > 0.001)
        ]
        objects_with_modifiers = [o for o in mesh_objects if o.modifiers]

        multi_user = 0
        for obj in mesh_objects:
            if obj.data.users > 1:
                obj.data = obj.data.copy()
                multi_user += 1

        bpy.ops.object.select_all(action="DESELECT")
        for obj in mesh_objects:
            obj.select_set(True)
        if mesh_objects:
            bpy.context.view_layer.objects.active = mesh_objects[0]
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        bpy.ops.object.select_all(action="DESELECT")

        applied_mods = 0
        for obj in objects_with_modifiers:
            bpy.context.view_layer.objects.active = obj
            for mod in list(obj.modifiers):
                try:
                    bpy.ops.object.modifier_apply(modifier=mod.name)
                    applied_mods += 1
                except RuntimeError:
                    pass

        normals_fixed = 0
        for obj in neg_scale_objs:
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.object.mode_set(mode="OBJECT")
            normals_fixed += 1

        logger.info(
            "NormalizeScene: scale_length %.4f → 1.0, %d multi-user meshes made single, "
            "transforms applied on %d objects, %d modifiers applied, %d normal fixes",
            old_scale, multi_user, len(non_unit_transforms), applied_mods, normals_fixed,
        )

        context["scene_normalized"] = True
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=["scene_normalized"],
            outputs={
                "old_scale_length": round(old_scale, 6),
                "multi_user_fixed": multi_user,
                "transforms_applied": len(non_unit_transforms),
                "modifiers_applied": applied_mods,
                "normals_fixed": normals_fixed,
                "total_mesh_objects": len(mesh_objects),
            },
        )


class PrepareSceneStep(BlenderStep):
    """Reset the Blender scene to an empty state."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="prepare_scene", provides=["scene_prepared"], **kwargs)

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        from lib.bpy.scene import reset_scene
        reset_scene()
        context["scene_prepared"] = True
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=["scene_prepared"],
            outputs={"note": "scene reset to empty state"},
        )


class MergeMeshesStep(BlenderStep):
    """Join all scene mesh objects into a single temporary object.

    The merged mesh is in world space with all transforms applied.
    Original scene objects are NOT modified — this creates a copy.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="merge_meshes",
            requires=["scene_loaded"],
            provides=["merged_mesh"],
            **kwargs,
        )

    def get_artifact(self, context):
        import bpy
        name = context.get("merged_mesh")
        return bpy.data.objects.get(name) if name else None

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        import bpy
        from lib.bpy.geometry_nodes import join_scene_meshes, _PREFIX

        scene = bpy.context.scene
        mesh_objects = [
            o for o in scene.objects
            if o.type == "MESH" and not o.name.startswith(_PREFIX)
        ]
        if not mesh_objects:
            return CompletedState(
                success=False,
                timestamp=CompletedState.now_iso(),
                duration_s=0.0,
                error={"message": "no mesh objects in scene"},
            )

        merged_obj = join_scene_meshes(mesh_objects)
        context["merged_mesh"] = merged_obj.name
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=["merged_mesh"],
            outputs={"merged_obj": merged_obj.name, "num_meshes": len(mesh_objects)},
        )


class CleanupSceneStep(BlenderStep):
    """Reset the scene after processing (free memory for batch loops)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="cleanup_scene", provides=["cleaned"], **kwargs)

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        from lib.bpy.scene import reset_scene
        reset_scene()
        context["cleaned"] = True
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=["cleaned"],
            outputs={"note": "scene cleaned"},
        )
