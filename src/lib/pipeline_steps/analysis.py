"""Analysis steps: mesh collection and geometry computation."""

from __future__ import annotations

from typing import Any, Dict

from lib.pipeline import CompletedState, _short_signature
from lib.pipeline_steps.blender_step import BlenderStep


class CollectMeshesStep(BlenderStep):
    """Collect all MESH-type objects from the current scene."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="collect_meshes",
            requires=["imported_objects"],
            provides=["mesh_objects"],
            **kwargs,
        )

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        from lib.bpy.mesh import collect_meshes
        meshes = collect_meshes()
        context["mesh_objects"] = meshes
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=["mesh_objects"],
            outputs={"mesh_count": len(meshes)},
        )


class ComputeCountsStep(BlenderStep):
    """Compute geometry counts, topology breakdown, and bounding box."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="compute_counts",
            requires=["mesh_objects"],
            provides=["counts"],
            **kwargs,
        )

    def execute(self, context: Dict[str, Any]) -> CompletedState:
        from lib.bpy.mesh import (
            compute_geometry_counts,
            compute_topology,
            compute_world_bbox,
        )
        meshes = context["mesh_objects"]
        counts = compute_geometry_counts(meshes)
        topology = compute_topology(meshes)
        bbox = compute_world_bbox(meshes)
        context["counts"] = counts
        context["topology"] = topology
        context["bbox"] = bbox
        return CompletedState(
            success=True,
            timestamp=CompletedState.now_iso(),
            duration_s=0.0,
            provides=["counts"],
            outputs={"counts": counts, "topology": topology},
            signature=_short_signature(counts),
        )
