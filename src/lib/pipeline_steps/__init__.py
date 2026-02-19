"""Reusable pipeline step library.

Each module contains PipelineStep subclasses organized by domain.
These are the building blocks ("nodes") for composing pipelines.

Modules:
    blender_step — BlenderStep base class (for bpy-dependent steps)
    scene        — PrepareSceneStep, CleanupSceneStep
    io           — ImportModelStep, GrepModelsStep
    analysis     — CollectMeshesStep, ComputeCountsStep
    camera       — SetupCameraStep
    lighting     — SetupEnvironmentLightStep
    render       — ConfigureRendererStep, RenderTexturedStep, RenderNormalStep,
                   RenderDepthStep, RenderEdgeStep
    reporting    — WriteJSONStep, WriteMetadataStep
"""

from lib.pipeline_steps.blender_step import BlenderStep

__all__ = ["BlenderStep"]
