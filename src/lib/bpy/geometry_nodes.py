"""Geometry Nodes utilities: atomic GN tree builders for SDF operations.

Each function builds a single-purpose Geometry Nodes tree. Pipeline steps
in lib/pipeline_steps/volume.py call these to construct their GN modifiers.

Uses Blender 5.0 volume grid nodes: Mesh to SDF Grid, SDF Grid Boolean,
Grid to Mesh, Mesh to Volume, Distribute Points in Volume.

All ``build_*`` functions are decorated with ``@auto_layout`` which
automatically positions nodes left-to-right following link topology.
"""

from __future__ import annotations

import functools
import logging
from typing import Callable, List, Optional, Tuple

import bpy
from mathutils import Vector

logger = logging.getLogger("pipeline.bpy.geometry_nodes")

_PREFIX = "_proc_cam_"
_COLLECTION_NAME = "_proc_cam"


# ---------------------------------------------------------------------------
# Node tree auto-layout
# ---------------------------------------------------------------------------

def layout_node_tree(
    node_tree: bpy.types.NodeTree,
    gap_x: float = 250.0,
    gap_y: float = 150.0,
) -> None:
    """Position nodes in a GN tree left-to-right following link topology.

    Uses longest-path depth to assign columns, then orders nodes within
    each column by the average Y of their upstream neighbours to reduce
    link crossings (simplified Sugiyama).

    Args:
        node_tree: The Geometry Nodes tree to lay out.
        gap_x: Horizontal spacing between columns.
        gap_y: Vertical spacing between nodes in the same column.
    """
    nodes = list(node_tree.nodes)
    if not nodes:
        return

    upstream: dict = {n: set() for n in nodes}
    for link in node_tree.links:
        upstream[link.to_node].add(link.from_node)

    depth: dict = {}

    def _depth_of(node, visiting=None):
        if node in depth:
            return depth[node]
        if visiting is None:
            visiting = set()
        if node in visiting:
            depth[node] = 0
            return 0
        visiting.add(node)
        parents = upstream[node]
        depth[node] = 0 if not parents else max(_depth_of(p, visiting) + 1 for p in parents)
        return depth[node]

    for n in nodes:
        _depth_of(n)

    max_non_output = max(
        (d for n, d in depth.items() if n.bl_idname != "NodeGroupOutput"),
        default=0,
    )
    for n in nodes:
        if n.bl_idname == "NodeGroupOutput":
            depth[n] = max_non_output + 1

    columns: dict = {}
    for n in nodes:
        columns.setdefault(depth[n], []).append(n)

    for d in sorted(columns):
        col = columns[d]
        if d > 0:
            def _avg_upstream_y(node, _up=upstream):
                parents = _up[node]
                if not parents:
                    return 0.0
                return sum(p.location[1] for p in parents) / len(parents)
            col.sort(key=_avg_upstream_y, reverse=True)

        x = d * gap_x
        total_h = (len(col) - 1) * gap_y
        y_start = total_h / 2.0
        for i, node in enumerate(col):
            node.location = (x, y_start - i * gap_y)


def auto_layout(fn: Callable[..., bpy.types.NodeTree]) -> Callable[..., bpy.types.NodeTree]:
    """Decorator: auto-layout the returned NodeTree left-to-right.

    Apply to any ``build_*`` function that returns a ``bpy.types.NodeTree``.
    The layout runs after the function body, so all nodes and links are
    already in place.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        node_tree = fn(*args, **kwargs)
        if isinstance(node_tree, bpy.types.NodeTree):
            layout_node_tree(node_tree)
        return node_tree
    return wrapper


def get_temp_collection() -> bpy.types.Collection:
    """Get or create the top-level temp collection for pipeline objects.

    All pipeline temp objects go here so they're easy to find in the
    outliner and can be toggled as a group.
    """
    if _COLLECTION_NAME in bpy.data.collections:
        return bpy.data.collections[_COLLECTION_NAME]

    col = bpy.data.collections.new(_COLLECTION_NAME)
    bpy.context.scene.collection.children.link(col)
    logger.debug("Created temp collection '%s'", _COLLECTION_NAME)
    return col


def _link_to_temp_collection(obj: bpy.types.Object) -> None:
    """Link an object to the temp collection (and unlink from others)."""
    col = get_temp_collection()
    if obj.name not in col.objects:
        col.objects.link(obj)
    for other_col in obj.users_collection:
        if other_col != col:
            other_col.objects.unlink(obj)


# ---------------------------------------------------------------------------
# Atomic GN tree builders
# ---------------------------------------------------------------------------

@auto_layout
def build_mesh_to_sdf_tree(voxel_size: float) -> bpy.types.NodeTree:
    """GN tree: input mesh -> Mesh to SDF Grid -> output mesh (via Grid to Mesh).

    The SDF grid is converted to a mesh at threshold=0 so the result is
    visible in the viewport for inspection.
    """
    ng = bpy.data.node_groups.new(f"{_PREFIX}mesh_to_sdf", "GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    inp = ng.nodes.new("NodeGroupInput")
    out = ng.nodes.new("NodeGroupOutput")

    sdf = ng.nodes.new("GeometryNodeMeshToSDFGrid")
    sdf.inputs["Voxel Size"].default_value = voxel_size

    g2m = ng.nodes.new("GeometryNodeGridToMesh")
    g2m.inputs["Threshold"].default_value = 0.0
    g2m.inputs["Adaptivity"].default_value = 0.1

    ng.links.new(inp.outputs["Geometry"], sdf.inputs["Mesh"])
    ng.links.new(sdf.outputs["SDF Grid"], g2m.inputs["Grid"])
    ng.links.new(g2m.outputs["Mesh"], out.inputs["Geometry"])

    return ng


@auto_layout
def build_sdf_boolean_tree(
    voxel_size: float,
    box_obj: bpy.types.Object,
) -> bpy.types.NodeTree:
    """GN tree: host geometry SDF DIFFERENCE with box object SDF.

    The host object's mesh flows through the first Geometry input.
    The box is referenced via an Object Info node.
    Output is the negative space boundary mesh (Grid to Mesh at threshold=0).
    """
    ng = bpy.data.node_groups.new(f"{_PREFIX}sdf_boolean", "GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    inp = ng.nodes.new("NodeGroupInput")
    out = ng.nodes.new("NodeGroupOutput")

    obj_info = ng.nodes.new("GeometryNodeObjectInfo")
    obj_info.inputs["Object"].default_value = box_obj
    obj_info.inputs["As Instance"].default_value = False

    sdf_geo = ng.nodes.new("GeometryNodeMeshToSDFGrid")
    sdf_geo.inputs["Voxel Size"].default_value = voxel_size

    sdf_box = ng.nodes.new("GeometryNodeMeshToSDFGrid")
    sdf_box.inputs["Voxel Size"].default_value = voxel_size

    boolean = ng.nodes.new("GeometryNodeSDFGridBoolean")
    boolean.operation = "DIFFERENCE"

    g2m = ng.nodes.new("GeometryNodeGridToMesh")
    g2m.inputs["Threshold"].default_value = 0.0
    g2m.inputs["Adaptivity"].default_value = 0.1

    ng.links.new(inp.outputs["Geometry"], sdf_geo.inputs["Mesh"])
    ng.links.new(obj_info.outputs["Geometry"], sdf_box.inputs["Mesh"])
    ng.links.new(sdf_box.outputs["SDF Grid"], boolean.inputs["Grid 1"])
    ng.links.new(sdf_geo.outputs["SDF Grid"], boolean.inputs["Grid 2"])
    ng.links.new(boolean.outputs["Grid"], g2m.inputs["Grid"])
    ng.links.new(g2m.outputs["Mesh"], out.inputs["Geometry"])

    return ng


@auto_layout
def build_poisson_scatter_tree(
    distance_min: float,
    density_max: float = 1000.0,
) -> bpy.types.NodeTree:
    """GN tree: input mesh -> Poisson Disk scatter -> output points.

    Just the scatter — no SDF conversion. Used per-object for fast
    Poisson scatter on small individual meshes.
    """
    ng = bpy.data.node_groups.new(f"{_PREFIX}poisson_pts", "GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    inp = ng.nodes.new("NodeGroupInput")
    out = ng.nodes.new("NodeGroupOutput")

    scatter = ng.nodes.new("GeometryNodeDistributePointsOnFaces")
    scatter.distribute_method = "POISSON"
    scatter.inputs["Distance Min"].default_value = distance_min
    scatter.inputs["Density Max"].default_value = density_max

    ng.links.new(inp.outputs["Geometry"], scatter.inputs["Mesh"])
    ng.links.new(scatter.outputs["Points"], out.inputs["Geometry"])

    return ng


@auto_layout
def build_scatter_to_iso_mesh_tree(
    radius: float,
    voxel_size: float,
    distance_min: float = 0.0,
    density: float = 0.0,
    method: str = "POISSON",
) -> bpy.types.NodeTree:
    """GN tree: input mesh -> scatter on faces -> Points to SDF Grid -> Grid to Mesh.

    Runs entirely in the GN engine. Supports POISSON (even spacing) or
    RANDOM (faster, less even) scatter modes.
    """
    ng = bpy.data.node_groups.new(f"{_PREFIX}scatter_iso", "GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    inp = ng.nodes.new("NodeGroupInput")
    out = ng.nodes.new("NodeGroupOutput")

    scatter = ng.nodes.new("GeometryNodeDistributePointsOnFaces")
    scatter.distribute_method = method
    if method == "POISSON":
        scatter.inputs["Distance Min"].default_value = distance_min
        scatter.inputs["Density Max"].default_value = 1000.0
    else:
        scatter.inputs["Density"].default_value = density

    pts_to_sdf = ng.nodes.new("GeometryNodePointsToSDFGrid")
    pts_to_sdf.inputs["Radius"].default_value = radius
    pts_to_sdf.inputs["Voxel Size"].default_value = voxel_size

    g2m = ng.nodes.new("GeometryNodeGridToMesh")
    g2m.inputs["Threshold"].default_value = 0.0
    g2m.inputs["Adaptivity"].default_value = 0.1

    ng.links.new(inp.outputs["Geometry"], scatter.inputs["Mesh"])
    ng.links.new(scatter.outputs["Points"], pts_to_sdf.inputs["Points"])
    ng.links.new(pts_to_sdf.outputs["SDF Grid"], g2m.inputs["Grid"])
    ng.links.new(g2m.outputs["Mesh"], out.inputs["Geometry"])

    return ng


@auto_layout
def build_sdf_cull_tree(
    shell_obj: bpy.types.Object,
    voxel_size: float,
) -> bpy.types.NodeTree:
    """GN tree: input points -> sample SDF of shell -> delete points inside.

    Uses Mesh to SDF Grid on the reference shell (via Object Info),
    then Sample Grid at each point's position. Points where SDF < 0
    (inside the shell) are deleted.
    """
    ng = bpy.data.node_groups.new(f"{_PREFIX}sdf_cull", "GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    inp = ng.nodes.new("NodeGroupInput")
    out = ng.nodes.new("NodeGroupOutput")

    obj_info = ng.nodes.new("GeometryNodeObjectInfo")
    obj_info.inputs["Object"].default_value = shell_obj
    obj_info.inputs["As Instance"].default_value = False

    sdf = ng.nodes.new("GeometryNodeMeshToSDFGrid")
    sdf.inputs["Voxel Size"].default_value = voxel_size

    position = ng.nodes.new("GeometryNodeInputPosition")

    sample = ng.nodes.new("GeometryNodeSampleGrid")

    compare = ng.nodes.new("FunctionNodeCompare")
    compare.operation = "LESS_THAN"
    compare.inputs[1].default_value = 0.0  # B = 0 (threshold)

    delete = ng.nodes.new("GeometryNodeDeleteGeometry")
    delete.domain = "POINT"

    ng.links.new(obj_info.outputs["Geometry"], sdf.inputs["Mesh"])
    ng.links.new(sdf.outputs["SDF Grid"], sample.inputs["Grid"])
    ng.links.new(position.outputs["Position"], sample.inputs["Position"])
    ng.links.new(sample.outputs["Value"], compare.inputs[0])  # A = SDF value
    ng.links.new(compare.outputs["Result"], delete.inputs["Selection"])
    ng.links.new(inp.outputs["Geometry"], delete.inputs["Geometry"])
    ng.links.new(delete.outputs["Geometry"], out.inputs["Geometry"])

    return ng


@auto_layout
def build_points_to_iso_mesh_tree(
    radius: float,
    voxel_size: float,
) -> bpy.types.NodeTree:
    """GN tree: input points -> Points to SDF Grid -> Grid to Mesh -> output mesh.

    Converts a point cloud into a watertight shell mesh via SDF reconstruction.
    Each point contributes a sphere of the given radius to the SDF.
    """
    ng = bpy.data.node_groups.new(f"{_PREFIX}pts_to_iso", "GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    inp = ng.nodes.new("NodeGroupInput")
    out = ng.nodes.new("NodeGroupOutput")

    pts_to_sdf = ng.nodes.new("GeometryNodePointsToSDFGrid")
    pts_to_sdf.inputs["Radius"].default_value = radius
    pts_to_sdf.inputs["Voxel Size"].default_value = voxel_size

    g2m = ng.nodes.new("GeometryNodeGridToMesh")
    g2m.inputs["Threshold"].default_value = 0.0
    g2m.inputs["Adaptivity"].default_value = 0.1

    ng.links.new(inp.outputs["Geometry"], pts_to_sdf.inputs["Points"])
    ng.links.new(pts_to_sdf.outputs["SDF Grid"], g2m.inputs["Grid"])
    ng.links.new(g2m.outputs["Mesh"], out.inputs["Geometry"])

    return ng


@auto_layout
def build_poisson_scatter_to_iso_mesh_tree(
    distance_min: float,
    density_max: float,
    radius: float,
    voxel_size: float,
) -> bpy.types.NodeTree:
    """GN tree: input mesh -> Poisson Disk scatter -> Points to SDF Grid -> Grid to Mesh.

    Single GN tree that takes the single-sided face mesh, scatters points
    evenly via Poisson Disk distribution, then converts the point cloud
    to a watertight shell mesh via SDF reconstruction.

    Poisson Disk guarantees even spacing — no gaps, no clumps.
    """
    ng = bpy.data.node_groups.new(f"{_PREFIX}poisson_iso", "GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    inp = ng.nodes.new("NodeGroupInput")
    out = ng.nodes.new("NodeGroupOutput")

    scatter = ng.nodes.new("GeometryNodeDistributePointsOnFaces")
    scatter.distribute_method = "POISSON"
    scatter.inputs["Distance Min"].default_value = distance_min
    scatter.inputs["Density Max"].default_value = density_max

    pts_to_sdf = ng.nodes.new("GeometryNodePointsToSDFGrid")
    pts_to_sdf.inputs["Radius"].default_value = radius
    pts_to_sdf.inputs["Voxel Size"].default_value = voxel_size

    g2m = ng.nodes.new("GeometryNodeGridToMesh")
    g2m.inputs["Threshold"].default_value = 0.0
    g2m.inputs["Adaptivity"].default_value = 0.1

    ng.links.new(inp.outputs["Geometry"], scatter.inputs["Mesh"])
    ng.links.new(scatter.outputs["Points"], pts_to_sdf.inputs["Points"])
    ng.links.new(pts_to_sdf.outputs["SDF Grid"], g2m.inputs["Grid"])
    ng.links.new(g2m.outputs["Mesh"], out.inputs["Geometry"])

    return ng


@auto_layout
def build_vol_scatter_tree(
    voxel_size: float,
    spacing: float,
    threshold: float = 0.5,
    interior_band: float = 0.0,
) -> bpy.types.NodeTree:
    """GN tree: input mesh -> Mesh to Volume -> Distribute Points in Volume (GRID).

    Scatters a regular grid of points inside the mesh boundary.
    voxel_size controls volume resolution, spacing controls point density.
    interior_band controls how deep the fog extends inward from the surface;
    set to >= half the min dimension of the mesh to fill the entire interior.

    The PointCloud output is converted to Mesh vertices via
    Instance on Points + Realize Instances so that the result is
    readable through the evaluated depsgraph Python API.
    """
    ng = bpy.data.node_groups.new(f"{_PREFIX}scatter", "GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    inp = ng.nodes.new("NodeGroupInput")
    out = ng.nodes.new("NodeGroupOutput")

    m2v = ng.nodes.new("GeometryNodeMeshToVolume")
    m2v.inputs["Resolution Mode"].default_value = "Size"
    m2v.inputs["Voxel Size"].default_value = voxel_size
    m2v.inputs["Density"].default_value = 1.0
    if interior_band > 0:
        m2v.inputs["Interior Band Width"].default_value = interior_band

    scatter = ng.nodes.new("GeometryNodeDistributePointsInVolume")
    scatter.inputs["Mode"].default_value = "Grid"
    scatter.inputs["Spacing"].default_value = (spacing, spacing, spacing)
    scatter.inputs["Threshold"].default_value = threshold

    # PointCloud -> Mesh conversion so Python can read the result.
    # Instance a single vertex on each point, then realize.
    mesh_line = ng.nodes.new("GeometryNodeMeshLine")
    mesh_line.inputs["Count"].default_value = 1
    mesh_line.inputs["Offset"].default_value = (0, 0, 0)

    instance = ng.nodes.new("GeometryNodeInstanceOnPoints")
    realize = ng.nodes.new("GeometryNodeRealizeInstances")

    ng.links.new(inp.outputs["Geometry"], m2v.inputs["Mesh"])
    ng.links.new(m2v.outputs["Volume"], scatter.inputs["Volume"])
    ng.links.new(scatter.outputs["Points"], instance.inputs["Points"])
    ng.links.new(mesh_line.outputs["Mesh"], instance.inputs["Instance"])
    ng.links.new(instance.outputs["Instances"], realize.inputs["Geometry"])
    ng.links.new(realize.outputs["Geometry"], out.inputs["Geometry"])

    return ng


@auto_layout
def build_perturb_tree(
    jitter_radius: float,
    seed: int = 0,
) -> bpy.types.NodeTree:
    """GN tree: jitter each point by a seeded random offset.

    Adds a uniform random 3D offset in [-jitter_radius, +jitter_radius]
    per axis to every point. Deterministic: same seed + same point index
    = same offset every time.

    Used for stratified jitter on a regular grid to break lattice bias
    while preserving coverage.
    """
    ng = bpy.data.node_groups.new(f"{_PREFIX}perturb", "GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    inp = ng.nodes.new("NodeGroupInput")
    out = ng.nodes.new("NodeGroupOutput")

    rand = ng.nodes.new("FunctionNodeRandomValue")
    rand.data_type = "FLOAT_VECTOR"
    r = jitter_radius
    rand.inputs["Min"].default_value = (-r, -r, -r)
    rand.inputs["Max"].default_value = (r, r, r)
    rand.inputs["Seed"].default_value = seed

    set_pos = ng.nodes.new("GeometryNodeSetPosition")

    ng.links.new(inp.outputs["Geometry"], set_pos.inputs["Geometry"])
    ng.links.new(rand.outputs["Value"], set_pos.inputs["Offset"])
    ng.links.new(set_pos.outputs["Geometry"], out.inputs["Geometry"])

    return ng


@auto_layout
def build_sample_sdf_tree(
    shell_obj: bpy.types.Object,
    voxel_size: float,
    attr_name: str = "sdf_distance",
) -> bpy.types.NodeTree:
    """GN tree: sample SDF at each point and store as a named attribute."""
    ng = bpy.data.node_groups.new(f"{_PREFIX}sample_sdf", "GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    inp = ng.nodes.new("NodeGroupInput")
    out = ng.nodes.new("NodeGroupOutput")

    obj_info = ng.nodes.new("GeometryNodeObjectInfo")
    obj_info.inputs["Object"].default_value = shell_obj
    obj_info.inputs["As Instance"].default_value = False

    sdf = ng.nodes.new("GeometryNodeMeshToSDFGrid")
    sdf.inputs["Voxel Size"].default_value = voxel_size

    position = ng.nodes.new("GeometryNodeInputPosition")
    sample = ng.nodes.new("GeometryNodeSampleGrid")

    store = ng.nodes.new("GeometryNodeStoreNamedAttribute")
    store.data_type = "FLOAT"
    store.domain = "POINT"
    store.inputs["Name"].default_value = attr_name

    ng.links.new(obj_info.outputs["Geometry"], sdf.inputs["Mesh"])
    ng.links.new(sdf.outputs["SDF Grid"], sample.inputs["Grid"])
    ng.links.new(position.outputs["Position"], sample.inputs["Position"])
    ng.links.new(sample.outputs["Value"], store.inputs["Value"])
    ng.links.new(inp.outputs["Geometry"], store.inputs["Geometry"])
    ng.links.new(store.outputs["Geometry"], out.inputs["Geometry"])

    return ng


@auto_layout
def build_normalize_attr_tree(
    input_attr: str = "sdf_distance",
    output_attr: str = "sdf_normalized",
) -> bpy.types.NodeTree:
    """GN tree: clamp attr to positive, normalize to 0-1 by max, store."""
    ng = bpy.data.node_groups.new(f"{_PREFIX}normalize_attr", "GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    inp = ng.nodes.new("NodeGroupInput")
    out = ng.nodes.new("NodeGroupOutput")

    read = ng.nodes.new("GeometryNodeInputNamedAttribute")
    read.data_type = "FLOAT"
    read.inputs["Name"].default_value = input_attr

    clamp = ng.nodes.new("ShaderNodeClamp")
    clamp.inputs["Min"].default_value = 0.0
    clamp.inputs["Max"].default_value = 1e9

    attr_stat = ng.nodes.new("GeometryNodeAttributeStatistic")
    attr_stat.data_type = "FLOAT"

    divide = ng.nodes.new("ShaderNodeMath")
    divide.operation = "DIVIDE"

    store = ng.nodes.new("GeometryNodeStoreNamedAttribute")
    store.data_type = "FLOAT"
    store.domain = "POINT"
    store.inputs["Name"].default_value = output_attr

    ng.links.new(read.outputs["Attribute"], clamp.inputs["Value"])
    ng.links.new(inp.outputs["Geometry"], attr_stat.inputs["Geometry"])
    ng.links.new(clamp.outputs["Result"], attr_stat.inputs["Attribute"])
    ng.links.new(clamp.outputs["Result"], divide.inputs[0])
    ng.links.new(attr_stat.outputs["Max"], divide.inputs[1])
    ng.links.new(divide.outputs["Value"], store.inputs["Value"])
    ng.links.new(inp.outputs["Geometry"], store.inputs["Geometry"])
    ng.links.new(store.outputs["Geometry"], out.inputs["Geometry"])

    return ng


@auto_layout
def build_randomize_weight_tree(
    input_attr: str = "sdf_normalized",
    output_attr: str = "weight",
    seed: int = 0,
) -> bpy.types.NodeTree:
    """GN tree: weight = (1 - normalized) × seeded noise, store as attr."""
    ng = bpy.data.node_groups.new(f"{_PREFIX}randomize_wt", "GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    inp = ng.nodes.new("NodeGroupInput")
    out = ng.nodes.new("NodeGroupOutput")

    read = ng.nodes.new("GeometryNodeInputNamedAttribute")
    read.data_type = "FLOAT"
    read.inputs["Name"].default_value = input_attr

    invert = ng.nodes.new("ShaderNodeMath")
    invert.operation = "SUBTRACT"
    invert.inputs[0].default_value = 1.0

    rand = ng.nodes.new("FunctionNodeRandomValue")
    rand.data_type = "FLOAT"
    rand.inputs["Min"].default_value = 0.0
    rand.inputs["Max"].default_value = 1.0
    rand.inputs["Seed"].default_value = seed

    multiply = ng.nodes.new("ShaderNodeMath")
    multiply.operation = "MULTIPLY"

    store = ng.nodes.new("GeometryNodeStoreNamedAttribute")
    store.data_type = "FLOAT"
    store.domain = "POINT"
    store.inputs["Name"].default_value = output_attr

    ng.links.new(read.outputs["Attribute"], invert.inputs[1])
    ng.links.new(invert.outputs["Value"], multiply.inputs[0])
    ng.links.new(rand.outputs["Value"], multiply.inputs[1])
    ng.links.new(multiply.outputs["Value"], store.inputs["Value"])
    ng.links.new(inp.outputs["Geometry"], store.inputs["Geometry"])
    ng.links.new(store.outputs["Geometry"], out.inputs["Geometry"])

    return ng


@auto_layout
def build_colorize_attr_tree(
    input_attr: str = "weight",
    output_attr: str = "color",
) -> bpy.types.NodeTree:
    """GN tree: map a float attribute through a blue→green→red ramp, store as color."""
    ng = bpy.data.node_groups.new(f"{_PREFIX}colorize", "GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    inp = ng.nodes.new("NodeGroupInput")
    out = ng.nodes.new("NodeGroupOutput")

    read = ng.nodes.new("GeometryNodeInputNamedAttribute")
    read.data_type = "FLOAT"
    read.inputs["Name"].default_value = input_attr

    ramp = ng.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0, 0, 1, 1)
    mid = ramp.color_ramp.elements.new(0.5)
    mid.color = (0, 1, 0, 1)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (1, 0, 0, 1)

    store = ng.nodes.new("GeometryNodeStoreNamedAttribute")
    store.data_type = "FLOAT_COLOR"
    store.domain = "POINT"
    store.inputs["Name"].default_value = output_attr

    ng.links.new(read.outputs["Attribute"], ramp.inputs["Fac"])
    ng.links.new(ramp.outputs["Color"], store.inputs["Value"])
    ng.links.new(inp.outputs["Geometry"], store.inputs["Geometry"])
    ng.links.new(store.outputs["Geometry"], out.inputs["Geometry"])

    return ng


@auto_layout
def build_cull_by_attr_tree(
    attr_name: str = "weight",
    threshold: float = 0.3,
) -> bpy.types.NodeTree:
    """GN tree: delete points where a named float attribute < threshold."""
    ng = bpy.data.node_groups.new(f"{_PREFIX}cull_attr", "GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    inp = ng.nodes.new("NodeGroupInput")
    out = ng.nodes.new("NodeGroupOutput")

    read = ng.nodes.new("GeometryNodeInputNamedAttribute")
    read.data_type = "FLOAT"
    read.inputs["Name"].default_value = attr_name

    compare = ng.nodes.new("FunctionNodeCompare")
    compare.operation = "LESS_THAN"
    compare.inputs[1].default_value = threshold

    delete = ng.nodes.new("GeometryNodeDeleteGeometry")
    delete.domain = "POINT"

    ng.links.new(read.outputs["Attribute"], compare.inputs[0])
    ng.links.new(compare.outputs["Result"], delete.inputs["Selection"])
    ng.links.new(inp.outputs["Geometry"], delete.inputs["Geometry"])
    ng.links.new(delete.outputs["Geometry"], out.inputs["Geometry"])

    return ng


# ---------------------------------------------------------------------------
# Object creation helpers (public API)
# ---------------------------------------------------------------------------

def join_scene_meshes(
    mesh_objects: List[bpy.types.Object],
) -> bpy.types.Object:
    """Join all mesh objects into a single temporary object.

    Applies world transforms so the resulting mesh is in world space.
    """
    import bmesh

    bm = bmesh.new()
    depsgraph = bpy.context.evaluated_depsgraph_get()

    for obj in mesh_objects:
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        temp_bm = bmesh.new()
        temp_bm.from_mesh(mesh)
        temp_bm.transform(obj.matrix_world)

        offset = len(bm.verts)
        for v in temp_bm.verts:
            bm.verts.new(v.co)
        bm.verts.ensure_lookup_table()
        for f in temp_bm.faces:
            try:
                bm.faces.new([bm.verts[offset + v.index] for v in f.verts])
            except ValueError:
                pass

        temp_bm.free()
        eval_obj.to_mesh_clear()

    mesh_data = bpy.data.meshes.new(f"{_PREFIX}joined_mesh")
    bm.to_mesh(mesh_data)
    bm.free()

    obj = bpy.data.objects.new(f"{_PREFIX}joined", mesh_data)
    _link_to_temp_collection(obj)
    logger.info("Joined %d meshes -> '%s' (%d verts)", len(mesh_objects), obj.name, len(mesh_data.vertices))
    return obj


def create_aabb_box(
    bbox_min: Vector,
    bbox_max: Vector,
    padding: float = 0.01,
) -> bpy.types.Object:
    """Create a box mesh at the scene AABB bounds with slight padding."""
    import bmesh

    center = (bbox_min + bbox_max) / 2
    size = bbox_max - bbox_min
    padded_size = Vector((
        size.x * (1.0 + padding),
        size.y * (1.0 + padding),
        size.z * (1.0 + padding),
    ))

    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co.x = v.co.x * padded_size.x + center.x
        v.co.y = v.co.y * padded_size.y + center.y
        v.co.z = v.co.z * padded_size.z + center.z

    mesh_data = bpy.data.meshes.new(f"{_PREFIX}aabb_box")
    bm.to_mesh(mesh_data)
    bm.free()

    obj = bpy.data.objects.new(f"{_PREFIX}aabb_box", mesh_data)
    _link_to_temp_collection(obj)
    logger.info("Created AABB box: center=%s, size=%s", center, padded_size)
    return obj


def scene_aabb(
    mesh_objects: List[bpy.types.Object],
) -> Tuple[Vector, Vector]:
    """Compute the world-space AABB of all mesh objects."""
    all_coords = []
    for obj in mesh_objects:
        all_coords.extend(obj.matrix_world @ Vector(c) for c in obj.bound_box)

    xs = [v.x for v in all_coords]
    ys = [v.y for v in all_coords]
    zs = [v.z for v in all_coords]
    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


# ---------------------------------------------------------------------------
# Evaluation helpers (public API)
# ---------------------------------------------------------------------------

def apply_gn_modifier(
    obj: bpy.types.Object,
    node_tree: bpy.types.NodeTree,
    name: str = "GN",
) -> None:
    """Apply a Geometry Nodes modifier to an object and force evaluation."""
    mod = obj.modifiers.new(name=name, type="NODES")
    mod.node_group = node_tree
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj.evaluated_get(depsgraph)


def evaluated_mesh_copy(
    obj: bpy.types.Object,
    name: str = "",
) -> bpy.types.Object:
    """Create a new object from the evaluated (modifier-applied) mesh."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)

    suffix = name or "eval"
    new_mesh = bpy.data.meshes.new(f"{_PREFIX}{suffix}_mesh")
    new_mesh.from_pydata(
        [v.co.copy() for v in eval_obj.data.vertices],
        [],
        [list(p.vertices) for p in eval_obj.data.polygons],
    )
    new_mesh.update()

    new_obj = bpy.data.objects.new(f"{_PREFIX}{suffix}", new_mesh)
    _link_to_temp_collection(new_obj)

    logger.debug(
        "Copied evaluated mesh -> '%s' (%d verts, %d faces)",
        new_obj.name, len(new_mesh.vertices), len(new_mesh.polygons),
    )
    return new_obj


def read_points_from_evaluated(
    obj: bpy.types.Object,
) -> List[Tuple[float, float, float]]:
    """Read point positions from an evaluated GN output (point cloud or mesh)."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    data = eval_obj.data

    points = []
    if data is None:
        return points

    if hasattr(data, "vertices") and len(data.vertices) > 0:
        for v in data.vertices:
            points.append((v.co.x, v.co.y, v.co.z))
    elif hasattr(data, "attributes") and "position" in data.attributes:
        pos_attr = data.attributes["position"]
        for i in range(len(pos_attr.data)):
            v = pos_attr.data[i].vector
            points.append((v[0], v[1], v[2]))

    return points


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup_temp_data() -> None:
    """Remove all temp objects, node groups, and the temp collection."""
    for obj in list(bpy.data.objects):
        if obj.name.startswith(_PREFIX):
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if data and data.users == 0:
                if isinstance(data, bpy.types.Mesh):
                    bpy.data.meshes.remove(data)
                elif isinstance(data, bpy.types.Volume):
                    bpy.data.volumes.remove(data)

    for ng in list(bpy.data.node_groups):
        if ng.name.startswith(_PREFIX):
            bpy.data.node_groups.remove(ng)

    if _COLLECTION_NAME in bpy.data.collections:
        col = bpy.data.collections[_COLLECTION_NAME]
        bpy.data.collections.remove(col)

    logger.debug("Cleaned up all %s temp data", _PREFIX)
