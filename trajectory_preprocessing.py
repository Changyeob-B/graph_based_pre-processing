"""Graph conversion-based pre-processing of active travel GNSS trajectories."""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, MultiLineString
from shapely.geometry.base import BaseGeometry


INPUT_FILE = Path(r"input.gpkg")
OUTPUT_FILE = Path(r"output.gpkg")
INPUT_LAYER = None
OUTPUT_LAYER = "processed_segments"

PRECISION_METERS = 30.0  # Spatial tolerance (alpha)
DANGLE_THRESHOLD = 2  # First-stage threshold (beta)
ISO_THRESHOLD = 9  # Second-stage threshold (gamma)
REMOVE_DANGLES = True
REMOVE_ISOLATED = True

USE_DISTANCE_SEGMENTATION = False
SEGMENT_LENGTH_METERS = 5.0
REMOVE_SNAPPED_ZERO_LENGTH_SEGMENTS = True


GridNode = Tuple[int, int]
Segment = Tuple[Tuple[float, float], Tuple[float, float]]
SegmentIndex = Dict[int, List[int]]


def round_pt_xy(x: float, y: float, prec_m: float) -> GridNode:
    """Map a coordinate pair to the nearest grid node."""
    return int(round(x / prec_m)), int(round(y / prec_m))


def iter_lines(geom: BaseGeometry) -> Iterable[LineString]:
    """Yield each line component of a trajectory."""
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, MultiLineString):
        for part in geom.geoms:
            if isinstance(part, LineString):
                yield part


def iter_original_segments(ls: LineString) -> Iterable[Segment]:
    """Yield segments between consecutive distinct vertices."""
    coords = list(ls.coords)
    for i in range(len(coords) - 1):
        p0 = (coords[i][0], coords[i][1])
        p1 = (coords[i + 1][0], coords[i + 1][1])
        if p0 != p1:
            yield p0, p1


def iter_distance_segments(ls: LineString, segment_length_m: float) -> Iterable[Segment]:
    """Yield segments between points sampled at fixed distances along the line."""
    if ls is None or ls.is_empty:
        return
    if segment_length_m <= 0:
        raise ValueError("The segment length must be greater than zero.")

    total_length = ls.length
    if total_length == 0:
        return

    start_dist = 0.0
    while start_dist < total_length:
        end_dist = min(start_dist + segment_length_m, total_length)
        p0 = ls.interpolate(start_dist)
        p1 = ls.interpolate(end_dist)
        a = (p0.x, p0.y)
        b = (p1.x, p1.y)
        if a != b:
            yield a, b
        start_dist = end_dist


def iter_segments(
    ls: LineString,
    use_distance_segmentation: bool,
    segment_length_m: float,
) -> Iterable[Segment]:
    """Yield segments using the selected segmentation method."""
    if use_distance_segmentation:
        yield from iter_distance_segments(ls, segment_length_m)
    else:
        yield from iter_original_segments(ls)


def build_graph_from_segments(
    gdf: gpd.GeoDataFrame,
    prec_m: float,
    use_distance_segmentation: bool,
    segment_length_m: float,
    remove_snapped_zero_length_segments: bool = True,
) -> Tuple[nx.MultiGraph, SegmentIndex]:
    """Build a multigraph and index registered segments by trajectory.

    Segments whose endpoints map to the same node are skipped by default.
    """
    G = nx.MultiGraph()
    fid_to_seg_idxs: SegmentIndex = {}

    for fid, geom in enumerate(gdf.geometry):
        seg_idx = 0
        for ls in iter_lines(geom):
            for (x0, y0), (x1, y1) in iter_segments(
                ls, use_distance_segmentation, segment_length_m
            ):
                a = round_pt_xy(x0, y0, prec_m)
                b = round_pt_xy(x1, y1, prec_m)
                if a == b and remove_snapped_zero_length_segments:
                    continue
                G.add_edge(a, b, fid=fid, seg_idx=seg_idx)
                fid_to_seg_idxs.setdefault(fid, []).append(seg_idx)
                seg_idx += 1

    return G, fid_to_seg_idxs


def remove_trajectories_by_beta(
    G: nx.MultiGraph,
    threshold: int,
) -> Set[int]:
    """Remove trajectories whose registered endpoints all have degree at most beta."""
    degrees = dict(G.degree())
    retained_ids = {
        data["fid"]
        for u, v, data in G.edges(data=True)
        if degrees[u] > threshold or degrees[v] > threshold
    }
    edges_to_remove = [
        (u, v, key)
        for u, v, key, data in G.edges(keys=True, data=True)
        if data["fid"] not in retained_ids
    ]
    G.remove_edges_from(edges_to_remove)
    return retained_ids


def filter_gdf_by_survivors(gdf: gpd.GeoDataFrame, survivors: Set[int]) -> gpd.GeoDataFrame:
    """Return the original features for the retained trajectory IDs."""
    if not survivors:
        return gdf.iloc[[]].copy()
    return gdf.iloc[sorted(survivors)].reset_index(drop=True)


def rejected_trajectories_by_gamma(G: nx.MultiGraph, threshold: int) -> Set[int]:
    """Identify trajectories with any registered endpoint of degree at most gamma."""
    degrees = dict(G.degree())
    return {
        data["fid"]
        for u, v, data in G.edges(data=True)
        if degrees[u] <= threshold or degrees[v] <= threshold
    }


def simplify_network(
    in_path: Path,
    out_path: Path,
    *,
    in_layer: Optional[str] = None,
    out_layer: Optional[str] = "processed_segments",
    remove_dangles_flag: bool = False,
    remove_isolated_flag: bool = False,
    precision_meters: float = 15.0,
    use_distance_segmentation: bool = False,
    segment_length_meters: float = 5.0,
    remove_snapped_zero_length_segments: bool = True,
    dangle_threshold: int = 1,
    iso_threshold: int = 1,
) -> gpd.GeoDataFrame:
    """Filter trajectories using graph connectivity and save the retained features."""
    if precision_meters <= 0:
        raise ValueError("The grid size must be greater than zero.")
    if use_distance_segmentation and segment_length_meters <= 0:
        raise ValueError("The segment length must be greater than zero when distance segmentation is enabled.")

    if Path(in_path).resolve() == Path(out_path).resolve():
        raise ValueError("The input and output paths must be different.")

    gdf = gpd.read_file(in_path, layer=in_layer, on_invalid="ignore").reset_index(drop=True)
    if gdf.empty:
        raise ValueError("The input contains no features.")

    original_count = len(gdf)
    null_count = int(gdf.geometry.isna().sum())
    gdf = gdf[gdf.geometry.notna()].copy()
    empty_count = int(gdf.geometry.is_empty.sum())
    gdf = gdf[~gdf.geometry.is_empty].copy()
    invalid_count = int((~gdf.geometry.is_valid).sum())
    gdf = gdf[gdf.geometry.is_valid].copy()
    removed_count = original_count - len(gdf)

    if removed_count:
        print(
            f"[read] invalid geometries removed: {removed_count} "
            f"(null={null_count}, empty={empty_count}, invalid={invalid_count})"
        )
    if gdf.empty:
        raise ValueError("The input contains no valid geometries.")
    if gdf.crs is None:
        raise ValueError("The input has no CRS. Assign a projected CRS with metre units.")
    if gdf.crs.is_geographic:
        raise ValueError("The input uses a geographic CRS. Reproject it to a projected CRS with metre units.")

    line_mask = gdf.geometry.apply(lambda geom: isinstance(geom, (LineString, MultiLineString)))
    if not line_mask.any():
        raise ValueError("The input contains no line geometries.")
    gdf = gdf.loc[line_mask].reset_index(drop=True)

    mode = "distance" if use_distance_segmentation else "original"
    print(f"[mode] {mode}" + (f" (interval={segment_length_meters}m)" if use_distance_segmentation else ""))
    print(f"[snap] remove_collapsed_segments={remove_snapped_zero_length_segments}")

    G, fid_to_seg_idxs = build_graph_from_segments(
        gdf,
        prec_m=precision_meters,
        use_distance_segmentation=use_distance_segmentation,
        segment_length_m=segment_length_meters,
        remove_snapped_zero_length_segments=remove_snapped_zero_length_segments,
    )
    print(f"[build] features={len(gdf)}, edges(segments)={G.number_of_edges()}, nodes={G.number_of_nodes()}")

    if remove_dangles_flag or remove_isolated_flag:
        surviving_ids = set(fid_to_seg_idxs)
        if remove_dangles_flag:
            surviving_ids = remove_trajectories_by_beta(G, dangle_threshold)
            print(f"[beta] survivors(features)={len(surviving_ids)}")
        if remove_isolated_flag:
            surviving_ids -= rejected_trajectories_by_gamma(G, iso_threshold)
            print(f"[gamma] survivors(features)={len(surviving_ids)}")
        gdf = filter_gdf_by_survivors(gdf, surviving_ids)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".gpkg":
        gdf.to_file(out_path, layer=out_layer if out_layer else "layer", driver="GPKG")
    else:
        gdf.to_file(out_path)

    print(f"[saved] {out_path} (features={len(gdf)})")
    return gdf


if __name__ == "__main__":
    simplify_network(
        INPUT_FILE,
        OUTPUT_FILE,
        in_layer=INPUT_LAYER,
        out_layer=OUTPUT_LAYER,
        remove_dangles_flag=REMOVE_DANGLES,
        remove_isolated_flag=REMOVE_ISOLATED,
        precision_meters=PRECISION_METERS,
        use_distance_segmentation=USE_DISTANCE_SEGMENTATION,
        segment_length_meters=SEGMENT_LENGTH_METERS,
        remove_snapped_zero_length_segments=REMOVE_SNAPPED_ZERO_LENGTH_SEGMENTS,
        dangle_threshold=DANGLE_THRESHOLD,
        iso_threshold=ISO_THRESHOLD,
    )
