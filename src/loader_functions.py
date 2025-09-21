from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict
import pandas as pd
import geopandas as gpd
import fiona
import matplotlib.pyplot as plt

# shapely may vary across environments; import loosely
try:
    import shapely
    from shapely.geometry import Point
    _HAS_SHAPELY = True
except Exception:  # very rare
    shapely = None
    Point = None
    _HAS_SHAPELY = False

DEFAULT_CRS = "EPSG:27700"

# ---------------- helpers ----------------

def _ensure_crs(gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs(target_crs, allow_override=True)
    elif str(gdf.crs).upper() != target_crs.upper():
        gdf = gdf.to_crs(target_crs)
    return gdf

def _apply_filter(gdf: gpd.GeoDataFrame, col: Optional[str], vals: Optional[Sequence]):
    if col is None or vals is None:
        return gdf
    if not isinstance(vals, (list, tuple, set)):
        vals = [vals]
    return gdf[gdf[col].isin(vals)]

def _read_vector(path: str, layer: Optional[str] = None) -> gpd.GeoDataFrame:
    path = str(path)
    suffix = Path(path).suffix.lower()

    if suffix in {".gpkg", ".geopackage"}:
        if layer is None:
            layers = fiona.listlayers(path)
            if not layers:
                raise ValueError(f"No layers found in {path}")
            layer = layers[0]  # default: first layer
        return gpd.read_file(path, layer=layer)

    # shapefile, geoparquet, geojson, etc.
    return gpd.read_file(path)

def _detect_xy_cols(df: pd.DataFrame):
    candidates = [
        ("x", "y"), ("X", "Y"),
        ("longitude", "latitude"), ("Longitude", "Latitude"),
        ("lon", "lat"), ("Lon", "Lat"),
        ("easting", "northing"), ("Easting", "Northing"), ("EASTING", "NORTHING"),
        ("xcoord", "ycoord"), ("XCoord", "YCoord"), ("X_COORD", "Y_COORD"),
    ]
    for x, y in candidates:
        if x in df.columns and y in df.columns:
            return x, y
    return None

def _points_from_xy(df: pd.DataFrame, x_col: str, y_col: str, crs: str) -> gpd.GeoDataFrame:
    # 1) try shapely.Point (fast)
    if _HAS_SHAPELY and Point is not None:
        try:
            geom = [Point(xy) for xy in zip(df[x_col], df[y_col])]
            return gpd.GeoDataFrame(df, geometry=geom, crs=crs)
        except Exception:
            pass
    # 2) fallback: geopandas helper
    geom = gpd.points_from_xy(df[x_col], df[y_col])
    return gpd.GeoDataFrame(df, geometry=geom, crs=crs)

def _read_points_csv(
    path: str,
    crs: str = DEFAULT_CRS,
    wkt_col: Optional[str] = None,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
) -> gpd.GeoDataFrame:
    df = pd.read_csv(path)
    # explicit WKT column
    if wkt_col and wkt_col in df.columns:
        from shapely import wkt as _wkt
        geom = df[wkt_col].apply(_wkt.loads)
        return gpd.GeoDataFrame(df.drop(columns=[wkt_col]), geometry=geom, crs=crs)
    # common WKT names
    for cand in ("wkt", "WKT", "geometry", "Geometry", "geom", "GEOMETRY"):
        if cand in df.columns:
            from shapely import wkt as _wkt
            geom = df[cand].apply(_wkt.loads)
            return gpd.GeoDataFrame(df.drop(columns=[cand]), geometry=geom, crs=crs)
    # XY route
    if not (x_col and y_col):
        xy = _detect_xy_cols(df)
        if xy:
            x_col, y_col = xy
    if not (x_col and y_col):
        raise ValueError("CSV points need a WKT/Geometry column or X/Y columns.")
    return _points_from_xy(df, x_col, y_col, crs)

def _read_points_any(
    path: str,
    layer: Optional[str],
    crs: str,
    wkt_col: Optional[str],
    x_col: Optional[str],
    y_col: Optional[str],
) -> gpd.GeoDataFrame:
    ext = Path(path).suffix.lower()
    gdf = _read_points_csv(path, crs, wkt_col, x_col, y_col) if ext == ".csv" else _read_vector(path, layer)
    return _ensure_crs(gdf, crs)

def _union_polygon(gdf: gpd.GeoDataFrame):
    """Robust union with multiple fallbacks."""
    # 1) GeoSeries.unary_union
    try:
        return gdf.geometry.unary_union
    except Exception:
        pass
    # 2) shapely.union_all (Shapely 2.x)
    if _HAS_SHAPELY:
        try:
            return shapely.union_all(gdf.geometry.values)
        except Exception:
            pass
    # 3) dissolve (old but reliable)
    try:
        return gdf.dissolve().geometry.iloc[0]
    except Exception as e:
        raise RuntimeError(f"Failed to build study area union: {e}")

def _clip_points_to_poly(points: gpd.GeoDataFrame, poly) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = poly.bounds
    bbox = points.cx[minx:maxx, miny:maxy]
    return bbox[bbox.geometry.intersects(poly)]

def _clip_any_to_poly(gdf: gpd.GeoDataFrame, poly) -> gpd.GeoDataFrame:
    if gdf.geom_type.isin(["Point", "MultiPoint"]).all():
        return _clip_points_to_poly(gdf, poly)
    return gpd.clip(gdf, gpd.GeoSeries([poly], crs=gdf.crs))

# load spatial data

def load_spatial_data(
    # Study area
    study_path: str,
    study_layer: Optional[str] = None,
    study_filter: Optional[Tuple[str, Sequence]] = None,  # ("Borough", ["Middlesbrough","Stockton-on-Tees"])
    study_crs: str = DEFAULT_CRS,
    # Crime points (optional)
    crime_path: Optional[str] = None,
    crime_layer: Optional[str] = None,
    crime_filter: Optional[Tuple[str, Sequence]] = None,  # ("Crime Type", ["Violence","Robbery"])
    crime_wkt_col: Optional[str] = None,
    crime_x_col: Optional[str] = None,
    crime_y_col: Optional[str] = None,
    # POI points (optional)
    poi_path: Optional[str] = None,
    poi_layer: Optional[str] = None,
    poi_filter: Optional[Tuple[str, Sequence]] = None,    # ("Category", ["Supermarket","ATM","Bank"])
    poi_wkt_col: Optional[str] = None,
    poi_x_col: Optional[str] = None,
    poi_y_col: Optional[str] = None,
    # General
    target_crs: str = DEFAULT_CRS,
) -> Dict[str, Optional[gpd.GeoDataFrame]]:
    """
    Returns dict with:
      - 'study_area': single-row GeoDataFrame (union polygon) in target_crs
      - 'crime_data': points intersecting study area (or None)
      - 'points_of_interest': points intersecting study area (or None)
    """
    # Study area
    study = _read_vector(study_path, layer=study_layer)
    study = _ensure_crs(study, study_crs)
    if study_crs.upper() != target_crs.upper():
        study = study.to_crs(target_crs)
    if study_filter:
        col, vals = study_filter
        study = _apply_filter(study, col, vals)
        if study.empty:
            raise ValueError("Study filter returned no features.")
    poly = _union_polygon(study)
    study_area = gpd.GeoDataFrame({"name": ["study_area"]}, geometry=[poly], crs=target_crs)

    # Crimes
    crime_gdf = None
    if crime_path:
        crime_gdf = _read_points_any(crime_path, crime_layer, target_crs, crime_wkt_col, crime_x_col, crime_y_col)
        if crime_filter:
            ccol, cvals = crime_filter
            crime_gdf = _apply_filter(crime_gdf, ccol, cvals)
        if not crime_gdf.empty:
            crime_gdf = _clip_points_to_poly(crime_gdf, poly)

    # POIs
    poi_gdf = None
    if poi_path:
        poi_gdf = _read_points_any(poi_path, poi_layer, target_crs, poi_wkt_col, poi_x_col, poi_y_col)
        if poi_filter:
            pcol, pvals = poi_filter
            poi_gdf = _apply_filter(poi_gdf, pcol, pvals)
        if not poi_gdf.empty:
            poi_gdf = _clip_points_to_poly(poi_gdf, poly)

    return {"study_area": study_area, "crime_data": crime_gdf, "points_of_interest": poi_gdf}

# check data load

def check_data_load(data_dict, figsize=(8, 8)):
    """
    Quick diagnostic plot for study area, crime points, and POIs.
    """
    study_area = data_dict.get("study_area")
    crime = data_dict.get("crime_data")
    poi = data_dict.get("points_of_interest")

    fig, ax = plt.subplots(figsize=figsize)

    if study_area is not None and not study_area.empty:
        study_area.boundary.plot(ax=ax, color="black", linewidth=1, label="Study Area")

    if crime is not None and not crime.empty:
        crime.plot(ax=ax, color="red", markersize=5, alpha=0.6,
                   marker="o", label="Crimes")

    if poi is not None and not poi.empty:
        poi.plot(ax=ax, color="blue", markersize=5, alpha=0.6,
                 marker="^", label="POIs")

    ax.set_title("Check Data Load", fontsize=14)
    ax.set_axis_off()
    ax.legend()

    plt.show()