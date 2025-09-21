from __future__ import annotations
from typing import Literal, Optional, Dict
from math import sqrt
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from sklearn.neighbors import BallTree
from sklearn.neighbors import KernelDensity

# ---- small geometry fallbacks (match your loader’s robustness) ----------------
try:
    import shapely
    from shapely.geometry import Point, Polygon
    _HAS_SHAPELY = True
except Exception:
    shapely = None
    Point = Polygon = None
    _HAS_SHAPELY = False

def _union_polygon(gdf: gpd.GeoDataFrame):
    """GeoSeries.unary_union → shapely.union_all → dissolve."""
    try:
        return gdf.geometry.unary_union
    except Exception:
        pass
    if _HAS_SHAPELY:
        try:
            return shapely.union_all(gdf.geometry.values)
        except Exception:
            pass
    try:
        return gdf.dissolve().geometry.iloc[0]
    except Exception as e:
        raise RuntimeError(f"Failed to union study area: {e}")

def _clip_points_to_poly(points: gpd.GeoDataFrame, poly) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = poly.bounds
    bbox = points.cx[minx:maxx, miny:maxy]
    return bbox[bbox.geometry.intersects(poly)]

def _pick_geom_col(df):
    candidates = ["geom", "geometry", "Geometry", "GEOMETRY", "WKT", "wkt", "wkt_geom"]
    lowmap = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowmap:
            return lowmap[cand.lower()]
    return None
    
# ---- grids / hexes ------------------------------------------------------------
def _make_square_grid(poly, cell_size_m: float) -> gpd.GeoDataFrame:
    """Square grid covering polygon extent; clipped to polygon; returns polygons with 'unit_id'."""
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(minx, maxx + cell_size_m, cell_size_m)
    ys = np.arange(miny, maxy + cell_size_m, cell_size_m)

    polys = []
    ids = []
    uid = 0
    for x in xs[:-1]:
        for y in ys[:-1]:
            p = Polygon([(x, y),
                         (x + cell_size_m, y),
                         (x + cell_size_m, y + cell_size_m),
                         (x, y + cell_size_m)])
            polys.append(p)
            ids.append(uid)
            uid += 1
    grid = gpd.GeoDataFrame({"unit_id": ids}, geometry=polys, crs="EPSG:27700")
    # clip to study area
    grid = gpd.overlay(grid, gpd.GeoDataFrame(geometry=[poly], crs=grid.crs),
                       how="intersection")
    # keep sequential ids post-clip
    grid = grid.reset_index(drop=True)
    grid["unit_id"] = np.arange(len(grid))
    return grid

def _make_hex_grid(poly, edge_km: float) -> gpd.GeoDataFrame:
    """Hex grid covering polygon extent; edge length in KM; clipped; returns polygons with 'unit_id'."""
    a = edge_km * 1000.0  # edge in meters
    w = 2 * a                     # hex width
    h = sqrt(3) * a               # hex height
    x_step = 3/2 * a              # horizontal step
    y_step = h                    # vertical step

    minx, miny, maxx, maxy = poly.bounds

    polys = []
    ids = []
    uid = 0
    # Row by row with horizontal offset every other row
    j = 0
    y = miny - y_step
    while y <= maxy + y_step:
        x_offset = 0 if (j % 2 == 0) else (0.75 * w)  # 0.75w == 3/2 * a offset / 2
        x = (minx - w) + x_offset
        while x <= maxx + w:
            # regular pointy-top hex centered at (x, y)
            cx, cy = x, y
            # vertices (pointy-top) around center using edge a
            verts = [
                (cx + 0,     cy + a),
                (cx + (sqrt(3)/2)*a, cy + a/2),
                (cx + (sqrt(3)/2)*a, cy - a/2),
                (cx + 0,     cy - a),
                (cx - (sqrt(3)/2)*a, cy - a/2),
                (cx - (sqrt(3)/2)*a, cy + a/2),
            ]
            p = Polygon(verts)
            polys.append(p)
            ids.append(uid)
            uid += 1
            x += x_step
        y += y_step
        j += 1

    hexes = gpd.GeoDataFrame({"unit_id": ids}, geometry=polys, crs="EPSG:27700")
    hexes = gpd.overlay(hexes, gpd.GeoDataFrame(geometry=[poly], crs=hexes.crs),
                        how="intersection")
    hexes = hexes.reset_index(drop=True)
    hexes["unit_id"] = np.arange(len(hexes))
    return hexes

# ---- KDE helper ---------------------------------------------------------------
def _fit_kde(points_gdf: gpd.GeoDataFrame, bandwidth_m: float):
    """Fit sklearn KDE on x,y coordinates (assumes projected meters)."""
    if points_gdf is None or points_gdf.empty:
        return None
    coords = np.column_stack([points_gdf.geometry.x.values, points_gdf.geometry.y.values])
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth_m).fit(coords)
    return kde

def _score_kde(kde, sample_points_gdf: gpd.GeoDataFrame, out_col: str):
    """Evaluate KDE at sample point locations; add exp(score) into column."""
    if kde is None or sample_points_gdf is None or sample_points_gdf.empty:
        sample_points_gdf[out_col] = 0.0
        return sample_points_gdf
    samples = np.column_stack([sample_points_gdf.geometry.x.values,
                               sample_points_gdf.geometry.y.values])
    sample_points_gdf[out_col] = np.exp(kde.score_samples(samples))
    return sample_points_gdf

# ---- main feature engineering -------------------------------------------------
def engineer_model_dataset(
    loader_out: Dict[str, Optional[gpd.GeoDataFrame]],
    unit_type: Literal["grid", "hex"] = "grid",
    grid_size_m: float = 60.0,
    hex_edge_km: float = 0.075863783,
    crime_bandwidth_m: float = 300.0,
    poi_mode: Literal["both", "kde", "distance", "none"] = "both",
    poi_bandwidth_m: float = 300.0,
    poi_category_col: str = "Category"
) -> Dict[str, gpd.GeoDataFrame]:
    """
    Build regression-ready data from loaded study area, crime, and POI.
    Returns dict: {"units": polygons, "spine": centroids, "model": spine-with-features}

    Assumes CRS is a metric projection (EPSG:27700 by default in your loader).
    """
    study_area = loader_out.get("study_area")
    crime = loader_out.get("crime_data")
    poi = loader_out.get("points_of_interest")

    if study_area is None or study_area.empty:
        raise ValueError("study_area is empty or missing.")

    # 1) Build unit polygons
    poly = _union_polygon(study_area)
    if unit_type == "grid":
        units = _make_square_grid(poly, grid_size_m)
    elif unit_type == "hex":
        units = _make_hex_grid(poly, hex_edge_km)
    else:
        raise ValueError("unit_type must be 'grid' or 'hex'.")

    # 2) Spine: centroids as point units (retain polygon link by unit_id)
    spine = units[["unit_id"]].copy()
    spine = gpd.GeoDataFrame(spine, geometry=units.geometry.centroid, crs=units.crs)

    # 3) Outcome: crime_count via spatial join (count points in polygon)
    if crime is not None and not crime.empty:
        # Restrict crimes to study poly (cheap then exact)
        crime_clip = _clip_points_to_poly(crime, poly)
        joined = gpd.sjoin(crime_clip[["geometry"]], units[["unit_id", "geometry"]],
                           how="left", predicate="within")
        counts = joined.groupby("unit_id", dropna=False).size().rename("crime_count").reset_index()
        spine = spine.merge(counts, on="unit_id", how="left")
    else:
        spine["crime_count"] = 0

    spine["crime_count"] = spine["crime_count"].fillna(0).astype(int)

    # 4) Outcome: crime KDE at centroids
    kde_crime = _fit_kde(crime, crime_bandwidth_m) if (crime is not None and not crime.empty) else None
    spine = _score_kde(kde_crime, spine, "crime_kde")

    # 5) POI features (per category)
    if poi_mode != "none" and poi is not None and not poi.empty:
        # keep only POIs in study area
        poi_clip = _clip_points_to_poly(poi, poly)

        if poi_category_col not in poi_clip.columns:
            raise ValueError(f"POI category column '{poi_category_col}' not found in POI data.")

        categories = [c for c in poi_clip[poi_category_col].astype(str).unique() if pd.notna(c)]

        for cat in categories:
            subset = poi_clip[poi_clip[poi_category_col].astype(str) == str(cat)]

            # Distance to nearest instance of this category
            if poi_mode in ("both", "distance"):
                if subset.empty:
                    spine[f"Dist_{cat}"] = np.nan
                else:
                    # union then distance (fast and robust)
                    u = _union_polygon(subset)
                    spine[f"Dist_{cat}"] = spine.geometry.distance(u)

            # KDE density of this category
            if poi_mode in ("both", "kde"):
                kde = _fit_kde(subset, poi_bandwidth_m)
                col = f"KDE_{cat}"
                spine = _score_kde(kde, spine, col)

    # 6) Final model table: spine (points) with features; polygons available via unit_id
    model = spine.copy()

    return {"units": units, "spine": spine, "model": model}

# --- for some demographic features ---
def add_kde_features(
    features: dict,
    source_path: str,
    value_cols=("resident_pop","income_deprivation","employment_deprivation",
                "youth_workday_pop","workers_workday_pop","elderly_workday_pop"),
    bandwidth_m=400.0,
    layer=None,
    normalize=True,      # if True, applies 2D Gaussian normalization (1/(2πh²))
    clip_to_units=True,  # if True, restricts census points to features["units"]
    radius_factor=3.0    # search radius = radius_factor * bandwidth
):
    """
    Adds KDE_* columns to features["model"] by kernel-smoothing weighted census centroids.

    Parameters
    ----------
    features : dict with keys:
        - "model": GeoDataFrame (points)
        - "units": GeoDataFrame (polygons), same CRS as model (or reprojectable)
    source_path : str
        Path to CSV (expects 'wkt_geom') or GeoPackage (geometries stored natively).
    value_cols : iterable of str
        Columns in source to use as weights.
    bandwidth_m : float
        Spatial bandwidth in metres.
    layer : str or None
        GeoPackage layer name (ignored for CSV).
    normalize : bool
        If True, multiplies kernel by 1/(2πh²) yielding density per m².
    clip_to_units : bool
        If True, clips source points to the union of features["units"].
    radius_factor : float
        Multiplier for kernel search radius; 3≈99.7% mass for Gaussian.
    """

    # --- 1) Validate inputs
    if "model" not in features or "units" not in features:
        raise ValueError("features must contain 'model' (points) and 'units' (polygons).")

    gdf_model = features["model"].copy()
    gdf_units = features["units"].copy()

    if gdf_model.crs is None or gdf_units.crs is None:
        raise ValueError("Both model and units must have a projected CRS in metres.")

    # Align CRS
    if gdf_model.crs != gdf_units.crs:
        gdf_units = gdf_units.to_crs(gdf_model.crs)

    # Must be projected in metres (heuristic check)
    if gdf_model.crs.is_geographic:
        raise ValueError("Model CRS is geographic. Reproject to a metric CRS (e.g., EPSG:27700 or 3857).")

    # --- 2) Read source points (CSV or GPKG) and align CRS
    if source_path.lower().endswith(".csv"):
        df_src = pd.read_csv(source_path)
        geom_col = _pick_geom_col(df_src)
        if not geom_col:
            raise ValueError("CSV must contain a geometry column named one of: "
                             "'geom', 'Geometry', 'geometry', 'WKT', or 'wkt_geom'.")
    
        # Parse WKT if strings; accept shapely objects if present
        series = df_src[geom_col]
        if pd.api.types.is_string_dtype(series):
            geom = gpd.GeoSeries.from_wkt(series)
        else:
            # Try treating as shapely geometries
            non_null = series.dropna()
            if len(non_null) and hasattr(non_null.iloc[0], "geom_type"):
                geom = gpd.GeoSeries(series)
            else:
                raise ValueError(f"Geometry column '{geom_col}' is not WKT or shapely geometries.")
    
        gdf_src = gpd.GeoDataFrame(df_src.drop(columns=[geom_col]), geometry=geom, crs=gdf_model.crs)

    elif source_path.lower().endswith(".gpkg"):
        gdf_src = gpd.read_file(source_path, layer=layer) if layer else gpd.read_file(source_path)
    
        # Ensure a geometry column is active; fallback to common names if needed
        try:
            _ = gdf_src.geometry  # access to confirm set
            geom_is_set = gdf_src._geometry_column_name in gdf_src.columns
        except Exception:
            geom_is_set = False
    
        if not geom_is_set:
            geom_col = _pick_geom_col(gdf_src)
            if not geom_col:
                raise ValueError("GeoPackage has no active geometry and no column named "
                                 "'geom', 'Geometry', 'geometry', 'WKT', or 'wkt_geom'.")
            gdf_src = gdf_src.set_geometry(geom_col)
    
        # Align CRS to model
        if gdf_src.crs != gdf_model.crs:
            gdf_src = gdf_src.to_crs(gdf_model.crs)
    
    else:
        raise ValueError("source_path must be a .csv or .gpkg")

    # --- 3) Clip to units if requested
    if clip_to_units:
        ua = gdf_units.unary_union
        gdf_src = gdf_src[gdf_src.geometry.within(ua)].copy()

    if gdf_src.empty:
        # Create zero-valued KDE columns and return early
        for c in value_cols:
            gdf_model[f'KDE_{c}'] = 0.0
        features["model"] = gdf_model
        return features

    # --- 4) Build spatial index (BallTree on Euclidean coordinates)
    # Extract coordinates
    X_src = np.vstack([gdf_src.geometry.x.values, gdf_src.geometry.y.values]).T
    X_tgt = np.vstack([gdf_model.geometry.x.values, gdf_model.geometry.y.values]).T

    # BallTree requires a metric; use 'euclidean'
    tree = BallTree(X_src, metric='euclidean')

    # Query neighbors within radius
    radius = radius_factor * bandwidth_m
    ind_list = tree.query_radius(X_tgt, r=radius, return_distance=False)

    # Kernel constants
    h = float(bandwidth_m)
    inv_h2 = 1.0 / (h * h)
    norm = (1.0 / (2.0 * np.pi * h * h)) if normalize else 1.0

    # Pre-fetch weights as array
    W = gdf_src[list(value_cols)].to_numpy(dtype=float)

    # --- 5) Compute KDE for each target and each column
    kde_arrays = {c: np.zeros(len(gdf_model), dtype=float) for c in value_cols}

    # To avoid recomputing distances for each column, do per-target once
    for i, nbr_idx in enumerate(ind_list):
        if nbr_idx.size == 0:
            continue
        # Distances from target i to its neighbors
        dx = X_src[nbr_idx, 0] - X_tgt[i, 0]
        dy = X_src[nbr_idx, 1] - X_tgt[i, 1]
        d2 = dx*dx + dy*dy
        # Gaussian kernel weights: exp(-d^2 / (2h^2))
        k = np.exp(-0.5 * d2 * inv_h2) * norm
        # Accumulate per attribute
        for j, col in enumerate(value_cols):
            # Weighted sum of kernels
            kde_arrays[col][i] = np.dot(W[nbr_idx, j], k)

    # --- 6) Attach to model with KDE_ prefix
    for col in value_cols:
        gdf_model[f'KDE_{col}'] = kde_arrays[col]

    features["model"] = gdf_model
    return features
