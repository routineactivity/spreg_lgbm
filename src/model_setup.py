from __future__ import annotations
from typing import Literal, Optional, Dict
from math import sqrt
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from sklearn.neighbors import BallTree
from sklearn.neighbors import KernelDensity

# geometry fallbacks
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
    
# grids / hexes
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
            # vertices (pointy-top) around centre using edge a
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

# KDE helper
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

# main feature engineering
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

# Adding some demographic features
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.signal import fftconvolve

# helper: detect geometry column in CSV/DataFrame
def _pick_geom_col(df):
    candidates = ["geom", "geometry", "Geometry", "GEOMETRY", "WKT", "wkt", "wkt_geom"]
    lowmap = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowmap:
            return lowmap[cand.lower()]
    return None

# helper: quartic (biweight) kernel, normalised so sum(k)*cell_area ≈ 1 (intensity per m²)
def _quartic_kernel(radius_cells, cell_size):
    R = int(np.ceil(radius_cells))
    y, x = np.ogrid[-R:R+1, -R:R+1]
    d = np.sqrt(x*x + y*y)
    k = (1.0 - (d / R)**2)
    k[k < 0] = 0.0
    k = (k**2).astype(np.float32)
    # Normalise to intensity per m²
    k /= (k.sum() * (cell_size * cell_size))
    return k

# helper: bilinear sampler from float pixel coords
def _bilinear(arr, xf, yf):
    h, w = arr.shape
    x0 = np.floor(xf).astype(int); y0 = np.floor(yf).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1); y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1);     y0 = np.clip(y0, 0, h - 1)
    wx = xf - x0; wy = yf - y0
    v00 = arr[y0, x0]; v10 = arr[y0, x1]
    v01 = arr[y1, x0]; v11 = arr[y1, x1]
    return (1-wx)*(1-wy)*v00 + wx*(1-wy)*v10 + (1-wx)*wy*v01 + wx*wy*v11

def add_kde_regression_features(
    features: dict,
    source_path: str,
    *,
    pop_col="resident_pop",
    count_cols=("resident_pop","youth_workday_pop","workers_workday_pop","elderly_workday_pop"),
    avg_cols=("income_deprivation","employment_deprivation"),
    bandwidth_m=400,
    cell_size=100,
    kernel="quartic",
    layer=None,
    fill_avg="zero"  # "zero" | "global_mean" | None (leave NaN)
):
    """
    Adds KDE-style features to features["model"]:
      - KDE_* : intensity per m² (quartic kernel by default)
      - KWS_* : equivalent counts within radius (easier for regression)
      - AVG_* : population-weighted averages for index variables

    Parameters align with your existing code and keep Step 1 & 2 intact.
    """

    # 1) Validate inputs
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

    # 2) Read source points (CSV or GPKG) and align CRS
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

    # 3) Clip source points to units
    if not gdf_src.empty:
        # spatial join is faster than .within(unary_union) for larger sets
        gdf_src = gpd.sjoin(gdf_src, gdf_units[["geometry"]], predicate="within", how="inner").drop(columns=["index_right"])
    if gdf_src.empty:
        # If nothing left after clip: create zero outputs and return
        for c in count_cols:
            gdf_model[f"KDE_{c}"] = 0.0
            gdf_model[f"KWS_{c}"] = 0.0
        for c in avg_cols:
            if fill_avg == "zero":
                gdf_model[f"AVG_{c}"] = 0.0
            elif fill_avg == "global_mean":
                # no data -> fall back to 0; caller can overwrite later
                gdf_model[f"AVG_{c}"] = 0.0
            else:
                gdf_model[f"AVG_{c}"] = np.nan
        features["model"] = gdf_model
        return features

    # Ensure required columns exist
    missing_counts = [c for c in count_cols if c not in gdf_src.columns]
    missing_avgs   = [c for c in avg_cols   if c not in gdf_src.columns]
    if missing_counts or missing_avgs or pop_col not in gdf_src.columns:
        raise ValueError(f"Source missing columns. "
                         f"Counts missing: {missing_counts}; Averages missing: {missing_avgs}; "
                         f"pop_col present: {pop_col in gdf_src.columns}")

    # 4) Prepare raster extent (units bbox + pad by radius)
    xmin, ymin, xmax, ymax = gdf_units.total_bounds
    pad = bandwidth_m + cell_size
    xmin -= pad; ymin -= pad; xmax += pad; ymax += pad
    width  = int(np.ceil((xmax - xmin) / cell_size))
    height = int(np.ceil((ymax - ymin) / cell_size))

    # Map points -> cell indices
    xs = (gdf_src.geometry.x.values - xmin) / cell_size
    ys = (ymax - gdf_src.geometry.y.values) / cell_size
    cols = xs.astype(int); rows = ys.astype(int)
    good = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)

    # Float pixel coords for model sampling
    mx = (gdf_model.geometry.x.values - xmin) / cell_size
    my = (ymax - gdf_model.geometry.y.values) / cell_size

    # Kernel
    if kernel.lower() != "quartic":
        raise ValueError("Only 'quartic' is implemented here")
    K = _quartic_kernel(bandwidth_m / cell_size, cell_size)

    # Area factor to convert intensity (per m²) to equivalent count in a 400m radius
    # For quartic kernel, ∫K dA = 1; equivalent count within the radius is intensity * area_of_support
    # Use biweight effective area: (π/3) * R^2
    area_factor = (np.pi / 3.0) * (bandwidth_m ** 2)

    def _convolve(values: np.ndarray) -> np.ndarray:
        grid = np.zeros((height, width), dtype=np.float32)
        np.add.at(grid, (rows[good], cols[good]), values[good].astype(np.float32))
        return fftconvolve(grid, K, mode="same")  # intensity per m²

    # 5) Counts → KDE (intensity) and KWS (equiv. count)
    for c in count_cols:
        intensity = _convolve(gdf_src[c].values)
        eq_count  = intensity * area_factor
        gdf_model[f"KDE_{c}"] = _bilinear(intensity, mx, my).astype(np.float32)
        gdf_model[f"KWS_{c}"] = _bilinear(eq_count,  mx, my).astype(np.float32)
        # zero-fill for empty neighborhoods
        gdf_model[f"KDE_{c}"] = gdf_model[f"KDE_{c}"].fillna(0.0)
        gdf_model[f"KWS_{c}"] = gdf_model[f"KWS_{c}"].fillna(0.0)

    # 6) Population-weighted averages for index/proportion cols
    pop_intensity = _convolve(gdf_src[pop_col].values)     # per m²
    pop_eq = pop_intensity * area_factor                   # equiv. count
    denom = _bilinear(pop_eq, mx, my)                      # at target points

    for c in avg_cols:
        num = _convolve((gdf_src[c].values * gdf_src[pop_col].values).astype(np.float32)) * area_factor
        num_pts = _bilinear(num, mx, my)
        avg = np.divide(num_pts, denom, out=np.full_like(num_pts, np.nan, dtype=np.float32), where=denom > 0)
        if fill_avg == "zero":
            avg = np.nan_to_num(avg, nan=0.0)
        elif fill_avg == "global_mean":
            m = np.nanmean(avg)
            avg = np.where(np.isnan(avg), m, avg).astype(np.float32)
        gdf_model[f"AVG_{c}"] = avg.astype(np.float32)

    features["model"] = gdf_model
    return features