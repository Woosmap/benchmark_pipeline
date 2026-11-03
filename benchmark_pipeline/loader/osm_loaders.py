import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.ops import linemerge, unary_union

from benchmark_pipeline.config import (
    AREA_TAGS,
    BBOX,
    CATEGORIES,
    FEATURES,
    STREETS_TAGS,
    ECRS,
)
from benchmark_pipeline.loader.data_cleaning.cooking_column import (
    classify_cuisine_column,
)
from benchmark_pipeline.loader.data_cleaning.serialize import serialize_poi
from benchmark_pipeline.loader.osm_extractor import extract_osm_pois, get_category
from benchmark_pipeline.path import CACHE_DIR
from benchmark_pipeline.utils.bbox_reshap import bbox_wgs84_to_lambert

ox.settings.cache_folder = CACHE_DIR

import unicodedata

import geopandas as gpd
from shapely.ops import linemerge


def merge_close_streets(df_street, max_distance=500):
    """Merge pieces of streets with the same normalized name lying less than `max_distance` apart.

    Args:
        df_street (GeoDataFrame): Streets from `load_streets`, in a metric CRS.
        max_distance (float): Merge threshold, in CRS units (meters in EPSG:2154).

    Returns:
        GeoDataFrame: One row per merged street, with the index, `street_id` and name of its first piece.
    """
    def normalize(name):
        name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
        return " ".join(name.lower().split())

    name_norm = df_street["street_name"].map(normalize, na_action="ignore")
    kept_labels, merged_geoms = [], []
    for _, group in df_street.groupby(name_norm, sort=False):
        buffers = group.buffer(max_distance / 2).union_all()
        clusters = list(buffers.geoms) if buffers.geom_type == "MultiPolygon" else [buffers]
        for cluster in clusters:
            members = group[group.intersects(cluster)]
            geom = members.union_all()
            kept_labels.append(members.index[0])
            merged_geoms.append(linemerge(geom) if geom.geom_type == "MultiLineString" else geom)

    merged_geometry = gpd.GeoDataFrame(geometry=merged_geoms, index=kept_labels, crs=df_street.crs).geometry
    df_merged = df_street.loc[kept_labels].assign(geometry=merged_geometry).sort_index()
    print(f"{len(df_street) - len(df_merged)} morceaux de rue recollés")
    return df_merged

def load_streets():
    streets = ox.features_from_bbox(BBOX, tags=STREETS_TAGS)
    streets = streets[streets.geom_type.isin(["LineString", "MultiLineString"])].to_crs(ECRS)
    rows = []
    for name, g in streets.groupby("name")["geometry"]:
        u = unary_union(g.tolist())
        parts = list(u.geoms) if u.geom_type == "MultiLineString" else [u]
        buf = unary_union([p.buffer(50) for p in parts])
        clusters = list(buf.geoms) if buf.geom_type == "MultiPolygon" else [buf]
        for i, c in enumerate(clusters):
            geom = unary_union([p for p in parts if p.intersects(c)])
            if geom.geom_type == "MultiLineString":
                street = linemerge(geom)
                rows.append({"name": name, "geometry": street})
            else: rows.append({"name": name, "geometry": geom})

    df_streets = gpd.GeoDataFrame(rows, crs=streets.crs)
    df_streets["street_id"]= df_streets.index
    df_streets = df_streets[["street_id", "name", "geometry"]]
    df_streets = df_streets.rename(columns={"index": "street_id", "name": "street_name"})
    return merge_close_streets(df_streets)

def load_pois(use_cache=False):
    df_osm = extract_osm_pois(BBOX, CATEGORIES, use_cache=use_cache)
    df_osm = classify_cuisine_column(df_osm)
    poi_all_index = df_osm.index.tolist()
    df_osm["index"] = poi_all_index
    df_osm = df_osm[df_osm['geometry'].within(bbox_wgs84_to_lambert(BBOX))]
    df_osm = df_osm[["index", "name", "category", "cuisine", "geometry", "indoor_seating", "outdoor_seating"]] #"lat", "lon", "x", "y"
    df_osm = df_osm.dropna(subset=["name"])
    df_osm = df_osm.rename(columns={"index": "poi_id", "name": "poi_name", "cuisine": "cooking_type"})
    df_osm["bow_poi"] = [serialize_poi(row, [f.COLUMN for f in FEATURES]) for _, row in df_osm.iterrows()]
    return df_osm

def add_area_category(area, dic_tags=AREA_TAGS):
    area["category"] = area.apply(lambda row: get_category(row, dic_tags), axis=1)
    return area

def load_area():
    area = ox.features_from_bbox(BBOX, tags=AREA_TAGS)
    poi_all_index = area.index.tolist()
    area["area_id"] = [int(i) for _, i in poi_all_index]
    area = area[area.geom_type.isin(["Polygon"])].to_crs(ECRS)
    area = area[~area.name.isna()]
    lvl = pd.to_numeric(area["admin_level"], errors="coerce")
    area = area[lvl.isna() | (lvl >= 9)]
    area = add_area_category(area)
    df_area = area.drop_duplicates(subset="name", keep="first")
    df_area = df_area[["area_id", "name", "category", "geometry"]]
    df_area = df_area.rename(columns={"name": "area_name"})
    return df_area