import geopandas as gpd
from shapely.ops import linemerge
import osmnx as ox
from shapely.ops import unary_union
import pandas as pd

from src.config import BBOX, AREA_TAGS, STREETS_TAGS, CATEGORIES
from src.data_collection.osm_extractor import extract_osm_pois, get_category
from src.data_collection.data_cleaning.cooking_column import classify_cuisine_column
from src.utils.bbox_reshap import bbox_wgs84_to_lambert


def load_streets():
    streets = ox.features_from_bbox(BBOX, tags=STREETS_TAGS)
    streets = streets[streets.geom_type.isin(["LineString", "MultiLineString"])].to_crs(2154)
    rows = []
    for name, g in streets.groupby("name")["geometry"]:
        u = unary_union(g.tolist())
        parts = list(u.geoms) if u.geom_type == "MultiLineString" else [u]
        # regroupe les composantes qui se touchent ou sont à moins de 50 m
        buf = unary_union([p.buffer(50) for p in parts])
        clusters = list(buf.geoms) if buf.geom_type == "MultiPolygon" else [buf]
        for i, c in enumerate(clusters):
            geom = unary_union([p for p in parts if p.intersects(c)])
            if geom.geom_type == "MultiLineString":
                street = linemerge(geom)
                rows.append({"name": name, "geometry": street})
            else: rows.append({"name": name, "geometry": geom})

    df_streets = gpd.GeoDataFrame(rows, crs=streets.crs)
    #df_streets = df_streets[df_streets["part"]==0]
    df_streets["id_street"]= df_streets.index
    df_streets = df_streets[["id_street", "name", "geometry"]]
    df_streets = df_streets.rename(columns={"index": "id_street", "name": "street_name"})
    return df_streets

def load_pois():
    df_osm = extract_osm_pois(BBOX, CATEGORIES, use_cache=False)
    df_osm = classify_cuisine_column(df_osm)
    poi_all_index = df_osm.index.tolist()
    df_osm["index"] = poi_all_index
    df_osm = df_osm[df_osm['geometry'].within(bbox_wgs84_to_lambert(BBOX))]
    df_osm = df_osm[["index", "name", "category", "cuisine", "geometry", "lat", "lon", "x", "y", "indoor_seating", "outdoor_seating"]]
    df_osm = df_osm.dropna(subset=["name"])
    df_osm = df_osm.rename(columns={"index": "poi_id", "name": "poi_name"})
    return df_osm

def add_area_category(area, dic_tags=AREA_TAGS):
    # à appeler sur le GeoDataFrame brut : les colonnes de tags (`leisure`,
    # `boundary`) dont la catégorie est déduite sont supprimées ensuite
    area["category"] = area.apply(lambda row: get_category(row, dic_tags), axis=1)
    return area

def load_area():
    area = ox.features_from_bbox(BBOX, tags=AREA_TAGS)
    poi_all_index = area.index.tolist()
    area["index"] = poi_all_index
    area = area[area.geom_type.isin(["Polygon"])].to_crs(2154)
    area = area[~area.name.isna()]
    lvl = pd.to_numeric(area["admin_level"], errors="coerce")
    area = area[lvl.isna() | (lvl >= 9)]
    area = add_area_category(area)
    df_area = area.drop_duplicates(subset="name", keep="first")
    df_area = df_area[["index", "name", "category", "geometry"]]
    df_area = df_area.rename(columns={"name": "area_name"})
    return df_area