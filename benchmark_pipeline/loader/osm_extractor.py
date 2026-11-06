import json
import os

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import quackosm
from shapely.geometry import box

from benchmark_pipeline.path import CACHE_DIR, GOOGLE_PLACES_FILE, DATA, OSM_DATA


def safe_to_parquet(gdf, path):
    """
    Sauvegarde un GeoDataFrame en fichier Parquet en gérant les types incompatibles.

    Parcourt chaque colonne (sauf la géométrie) et tente une conversion PyArrow.
    Si une colonne contient des types non sérialisables (listes, objets mixtes, etc.),
    elle est convertie en chaîne de caractères pour garantir la compatibilité Parquet.

    Args:
        gdf (GeoDataFrame): Le GeoDataFrame à sauvegarder.
        path (str): Chemin de destination du fichier .parquet.
    """
    gdf = gdf.copy()
    for col in gdf.columns:
        if col == "geometry":
            continue
        try:
            pa.array(gdf[col], from_pandas=True)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
            gdf[col] = gdf[col].astype(str)
    gdf.to_parquet(path)

def extract_woosmap_pois(bbox, features, use_cache=True):
    """
    Charge les POIs issus d'un export Woosmap/Google Places et les filtre par bounding box.

    Lit un fichier JSON Lines local contenant les places Google, filtre les POIs
    inclus dans la bbox fournie, exclut les établissements fermés définitivement,
    et projette le résultat en Lambert-93 (EPSG:2154). Le résultat est mis en cache
    sous forme de fichier Parquet pour éviter les rechargements répétés.

    Args:
        bbox (tuple): Bounding box (lon_min, lat_min, lon_max, lat_max) en WGS84.
        features (list[str] | None): Liste de colonnes à conserver. Si None, toutes
            les colonnes sont conservées.
        use_cache (bool): Si True (défaut), charge le cache Parquet s'il existe.

    Returns:
        GeoDataFrame: POIs filtrés, projetés en EPSG:2154, avec colonne geometry.
    """
    if os.path.exists(CACHE_DIR) and use_cache:
        print("lecture cache")
        gdf = gpd.read_parquet(CACHE_DIR)
        return gdf
    else:
        records = []
        with open(GOOGLE_PLACES_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                records.append(json.loads(line))
        pois = pd.DataFrame(records)
        pois.rename(columns={"latitude": "lat", "longitude": "lon"}, inplace=True)
        df_bbox = pois[
            (pois['lon'] >= bbox[0]) &
            (pois['lon'] <= bbox[2]) &
            (pois['lat'] >= bbox[1]) &
            (pois['lat'] <= bbox[3]) &
            (pois['permanently_closed']==False)
            ].copy()
        if features is not None:
            cols = [c for c in features + ["lon", "lat", "geometry"] if c in df_bbox.columns]
            df_bbox = df_bbox[cols]
        gdf = gpd.GeoDataFrame(df_bbox, geometry=gpd.points_from_xy(df_bbox["lon"], df_bbox["lat"]), crs="EPSG:4326")
        gdf = gdf.to_crs(epsg=2154)
        safe_to_parquet(gdf, CACHE_DIR)
    return gdf


def get_category(row, dic_tags):
    """
    Détermine la catégorie sémantique d'un POI OSM à partir de ses tags.

    Inspecte les colonnes OSM dans un ordre de priorité décroissante (amenity,
    leisure, railway, public_transport, shop, landuse) et retourne la valeur
    du premier tag non-nul trouvé. Utilisée comme fonction appliquée ligne par
    ligne sur un GeoDataFrame via `apply`.

    Args:
        row (Series): Une ligne du GeoDataFrame OSM.

    Returns:
        str: La valeur du premier tag catégoriel présent, ou "unknown" si aucun
            tag reconnu n'est renseigné.
    """
    for col, values in dic_tags.items():
        if col in row and isinstance(row[col], str) and row[col] in values:
            return row[col]
    return "unknown"

def features_from_geofabrik(bbox, tags, city_name):
    """OSM features of `bbox` with one of `tags`, read from the Geofabrik extract covering it.

    The extract is downloaded into OSM_DATA/<city_name> the first time, then reused.

    Args:
        bbox (tuple): (west, south, east, north) in WGS84.
        tags (dict): OSM tags kept, as in osmnx, e.g. {"amenity": ["cafe", "restaurant"]}.
        city_name (str): Folder of the city in OSM_DATA.

    Returns:
        GeoDataFrame: in EPSG:4326, one column per OSM tag, indexed like osmnx by (element, id).
    """
    features = quackosm.convert_geometry_to_geodataframe(
        geometry_filter=box(*bbox), tags_filter=tags, osm_extract_source="Geofabrik",
        keep_all_tags=True, explode_tags=True, working_directory=OSM_DATA / city_name, verbosity_mode="silent")
    # quackosm ids are "way/123"; osmnx's are ("way", 123), which load_area relies on
    element, osm_id = zip(*(feature_id.split("/") for feature_id in features.index))
    features.index = pd.MultiIndex.from_arrays([list(element), [int(i) for i in osm_id]], names=["element", "id"])
    return features

# ── 1. DONNÉES ───────────────────────────────────────────────────────────────
def extract_osm_pois(bbox, ecrs, dic_tags, categories_excluded, city_name, use_cache=True):
    """
    Extrait les POIs OpenStreetMap dans une bounding box pour un ensemble de tags.

    Lit l'extrait Geofabrik de la zone (`features_from_geofabrik`) pour récupérer les features OSM correspondant
    aux tags fournis. Les géométries polygonales sont réduites à leur centroïde.
    Les coordonnées sont ajoutées en colonnes lat/lon (WGS84) et x/y (Lambert-93).
    Les catégories utilitaires sans intérêt sémantique (bancs, poubelles, etc.)
    sont filtrées. Le résultat brut est mis en cache Parquet.

    Args:
        bbox (tuple): Bounding box (lon_min, lat_min, lon_max, lat_max) en WGS84.
        dic_tags (dict): Tags OSM à requêter, ex. {"amenity": ["cafe", "restaurant"]}.
        categories_excluded (list[str]): Categories dropped from the result.
        city_name (str): Folder of the city in OSM_DATA.
        use_cache (bool): Si True (défaut), charge le cache Parquet s'il existe.

    Returns:
        GeoDataFrame: POIs nettoyés, projetés en EPSG:2154, avec colonnes
            lat, lon, x, y et category.
    """
    path_pois = DATA / "pois" / f"pois_osm_{bbox}.parquet"
    if os.path.exists(path_pois) and use_cache:
        print("lecture cache")
        pois = gpd.read_parquet(path_pois)
    else:
        pois = features_from_geofabrik(bbox, dic_tags, city_name)
        path_pois.parent.mkdir(parents=True, exist_ok=True)
        safe_to_parquet(pois, path_pois)
    # Tags read by the loaders: a city may have no POI carrying one of them (no indoor_seating in Leeuwarden)
    for column in ("name", "cuisine", "indoor_seating", "outdoor_seating"):
        if column not in pois:
            pois[column] = None
    pois["geometry"] = pois.geometry.centroid
    pois["lat"] = pois.geometry.y
    pois["lon"] = pois.geometry.x
    pois = pois.to_crs(epsg=ecrs)
    pois["x"] = pois.geometry.x
    pois["y"] = pois.geometry.y
    pois["category"] = pois.apply(lambda row: get_category(row, dic_tags), axis=1)
    pois["cuisine"] = pois["cuisine"].str.replace("_", " ", regex=False)
    pois = pois[~pois['category'].isin(categories_excluded)]
    return pois.reset_index(drop=True)


