import hashlib
import json

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import quackosm
from shapely.geometry import box

from benchmark_pipeline.path import CACHE_DIR, GOOGLE_PLACES_FILE, DATA, OSM_DATA
from benchmark_pipeline.config import USE_CACHE


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

def raw_features_path(bbox, tags, city_name):
    """Parquet file holding the raw features of `bbox` with `tags`: a new file whenever the bbox or the tags change."""
    key = hashlib.sha1(json.dumps([list(bbox), tags], sort_keys=True).encode()).hexdigest()[:12]
    return OSM_DATA / city_name / f"raw_{key}.parquet"


def features_from_geofabrik(bbox, tags, city_name, use_cache=USE_CACHE):
    """OSM features of `bbox` with one of `tags`, read from the Geofabrik extract covering it.

    The extract is downloaded into OSM_DATA/<city_name> the first time, then reused. The raw features are saved
    in OSM_DATA/<city_name>/raw_<key>.parquet (`raw_features_path`) on the first call; with `use_cache` later calls
    reread this file without opening the extract, without it they extract the features again and overwrite it.

    Args:
        bbox (tuple): (west, south, east, north) in WGS84.
        tags (dict): OSM tags kept, as in osmnx, e.g. {"amenity": ["cafe", "restaurant"]}.
        city_name (str): Folder of the city in OSM_DATA.
        use_cache (bool): Reread the saved raw features when they exist.

    Returns:
        GeoDataFrame: in EPSG:4326, one column per OSM tag, indexed like osmnx by (element, id).
    """
    path = raw_features_path(bbox, tags, city_name)
    if use_cache and path.exists():
        print(f"lecture cache {path.name}")
        return gpd.read_parquet(path)
    features = quackosm.convert_geometry_to_geodataframe(
        geometry_filter=box(*bbox), tags_filter=tags, osm_extract_source="Geofabrik",
        keep_all_tags=True, explode_tags=True, working_directory=OSM_DATA / city_name, verbosity_mode="silent",
        ignore_cache=not use_cache)
    # quackosm ids are "way/123"; osmnx's are ("way", 123), which load_area relies on
    element, osm_id = zip(*(feature_id.split("/") for feature_id in features.index))
    features.index = pd.MultiIndex.from_arrays([list(element), [int(i) for i in osm_id]], names=["element", "id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_to_parquet(features, path)
    return features

# ── 1. DONNÉES ───────────────────────────────────────────────────────────────
def extract_osm_pois(bbox, ecrs, dic_tags, categories_excluded, city_name, use_cache=USE_CACHE):
    """
    Extrait les POIs OpenStreetMap dans une bounding box pour un ensemble de tags.

    Lit l'extrait Geofabrik de la zone (`features_from_geofabrik`) pour récupérer les features OSM correspondant
    aux tags fournis. Les géométries polygonales sont réduites à leur centroïde.
    Les coordonnées sont ajoutées en colonnes lat/lon (WGS84) et x/y (Lambert-93).
    Les catégories utilitaires sans intérêt sémantique (bancs, poubelles, etc.)
    sont filtrées. Le résultat brut est mis en cache Parquet par `features_from_geofabrik`.

    Args:
        bbox (tuple): Bounding box (lon_min, lat_min, lon_max, lat_max) en WGS84.
        dic_tags (dict): Tags OSM à requêter, ex. {"amenity": ["cafe", "restaurant"]}.
        categories_excluded (list[str]): Categories dropped from the result.
        city_name (str): Folder of the city in OSM_DATA.
        use_cache (bool): Si True (défaut), relit les features brutes déjà enregistrées.

    Returns:
        GeoDataFrame: POIs nettoyés, projetés en EPSG:2154, avec colonnes
            lat, lon, x, y et category.
    """
    pois = features_from_geofabrik(bbox, dic_tags, city_name, use_cache=use_cache)
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


