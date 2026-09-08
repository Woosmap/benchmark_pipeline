import time
import os

import pandas as pd
import osmnx as ox
import geopandas as gpd
import json
import pyarrow as pa
from shapely.ops import unary_union

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
    cache_path = f"~/Documents/code/benchmark_NL_POI/data/pois/pois_woosmap_{bbox}.parquet"
    if os.path.exists(cache_path) and use_cache:
        print("lecture cache")
        gdf = gpd.read_parquet(cache_path)
        return gdf
    else:
        records = []
        with open('/Users/sese/Documents/code/benchmark_NL_POI/data/google_places_752452_1698_951_13.jl', 'r', encoding='utf-8') as f:
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
        safe_to_parquet(gdf, cache_path)
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

# ── 1. DONNÉES ───────────────────────────────────────────────────────────────
def extract_osm_pois(bbox, dic_tags, use_cache=True):
    """
    Extrait les POIs OpenStreetMap dans une bounding box pour un ensemble de tags.

    Interroge l'API Overpass via osmnx pour récupérer les features OSM correspondant
    aux tags fournis. Les géométries polygonales sont réduites à leur centroïde.
    Les coordonnées sont ajoutées en colonnes lat/lon (WGS84) et x/y (Lambert-93).
    Les catégories utilitaires sans intérêt sémantique (bancs, poubelles, etc.)
    sont filtrées. Le résultat brut est mis en cache Parquet.

    Args:
        bbox (tuple): Bounding box (lon_min, lat_min, lon_max, lat_max) en WGS84.
        dic_tags (dict): Tags OSM à requêter, ex. {"amenity": ["cafe", "restaurant"]}.
        use_cache (bool): Si True (défaut), charge le cache Parquet s'il existe.

    Returns:
        GeoDataFrame: POIs nettoyés, projetés en EPSG:2154, avec colonnes
            lat, lon, x, y et category.
    """
    cache_path = f"~/Documents/code/benchmark_NL_POI/data/pois/pois_osm_{bbox}.parquet"
    if os.path.exists(cache_path) and use_cache:
        print("lecture cache")
        pois = gpd.read_parquet(cache_path)
    else:
        pois = ox.features_from_bbox(
            bbox,
            tags=dic_tags
        )
        pois.to_parquet(cache_path)
    pois["geometry"] = pois.geometry.centroid
    pois["lat"] = pois.geometry.y
    pois["lon"] = pois.geometry.x
    pois = pois.to_crs(epsg=2154)
    pois["x"] = pois.geometry.x
    pois["y"] = pois.geometry.y
    pois["category"] = pois.apply(lambda row: get_category(row, dic_tags), axis=1)
    pois["cuisine"] = pois["cuisine"].str.replace("_", " ", regex=False)
    categories_a_exclure = [
    'bench', 'waste_basket', 'bicycle_parking', 
    'vending_machine', 'yes', 'parking_space',
    'ticket_validator', 'recycling'
    ]
    pois = pois[~pois['category'].isin(categories_a_exclure)]
    return pois.reset_index(drop=True)

def extraire_ancres(ville: str, ancres: dict) -> dict:
    """
    Récupère les géométries des ancres spatiales (stations de métro, parcs, etc.).

    Pour chaque ancre définie dans le dictionnaire, interroge OSM via osmnx pour
    extraire les features correspondantes dans la ville donnée. Les géométries
    non-ponctuelles sont réduites à leur centroïde. Un délai de 1 seconde est
    appliqué entre chaque requête pour respecter les limites de l'API Overpass.
    En cas d'erreur, la clé correspondante est conservée avec la valeur None.

    Args:
        ville (str): Nom de la ville au format osmnx, ex. "Paris, France".
        ancres (dict): Dictionnaire {nom_ancre: tags_osm}, ex.
            {"metro": {"railway": "subway_entrance"}}.

    Returns:
        dict: {nom_ancre: GeoDataFrame | None} — None si l'extraction a échoué.
    """
    resultats = {}
    for nom, tags in ancres.items():
        print(f"  → Extraction ancre {nom}...")
        try:
            gdf = ox.features_from_place(ville, tags=tags)
            gdf["geometry"] = gdf["geometry"].apply(
                lambda g: g.centroid if g.geom_type != "Point" else g
            )
            resultats[nom] = gdf
            print(f"     {len(gdf)} éléments trouvés")
            time.sleep(1)
        except Exception as e:
            print(f"     Erreur : {e}")
            resultats[nom] = None
    return resultats


def extraire_seine(ville: str = "Paris") -> gpd.GeoDataFrame:
    """
    Récupère la géométrie de la Seine depuis OpenStreetMap.

    Interroge osmnx avec les tags waterway=river et name=La Seine pour obtenir
    l'ensemble des segments du fleuve dans la zone de la ville fournie. Utilisé
    en amont du calcul de la feature `close_to_the_seine`.

    Args:
        ville (str): Nom de la ville au format osmnx, ex. "Paris, France".

    Returns:
        GeoDataFrame | None: GeoDataFrame contenant les segments de la Seine,
            ou None en cas d'erreur lors de la requête.
    """
    print("  → Extraction Seine...")
    try:
        tags = {"waterway": "river", "name": "La Seine"}
        seine = ox.features_from_place(ville, tags=tags)
        seine_lines = seine[seine.geom_type.isin(["LineString", "MultiLineString"])]
        seine_lines = seine_lines.to_crs(epsg=2154)

        # fusionne tous les tronçons en une seule géométrie
        seine_geom = unary_union(seine_lines.geometry.values).buffer(500)
        seine_buffer = gpd.GeoSeries([seine_geom], crs=2154)
        return seine_buffer
    except Exception as e:
        print(f"     Erreur Seine : {e}")
        return None

def get_poi_near_seine(pois_gdf, seine_buffer):
    pois_gdf["near_seine"] = pois_gdf.geometry.within(seine_buffer)
    return pois_gdf