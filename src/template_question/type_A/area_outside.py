import numpy as np
from collections import defaultdict
import duckdb
import pandas as pd

from src.config import *


def outside_area(area, category, pois, cat_col="category"):
    """Classe les POIs d'une catégorie par distance à une zone.

    Corps identique à `inside_area` ; c'est l'appelant qui restreint aux POIs
    extérieurs. Ici la distance est réellement discriminante, puisqu'elle est
    strictement positive hors du polygone.

    Args:
        area (BaseGeometry): Polygone de la zone, en EPSG:2154.
        category (str): Catégorie de POI recherchée.
        pois (GeoDataFrame): POIs candidats, déjà restreints par l'appelant.
        cat_col (str): Nom de la colonne de catégorie.

    Returns:
        DataFrame: `pois` filtré sur la catégorie, avec `dist`, trié par distance
            croissante et réindexé.
    """
    sel = pois[pois[cat_col] == category].copy()
    sel["dist"] = sel.geometry.distance(area)
    return sel.sort_values("dist").reset_index(drop=True)

def make_question_area_outside(df_osm, df_area, min_size=1000, nb_q=110, seed=42):
    """Génère les questions d'exclusion « X hors de la zone Z ».

    Même structure que `make_question_insidearea`, avec le complémentaire de la
    zone.

    Args:
        df_osm (GeoDataFrame): POIs cibles en EPSG:2154.
        df_area (GeoDataFrame): Zones, avec `name` et `geometry` en EPSG:2154.
        min_size (float): Surface minimale d'une zone retenue, en m².
        nb_q (int): Nombre total de questions visé, stratifié par catégorie cible.
        seed (int): Graine du générateur aléatoire.

    Returns:
        DataFrame: Une ligne par question, mêmes colonnes que
            `make_question_insidearea`.
    TODO: `results_poi_rank` 0-based, cf. `make_question_insidearea`.
    """
    dic_benchmark = defaultdict(list)
    rng = np.random.default_rng(seed)
    list_cat = df_osm["category"].unique()
    n_queries_per_stratum = nb_q // len(list_cat)

    areas = df_area[df_area.geometry.area > min_size]

    for cat_q in list_cat:
        nq = 0
        i = 0
        while nq != n_queries_per_stratum and i<200:
            i+=1
            area = areas.loc[rng.choice(areas.index)]
            df_poi_in_area = df_osm[~df_osm.geometry.within(area.geometry)]
            pois = df_poi_in_area[df_poi_in_area["category"]==cat_q]
            if pois.empty:
                continue
            else:
                results = outside_area(area.geometry, cat_q, pois)
                nq +=1

                dic_benchmark["query"].append(f"{cat_q} outside {area['name_area']}")
                dic_benchmark["category_query"].append(cat_q)
                dic_benchmark["area_index"].append(area.name)
                dic_benchmark["area_name"].append(area["name"])
                dic_benchmark["area_geometry"].append(area.geometry)
                dic_benchmark["function"].append("outside_area")
                dic_benchmark["results_poi_id"].append(list(results.poi_id))
                dic_benchmark["results_poi_name"].append(list(results.poi_name))
                dic_benchmark["results_poi_dist"].append(list(results["dist"]))
                dic_benchmark["results_poi_rank"].append(list(range(1, len(results) + 1)))


    return pd.DataFrame(dic_benchmark)