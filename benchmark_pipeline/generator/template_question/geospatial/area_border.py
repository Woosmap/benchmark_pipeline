import numpy as np
from collections import defaultdict
import duckdb
import pandas as pd

from benchmark_pipeline.config import *
from benchmark_pipeline.generator.template_question.registry import template


def pois_near_border(df_osm, area, band):
    border = area.boundary
    idx = df_osm.sindex.query(border, predicate="dwithin", distance=band)
    pois = df_osm.iloc[idx]
    return pois.assign(dist=pois.geometry.distance(border)).sort_values("dist")

@template("make_question_area_border")
def make_question_area_border(df_osm, df_area, min_size=20000, band=200, nb_q=110, seed=42):
    """Génère les questions de bordure « X en limite de la zone Z ».

    Pour chaque catégorie cible, tire des zones jusqu'à atteindre le quota, dans
    la limite de 200 tentatives.

    Args:
        df_osm (GeoDataFrame): POIs cibles en EPSG:2154.
        df_area (GeoDataFrame): Zones, avec `name` et `geometry` en EPSG:2154.
        min_size (float): Surface minimale d'une zone retenue, en m².
        nb_q (int): Nombre total de questions visé, stratifié par catégorie cible.
        seed (int): Graine du générateur aléatoire.

    Returns:
        DataFrame: Une ligne par question, mêmes colonnes que `make_question_insidearea`.
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
            results = pois_near_border(df_osm, area.geometry, band)
            if results.empty:
                continue
            nq +=1

            dic_benchmark["query"].append(f"{cat_q} at the border of {area['area_name']}")
            dic_benchmark["category_query"].append(cat_q)
            dic_benchmark["area_index"].append(area.name)
            dic_benchmark["area_name"].append(area["area_name"])
            dic_benchmark["area_geometry"].append(area.geometry)
            dic_benchmark["function"].append("border_area")
            dic_benchmark["results_poi_id"].append(list(results.poi_id))
            dic_benchmark["results_poi_name"].append(list(results.poi_name))
            dic_benchmark["results_poi_dist"].append(list(results["dist"]))
            dic_benchmark["results_poi_rank"].append(list(range(1, len(results) + 1)))


    return pd.DataFrame(dic_benchmark)