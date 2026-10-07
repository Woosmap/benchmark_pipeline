import numpy as np
from collections import defaultdict
import duckdb
import pandas as pd

from benchmark_pipeline.config import *
from benchmark_pipeline.generator.template_question.registry import template


@template("make_question_area_inside")
def make_question_area_inside(df_osm, df_area, min_size=1000, nb_q=110, seed=42):
    """Génère les questions d'appartenance « X dans la zone Z ».

    Pour chaque catégorie cible, tire des zones jusqu'à atteindre le quota, en
    ne retenant que celles qui contiennent au moins un POI de la catégorie. Le
    nombre de tentatives est plafonné à 200 par strate.

    Args:
        df_osm (GeoDataFrame): POIs cibles en EPSG:2154.
        df_area (GeoDataFrame): Zones, avec `name` et `geometry` en EPSG:2154.
        min_size (float): Surface minimale d'une zone retenue, en m².
        nb_q (int): Nombre total de questions visé, stratifié par catégorie cible.
        seed (int): Graine du générateur aléatoire.

    Returns:
        DataFrame: Une ligne par question. Colonnes `query`, `category_query`,
            `area_index`, `area_name`, `area_geometry`, `function`, plus les
            quatre colonnes `results_poi_*`.
    """
    dic_benchmark = defaultdict(list)
    rng = np.random.default_rng(seed)
    areas = df_area[df_area.geometry.area > min_size]
    areas = df_area[df_area.geometry.area > min_size]
    sample_areas = rng.choice(min(len(areas), nb_q), size=nb_q, replace=False)
    for id_a in sample_areas:
        area = areas.iloc[id_a]
        results = df_osm[df_osm.geometry.within(area.geometry)]
        if results.empty:
            continue
        else:
            dic_benchmark["query"].append(f"pois inside {area['area_name']}")
            dic_benchmark["area_index"].append(area.name)
            dic_benchmark["area_name"].append(area["area_name"])
            dic_benchmark["area_geometry"].append(area.geometry)
            dic_benchmark["function"].append("inside_area")
            dic_benchmark["results_poi_id"].append(list(results.poi_id))
            dic_benchmark["results_poi_name"].append(list(results.poi_name))
            dic_benchmark["results_poi_dist"].append(list(results.geometry.distance(area.geometry)))
            dic_benchmark["results_poi_rank"].append(list(range(1, len(results) + 1)))
    return pd.DataFrame(dic_benchmark)