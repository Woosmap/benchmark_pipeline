import numpy as np
from collections import defaultdict
import duckdb
import pandas as pd

from benchmark_pipeline.config import *
from benchmark_pipeline.generator.template_question.registry import template


def segregate_pois(pois, geom, list_direction):
    c = geom.centroid
    dx = pois.geometry.x - c.x
    dy = pois.geometry.y - c.y
    masks = {"north": dy >= dx.abs(), "south": -dy >= dx.abs(),
             "east":  dx >= dy.abs(), "west":  -dx >= dy.abs()}
    return {direction: pois[masks[direction]] for direction in list_direction} 

@template("make_question_area_direction")
def make_question_area_direction(df_osm, df_area, list_direction=["north", "east", "west", "south"], min_size=20000, nb_q=200, seed=42):
    """Génère les questions de position relative dans une zone « X au nord de Z ».

    Pour chaque catégorie cible, tire des zones et décline chacune sur les quatre
    directions demandées.

    Args:
        df_osm (GeoDataFrame): POIs cibles en EPSG:2154.
        df_area (GeoDataFrame): Zones, avec `name` et `geometry` en EPSG:2154.
        list_direction (list[str]): Directions à décliner.
        min_size (float): Surface minimale d'une zone retenue, en m².
        nb_q (int): Nombre total de questions visé, stratifié par catégorie cible.
        seed (int): Graine du générateur aléatoire.

    Returns:
        DataFrame: Une ligne par (zone, direction). Colonnes `query`,
            `category_query`, `area_index`, `area_name`, `area_geometry`,
            `function`, `direction`, `results_poi_id`, `results_poi_x`,
            `results_poi_y`, `results_poi_name`, `results_poi_rank`.
    """
    dic_benchmark = defaultdict(list)
    rng = np.random.default_rng(seed)
    areas = df_area[df_area.geometry.area > min_size]
    sample_areas = rng.choice(len(areas), size=nb_q, replace=False)
    for id_a in sample_areas:
        area = areas.iloc[id_a]
        pois = df_osm[df_osm.geometry.within(area.geometry)]
        if pois.empty:
            continue
        else:
            results = segregate_pois(pois, area.geometry, list_direction)
            for direction, pois in results.items():
                dic_benchmark["query"].append(f"pois at the {direction} of {area['area_name']}")
                dic_benchmark["area_index"].append(area.name)
                dic_benchmark["area_name"].append(area["area_name"])
                dic_benchmark["area_geometry"].append(area.geometry)
                dic_benchmark["function"].append("direction_area")
                dic_benchmark["direction"].append(direction)
                dic_benchmark["results_poi_id"].append(list(pois.poi_id))
                dic_benchmark["results_poi_x"].append(list(pois.geometry.x))
                dic_benchmark["results_poi_y"].append(list(pois.geometry.y))
                dic_benchmark["results_poi_name"].append(list(pois.poi_name))
                dic_benchmark["results_poi_rank"].append(list(range(1, len(pois) + 1)))


    return pd.DataFrame(dic_benchmark)