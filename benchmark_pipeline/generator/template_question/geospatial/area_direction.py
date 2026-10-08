from collections import defaultdict

import numpy as np
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
def make_question_area_direction(df_osm, df_area, list_direction=None, min_size=20000, nb_q=200, seed=42):
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
    if list_direction is None:
        list_direction=["north", "east", "west", "south"]
    dic_benchmark = defaultdict(list)
    rng = np.random.default_rng(seed)
    areas = df_area[df_area.geometry.area > min_size]
    # `size` plafonné au corpus : `replace=False` exige une population au moins
    # aussi grande que l'échantillon, sinon ValueError sur un corpus réduit.
    # `nb_q` est un plafond sur les *questions*, pas sur les zones : chacune
    # est déclinée sur chaque direction de `list_direction`, donc le quota se divise d'abord. Sans cette
    # division le template rendait 4 × nb_q. Diviser plutôt que tronquer après
    # coup garde chaque zone complet — couper dans le tas amputerait
    # les dernières de leurs directions.
    nb_zones = min(len(areas), nb_q // len(list_direction))
    sample_areas = rng.choice(len(areas), size=nb_zones, replace=False)
    for id_a in sample_areas:
        area = areas.iloc[id_a]
        pois = df_osm[df_osm.geometry.within(area.geometry)]
        if pois.empty:
            continue
        else:
            results = segregate_pois(pois, area.geometry, list_direction)
            centre = area.geometry.centroid
            # `segregate_pois` masque sans trier : le rang numérotait l'ordre
            # d'apparition dans `df_osm`, c'est-à-dire rien. Le classement se
            # fait ici, par distance croissante au centroïde — critère que les
            # templates voisins emploient déjà, et seul ordre qui respecte la
            # monotonie de `results_poi_dist` exigée par `validate_benchmark`.
            for direction, pois_direction in results.items():
                pois_direction = pois_direction.assign(
                    dist=pois_direction.geometry.distance(centre)
                ).sort_values("dist")
                dic_benchmark["query"].append(f"pois at the {direction} of {area['area_name']}")
                dic_benchmark["area_index"].append(area.name)
                dic_benchmark["area_name"].append(area["area_name"])
                dic_benchmark["area_geometry"].append(area.geometry)
                dic_benchmark["function"].append("segregate_pois")
                dic_benchmark["direction"].append(direction)
                dic_benchmark["results_poi_id"].append(list(pois_direction.poi_id))
                dic_benchmark["results_poi_x"].append(list(pois_direction.geometry.x))
                dic_benchmark["results_poi_y"].append(list(pois_direction.geometry.y))
                dic_benchmark["results_poi_name"].append(list(pois_direction.poi_name))
                dic_benchmark["results_poi_dist"].append(list(pois_direction["dist"]))
                dic_benchmark["results_poi_rank"].append(list(range(1, len(pois_direction) + 1)))


    return pd.DataFrame(dic_benchmark)