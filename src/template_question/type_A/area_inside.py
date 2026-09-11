import numpy as np
from collections import defaultdict
import duckdb
import pandas as pd

from src.config import *


def inside_area(area, category, pois, cat_col="category"):
    """Classe les POIs d'une catégorie par distance à une zone.

    Ne filtre pas sur l'appartenance à la zone : le filtre `within` est appliqué
    par l'appelant, cette fonction ne fait que trier ce qu'on lui donne.

    Args:
        area (BaseGeometry): Polygone de la zone, en EPSG:2154.
        category (str): Catégorie de POI recherchée.
        pois (GeoDataFrame): POIs candidats, déjà restreints par l'appelant.
        cat_col (str): Nom de la colonne de catégorie.

    Returns:
        DataFrame: `pois` filtré sur la catégorie, avec une colonne `dist`, trié
            par distance croissante et réindexé.

    TODO: le tri est inopérant dans le cas « inside ». La distance d'un point
        intérieur à son polygone vaut 0 en géométrie shapely, donc `dist` est nul
        pour tous les POIs et `sort_values` conserve l'ordre d'insertion : le
        classement de la vérité terrain est arbitraire. Choisir un critère qui
        ordonne réellement — distance au centroïde, ou distance au bord via
        `area.exterior` — ou assumer que la relation « inside » est un ensemble
        non ordonné et le documenter comme tel dans le protocole d'évaluation.
    TODO: corps identique à `outside_area` et `border_area` ; une seule fonction
        paramétrée par la géométrie de référence suffirait.
    """
    sel = pois[pois[cat_col] == category].copy()
    sel["dist"] = sel.geometry.distance(area)
    return sel.sort_values("dist").reset_index(drop=True)

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

    TODO: `results_poi_rank` vaut `list(results.index)` après `reset_index`, donc
        0-based, alors que les générateurs rue sont 1-based.
    TODO: le plafond de 200 tentatives est atteint silencieusement ; journaliser
    TODO: `area.name` est le label d'index de la Series, `area["name"]` le nom de
        la zone. Les deux sont utilisés à trois lignes d'écart — correct, mais à
        désambiguïser pour la relecture.
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
            df_poi_in_area = df_osm[df_osm.geometry.within(area.geometry)]
            pois = df_poi_in_area[df_poi_in_area["category"]==cat_q]
            if pois.empty:
                continue
            else:
                results = inside_area(area.geometry, cat_q, pois)
                nq +=1

                dic_benchmark["query"].append(f"{cat_q} inside {area['name_area']}")
                dic_benchmark["category_query"].append(cat_q)
                dic_benchmark["area_index"].append(area.name)
                dic_benchmark["area_name"].append(area["name"])
                dic_benchmark["area_geometry"].append(area.geometry)
                dic_benchmark["function"].append("inside_area")
                dic_benchmark["results_poi_id"].append(list(results.poi_id))
                dic_benchmark["results_poi_name"].append(list(results.poi_name))
                dic_benchmark["results_poi_dist"].append(list(results["dist"]))
                dic_benchmark["results_poi_rank"].append(list(range(1, len(results) + 1)))


    return pd.DataFrame(dic_benchmark)