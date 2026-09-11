import numpy as np
from collections import defaultdict
import duckdb
import pandas as pd

from src.config import *


def border_area(area, category, df_osm, cat_col="category"):
    """Classe les POIs d'une catégorie par distance à une zone.

    Contrairement à `inside_area` et `outside_area`, reçoit l'ensemble des POIs
    sans restriction préalable.

    Args:
        area (BaseGeometry): Polygone de la zone, en EPSG:2154.
        category (str): Catégorie de POI recherchée.
        df_osm (GeoDataFrame): POIs candidats, non filtrés.
        cat_col (str): Nom de la colonne de catégorie.

    Returns:
        DataFrame: POIs de la catégorie, avec `dist`, triés par distance
            croissante et réindexés.

    TODO: ne mesure pas la distance au bord mais au polygone plein. Tous les POIs
        intérieurs sont donc à distance 0 et occupent la tête du classement dans
        un ordre arbitraire, alors que la question porte précisément sur la
        bordure. Mesurer contre `area.boundary` (ou `area.exterior`), qui est la
        seule géométrie dont la distance est nulle exactement sur le pourtour.
    """
    sel = df_osm[df_osm[cat_col] == category].copy()
    sel["dist"] = sel.geometry.distance(area)
    return sel.sort_values("dist").reset_index(drop=True)

def make_question_area_border(df_osm, df_area, min_size=1000, nb_q=110, seed=42):
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
        DataFrame: Une ligne par question, mêmes colonnes que
            `make_question_insidearea`.

    TODO: hérite du défaut de `border_area` — la vérité terrain est actuellement
        « les POIs intérieurs, en ordre arbitraire », pas « les POIs en bordure ».
    TODO: aucun rayon maximal ; ajouter une bande (`|dist| <= 50 m` autour du
        bord) pour que la relation ait un sens.
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
            results = border_area(area.geometry, cat_q, df_osm)
            if results.empty:
                continue
            nq +=1

            dic_benchmark["query"].append(f"{cat_q} at the border of {area['name_area']}")
            dic_benchmark["category_query"].append(cat_q)
            dic_benchmark["area_index"].append(area.name)
            dic_benchmark["area_name"].append(area["name"])
            dic_benchmark["area_geometry"].append(area.geometry)
            dic_benchmark["function"].append("border_area")
            dic_benchmark["results_poi_id"].append(list(results.poi_id))
            dic_benchmark["results_poi_name"].append(list(results.poi_name))
            dic_benchmark["results_poi_dist"].append(list(results["dist"]))
            dic_benchmark["results_poi_rank"].append(list(range(1, len(results) + 1)))


    return pd.DataFrame(dic_benchmark)