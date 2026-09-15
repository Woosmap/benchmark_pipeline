import numpy as np
from collections import defaultdict
import duckdb
import pandas as pd

from src.config import *
from src.template_question.registry import template


def direction_area(pois, geom, direction):
    """Ordonne les POIs d'une zone selon un gradient directionnel.

    Restreint aux POIs contenus dans la zone, puis les trie selon la coordonnée
    projetée correspondant à l'axe demandé : l'ordonnée pour nord/sud, l'abscisse
    pour est/ouest, en ordre décroissant vers le nord et vers l'est.

    Args:
        pois (GeoDataFrame): POIs candidats en EPSG:2154.
        geom (BaseGeometry): Polygone de la zone.
        direction (str): "north", "south", "east" ou "west".

    Returns:
        GeoDataFrame: POIs de la zone, avec une colonne `coord`, triés du plus
            conforme à la direction au moins conforme.

    TODO: `direction in "north south"` est un test de sous-chaîne, pas
        d'appartenance à une collection. Il donne le bon résultat par accident
        ici, mais accepterait aussi `"north s"` ou `"h so"`. Idem pour
        `direction in ("south west")`, où l'absence de virgule fait de la
        parenthèse une simple chaîne et non un tuple. Écrire
        `direction in ("north", "south")` et `direction in ("south", "west")`.
    TODO: le tri porte sur la coordonnée absolue, donc « nord du parc » désigne la
        moitié nord de la zone. Rapporter la coordonnée au centroïde de la zone
        rendrait la relation indépendante de la position de la zone dans la bbox.
    """
    sub = pois[pois.within(geom)].copy()
    sub["coord"] = sub.geometry.y if direction in "north south" else sub.geometry.x
    return sub.sort_values("coord", ascending=direction in ("south west"))

@template("make_question_area_direction")
def make_question_area_direction(df_osm, df_area, list_direction=["north", "east", "west", "south"], min_size=1000, nb_q=110, seed=42):
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

    TODO: la condition d'arrêt `while nq != n_queries_per_stratum` ne peut pas
        être satisfaite. La boucle interne incrémente `nq` d'un par direction, soit
        jusqu'à quatre par tour, si bien que `nq` saute par-dessus le quota — pour
        10 questions et 4 directions il vaut 0, 4, 8, 12 et n'atteint jamais 10. La
        boucle ne s'arrête donc que sur le plafond `i < 200`, produisant environ
        200 questions par catégorie au lieu de 10. Écrire `while nq < n_...`, et
        tirer la zone hors de la boucle des directions pour que les quatre
        questions portent sur la même zone.
    TODO: schéma de sortie divergent — `results_poi_x`/`results_poi_y` au lieu de
        `results_poi_dist`, ce qui casse la concaténation avec les autres
        générateurs et l'outillage qui lit `results_poi_dist`.
    TODO: `results_poi_rank` vaut `range(len(results))`, donc 0-based.
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
            for direction in list_direction:
                i+=1
                area = areas.loc[rng.choice(areas.index)]
                df_poi_in_area = df_osm[df_osm.geometry.within(area.geometry)]
                pois = df_poi_in_area[df_poi_in_area["category"]==cat_q]
                if pois.empty:
                    continue
                else:
                    results = direction_area(pois, area.geometry, direction)
                    nq +=1

                    dic_benchmark["query"].append(f"{cat_q} at the {direction} of {area['area_name']}")
                    dic_benchmark["category_query"].append(cat_q)
                    dic_benchmark["area_index"].append(area.name)
                    dic_benchmark["area_name"].append(area["area_name"])
                    dic_benchmark["area_geometry"].append(area.geometry)
                    dic_benchmark["function"].append("direction_area")
                    dic_benchmark["direction"].append(direction)
                    dic_benchmark["results_poi_id"].append(list(results.poi_id))
                    dic_benchmark["results_poi_x"].append(list(results.geometry.x))
                    dic_benchmark["results_poi_y"].append(list(results.geometry.y))
                    dic_benchmark["results_poi_name"].append(list(results.poi_name))
                    dic_benchmark["results_poi_rank"].append(list(range(1, len(results) + 1)))


    return pd.DataFrame(dic_benchmark)