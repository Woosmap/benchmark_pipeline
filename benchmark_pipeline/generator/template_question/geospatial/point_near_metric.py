import numpy as np

from benchmark_pipeline.generator.template_question.ratio import allocate
from collections import defaultdict
import pandas as pd

from benchmark_pipeline.config import *
from benchmark_pipeline.generator.template_question.registry import template


def near_metric_sql(df, x, y, list_distance, k=100):
    """Retourne les lots de POIs situés à moins de chacun des rayons d'un point.

    Args:
        df (GeoDataFrame): POIs candidats, avec `poi_id`, `poi_name`, `x` et `y`
            en EPSG:2154.
        x (float): Abscisse Lambert-93 du point de référence.
        y (float): Ordonnée Lambert-93 du point de référence.
        list_distance (list[float]): Rayons maximaux, en mètres.
        k (int): Nombre maximal de résultats par rayon.

    Returns:
        DataFrame: Colonnes `poi_id`, `poi_name`, `dist`, `distance` (le rayon du
            lot), les lots concaténés dans l'ordre de `list_distance` et triés par
            distance croissante. Les rayons étant emboîtés, un POI apparaît dans
            tous les lots dont il respecte le rayon.
    """
    dist = np.hypot(df["x"].to_numpy() - x, df["y"].to_numpy() - y)
    ordre = np.argsort(dist, kind="stable")
    tailles = np.minimum(np.searchsorted(dist[ordre], list_distance), k)
    positions = np.concatenate([ordre[:n] for n in tailles]) if len(tailles) else ordre[:0]
    return pd.DataFrame({
        "poi_id": df["poi_id"].to_numpy()[positions],
        "poi_name": df["poi_name"].to_numpy()[positions],
        "dist": dist[positions],
        "distance": np.repeat(list_distance, tailles),
    })

@template("make_question_point_near_metric")
def make_question_point_near_metric(df_osm, ratio=None, list_distance=[100, 300, 500, 1000], nb_q=110, seed=42):
    """Génère les questions à contrainte métrique « X à moins de D mètres de Y ».

    Args:
        df_osm (GeoDataFrame): POIs servant d'ancres et de cibles.
        list_distance (list[float]): Rayons à décliner, en mètres.
        nb_q (int): Nombre de questions visé, stratifié par catégorie interrogée.
        seed (int): Graine du générateur aléatoire.

    Returns:
        DataFrame: Une ligne par (ancre, rayon). Mêmes colonnes que
            `make_question_nearsql`, plus `distance`.

    """
    dic_benchmark = defaultdict(list)
    rng = np.random.default_rng(seed)
    list_cat = df_osm["category"].unique()
    for k in range(nb_q):
        cat_anc = list_cat[k % len(list_cat)]
        sub = df_osm[df_osm["category"] == cat_anc]
        anchor = sub.sample(1, random_state=rng).iloc[0]
        results = near_metric_sql(df_osm, anchor.x, anchor.y, list_distance)
        results = results[results["poi_id"] != anchor.poi_id]
        for d in list_distance:
            result = results[results["distance"] == d]
            dic_benchmark["query"].append(f"pois at less than {d} meters from {anchor.poi_name}")
            dic_benchmark["anchor_index"].append(anchor.poi_id)
            dic_benchmark["anchor_name"].append(anchor.poi_name)
            dic_benchmark["anchor_category"].append(anchor.category)
            dic_benchmark["anchor_x"].append(anchor.x)
            dic_benchmark["anchor_y"].append(anchor.y)
            dic_benchmark["distance"].append(d)
            dic_benchmark["function"].append("near_metric_sql")
            dic_benchmark["results_poi_id"].append(list(result.poi_id))
            dic_benchmark["results_poi_name"].append(list(result.poi_name))
            dic_benchmark["results_poi_dist"].append(list(result.dist))
            dic_benchmark["results_poi_rank"].append(list(range(1, len(result) + 1)))
    return pd.DataFrame(dic_benchmark)
