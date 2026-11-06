from collections import defaultdict

import numpy as np
import pandas as pd

from benchmark_pipeline.config import *
from benchmark_pipeline.generator.template_question.registry import template


def near_metric_sql(df, x, y, list_distance):
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
    dist = np.hypot(df.geometry.x.to_numpy() - x, df.geometry.y.to_numpy() - y)
    ordre = np.argsort(dist, kind="stable")
    tailles = np.searchsorted(dist[ordre], list_distance)
    positions = np.concatenate([ordre[:n] for n in tailles]) if len(tailles) else ordre[:0]
    return pd.DataFrame({
        "poi_id": df["poi_id"].to_numpy()[positions],
        "poi_name": df["poi_name"].to_numpy()[positions],
        "dist": dist[positions],
        "distance": np.repeat(list_distance, tailles),
    })

@template("make_question_point_near_metric")
def make_question_point_near_metric(df_osm, ratio=None, list_distance=METRIC_DISTANCES, nb_q=110, seed=42):
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
    # Sampling without replacement: one shuffled pool of distinct anchors per category
    anchor_pools = {cat: df_osm[df_osm["category"] == cat].sample(frac=1, random_state=rng) for cat in list_cat}
    for k in range(nb_q // len(list_distance)):
        cat_anc = list_cat[k % len(list_cat)]
        rank_in_pool = k // len(list_cat)
        if rank_in_pool >= len(anchor_pools[cat_anc]):
            continue
        anchor = anchor_pools[cat_anc].iloc[rank_in_pool]
        results = near_metric_sql(df_osm, anchor.geometry.x, anchor.geometry.y, list_distance)
        results = results[results["poi_id"] != anchor.poi_id]
        for d in list_distance:
            result = results[results["distance"] == d]
            if result.empty:
                continue
            dic_benchmark["query"].append(f"pois at less than {d} meters from {anchor.poi_name}")
            dic_benchmark["anchor_index"].append(anchor.poi_id)
            dic_benchmark["anchor_name"].append(anchor.poi_name)
            dic_benchmark["anchor_category"].append(anchor.category)
            dic_benchmark["anchor_x"].append(anchor.geometry.x)
            dic_benchmark["anchor_y"].append(anchor.geometry.y)
            dic_benchmark["distance"].append(d)
            dic_benchmark["function"].append("near_metric_sql")
            dic_benchmark["results_poi_id"].append(list(result.poi_id))
            dic_benchmark["results_poi_name"].append(list(result.poi_name))
            dic_benchmark["results_poi_dist"].append(list(result.dist))
            dic_benchmark["results_poi_rank"].append(list(range(1, len(result) + 1)))
            dic_benchmark["anchors_type"].append(["point"])
            dic_benchmark["anchors_name"].append([anchor.poi_name])
            dic_benchmark["anchors_id"].append([anchor.poi_id])
    return pd.DataFrame(dic_benchmark)
