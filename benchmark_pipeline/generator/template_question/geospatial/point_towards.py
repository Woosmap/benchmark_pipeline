from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from benchmark_pipeline.config import *
from benchmark_pipeline.generator.template_question.registry import template


def towards_b_sql(df, ax, ay, bx, by, half_width=70.0, distance_max=DISTANCE_MAX):
    """Retourne les POIs d'une catégorie situés dans la direction d'un point B depuis un point A.

    Généralise `pack_by_direction` en remplaçant la direction cardinale par
    l'azimut du segment A→B. Le secteur est centré sur cet azimut ; les distances
    restent mesurées depuis A, si bien qu'un POI au-delà de B est retenu tant
    qu'il reste dans le cône.

    Args:
        df (GeoDataFrame): POIs candidats, avec `poi_id`, `poi_name`, `x` et `y`
            en EPSG:2154.
        ax (float): Abscisse Lambert-93 du point A, origine du cône.
        ay (float): Ordonnée Lambert-93 du point A.
        bx (float): Abscisse Lambert-93 du point B, qui donne la direction.
        by (float): Ordonnée Lambert-93 du point B.
        half_width (float): Demi-ouverture du cône, en degrés.
        distance_max (float | None): Max distance to A, in meters; `None` keeps every POI.

    Returns:
        DataFrame: Colonnes `poi_id`, `poi_name`, `dist` (mètres depuis A), `az`,
            `rank`, triées par distance croissante.

    Note:
        Distance euclidienne et azimut planaire, sans correction de latitude :
        c'est la forme correcte en CRS projeté, et le modèle à recopier dans
        `pack_by_direction`.
    """
    dx, dy = df.geometry.x.to_numpy() - ax, df.geometry.y.to_numpy() - ay
    center = (np.degrees(np.arctan2(bx - ax, by - ay)) + 360) % 360
    az = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    dist = np.hypot(dx, dy)
    keep = np.abs((az - center + 540) % 360 - 180) <= half_width
    if distance_max is not None:
        keep &= dist <= distance_max
    kept = np.flatnonzero(keep)
    kept = kept[np.argsort(dist[kept], kind="stable")]
    return pd.DataFrame({
        "poi_id": df["poi_id"].to_numpy()[kept],
        "poi_name": df["poi_name"].to_numpy()[kept],
        "dist": dist[kept],
        "az": az[kept],
        "rank": np.arange(1, len(kept) + 1),
    })

@template("make_question_point_towards")
def make_question_point_towards(df_osm, ratio=None, half_width=70.0, nb_q=110, seed=42):
    """Génère les questions directionnelles « X près de A en allant vers B ».

    Pour chaque ancre A, tire un second POI B parmi ceux situés à moins de 1 000 m,
    via un `cKDTree` construit sur les coordonnées projetées.

    Args:
        df_osm (GeoDataFrame): POIs servant d'ancres, de points B et de cibles.
        half_width (float): Demi-ouverture du cône, en degrés.
        nb_q (int): Nombre de questions visé, stratifié par catégorie interrogée.
        seed (int): Graine du générateur aléatoire.

    Returns:
        DataFrame: Une ligne par question. Mêmes colonnes que
            `make_question_nearsql`, plus `anchor_b_index`, `anchor_b_name`,
            `anchor_b_category`, `anchor_b_x`, `anchor_b_y`.
    """
    xy = np.column_stack([df_osm.geometry.x.values, df_osm.geometry.y.values])   # (n, 2)
    tree = cKDTree(xy)
    dic_benchmark = defaultdict(list)
    rng = np.random.default_rng(seed)
    list_cat = df_osm["category"].unique()
    pid = df_osm["poi_id"].to_numpy()
    # Sampling without replacement: one shuffled pool of distinct anchors per category
    anchor_pools = {cat: df_osm[df_osm["category"] == cat].sample(frac=1, random_state=rng) for cat in list_cat}
    for k in range(nb_q):
        cat_anc = list_cat[k % len(list_cat)]
        rank_in_pool = k // len(list_cat)
        if rank_in_pool >= len(anchor_pools[cat_anc]):
            continue
        anchor = anchor_pools[cat_anc].iloc[rank_in_pool]
        neighbour_positions = [position for position in tree.query_ball_point([anchor.geometry.x, anchor.geometry.y], 1000)
                               if pid[position] != anchor.poi_id]
        if not neighbour_positions:
            continue
        index_b = rng.choice(neighbour_positions)
        anchor_b = df_osm.iloc[index_b]
        results = towards_b_sql(df_osm, anchor.geometry.x, anchor.geometry.y, anchor_b.geometry.x, anchor_b.geometry.y, half_width=half_width)
        results = results[results["poi_id"] != anchor.poi_id]
        results = results[results["poi_id"] != anchor_b.poi_id]

        dic_benchmark["query"].append(f"pois near {anchor.poi_name} towards {anchor_b.poi_name}")
        dic_benchmark["anchor_index"].append(anchor.poi_id)
        dic_benchmark["anchor_name"].append(anchor.poi_name)
        dic_benchmark["anchor_category"].append(anchor.category)
        dic_benchmark["anchor_x"].append(anchor.geometry.x)
        dic_benchmark["anchor_y"].append(anchor.geometry.y)
        dic_benchmark["anchor_b_index"].append(anchor_b.poi_id)
        dic_benchmark["anchor_b_name"].append(anchor_b.poi_name)
        dic_benchmark["anchor_b_category"].append(anchor_b.category)
        dic_benchmark["anchor_b_x"].append(anchor_b.geometry.x)
        dic_benchmark["anchor_b_y"].append(anchor_b.geometry.y)
        dic_benchmark["function"].append("towards_b_sql")
        dic_benchmark["results_poi_id"].append(list(results.poi_id))
        dic_benchmark["results_poi_name"].append(list(results.poi_name))
        dic_benchmark["results_poi_dist"].append(list(results.dist))
        dic_benchmark["results_poi_rank"].append(list(range(1, len(results) + 1)))
        dic_benchmark["anchors_type"].append(["point", "point"])
        dic_benchmark["anchors_name"].append([anchor.poi_name, anchor_b.poi_name])
        dic_benchmark["anchors_id"].append([anchor.poi_id, anchor_b.poi_id])
    return pd.DataFrame(dic_benchmark)