import numpy as np
from collections import defaultdict
import duckdb
import pandas as pd

from src.config import *
from src.template_question.registry import template


def near_metric_sql(df, x, y, cat, distance, k=100):
    """Retourne les POIs d'une catégorie situés à moins d'une distance donnée d'un point.

    Variante bornée de `near_sql` : le seuil métrique est appliqué dans la clause
    WHERE, en mètres Lambert-93. Le résultat peut donc être vide, contrairement à
    une recherche par k plus proches voisins.

    Args:
        df (GeoDataFrame): POIs candidats en EPSG:2154.
        x (float): Abscisse Lambert-93 du point de référence.
        y (float): Ordonnée Lambert-93 du point de référence.
        cat (str): Catégorie de POI recherchée.
        distance (float): Rayon maximal, en mètres.
        k (int): Nombre maximal de résultats retournés.

    Returns:
        DataFrame: Colonnes `poi_id`, `poi_name`, `dist`, triées par distance
            croissante ; vide si aucun POI n'entre dans le rayon.
    """
    con = duckdb.connect()
    con.sql("INSTALL spatial; LOAD spatial;")
    df = pd.DataFrame(df.drop(columns="geometry")).assign(geom_wkb=df.geometry.to_wkb())

    results = con.execute("""
        SELECT poi_id, poi_name,
                ST_Distance(ST_GeomFromWKB(geom_wkb), ST_Point($x, $y))::DOUBLE AS dist
        FROM df
        WHERE category = $cat AND dist < $d
        ORDER BY dist
        LIMIT $k
    """, {"x": x, "y": y, "cat": cat, "k": k, "d": distance}).df()

    return results

@template("make_question_point_near_metric")
def make_question_point_near_metric(df_osm, list_distance=[100, 300, 500, 1000], nb_q=110, seed=42):
    """Génère les questions à contrainte métrique « X à moins de D mètres de Y ».

    Reprend l'échantillonnage stratifié de `make_question_nearsql` et décline
    chaque couple (ancre, catégorie cible) sur tous les rayons demandés.

    Args:
        df_osm (GeoDataFrame): POIs servant d'ancres et de cibles.
        list_distance (list[float]): Rayons à décliner, en mètres.
        nb_q (int): Nombre d'ancres visé, stratifié par catégorie.
        seed (int): Graine du générateur aléatoire.

    Returns:
        DataFrame: Une ligne par (ancre, rayon). Mêmes colonnes que
            `make_question_nearsql`, plus `distance`.

    """
    dic_benchmark = defaultdict(list)
    rng = np.random.default_rng(seed)
    list_cat = df_osm["category"].unique()
    n_queries_per_stratum = nb_q // (len(list_cat)+len(list_distance))
    for cat_anc in list_cat:
        sub = df_osm[df_osm['category'] == cat_anc]
        anchors = sub.sample(n=min(n_queries_per_stratum, len(sub)), random_state=rng)
        for anchor in anchors.itertuples():
            cat_q = rng.choice(list_cat)
            for d in list_distance:
                results = near_metric_sql(df_osm, anchor.x, anchor.y, cat_q, distance=d)
                if cat_anc == cat_q:
                    results = results[results["poi_id"] != anchor.poi_id]

                dic_benchmark["query"].append(f"{cat_q} at less than {d} meters from {anchor.poi_name}")
                dic_benchmark["anchor_index"].append(anchor.poi_id)
                dic_benchmark["anchor_name"].append(anchor.poi_name)
                dic_benchmark["anchor_category"].append(anchor.category)
                dic_benchmark["anchor_x"].append(anchor.x)
                dic_benchmark["anchor_y"].append(anchor.y)
                dic_benchmark["category_query"].append(cat_q)
                dic_benchmark["same_cat"].append(cat_anc==cat_q)
                dic_benchmark["distance"].append(d)
                dic_benchmark["function"].append("near_metric_sql")
                dic_benchmark["results_poi_id"].append(list(results.poi_id))
                dic_benchmark["results_poi_name"].append(list(results.poi_name))
                dic_benchmark["results_poi_dist"].append(list(results.dist))
                dic_benchmark["results_poi_rank"].append(list(range(1, len(results) + 1)))
    return pd.DataFrame(dic_benchmark)
