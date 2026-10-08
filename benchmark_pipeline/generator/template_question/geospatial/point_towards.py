from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from benchmark_pipeline.config import *
from benchmark_pipeline.generator.template_question.registry import template


def towards_b_sql(df, ax, ay, bx, by, k=100,
                  half_width=70.0, con=None):
    """Retourne les POIs d'une catégorie situés dans la direction d'un point B depuis un point A.

    Généralise `pack_by_direction` en remplaçant la direction cardinale par
    l'azimut du segment A→B. Le secteur est centré sur cet azimut ; les distances
    restent mesurées depuis A, si bien qu'un POI au-delà de B est retenu tant
    qu'il reste dans le cône.

    Args:
        df (GeoDataFrame): POIs candidats en EPSG:2154.
        ax (float): Abscisse Lambert-93 du point A, origine du cône.
        ay (float): Ordonnée Lambert-93 du point A.
        bx (float): Abscisse Lambert-93 du point B, qui donne la direction.
        by (float): Ordonnée Lambert-93 du point B.
        k (int): Nombre maximal de résultats retournés.
        half_width (float): Demi-ouverture du cône, en degrés.
        con (duckdb.DuckDBPyConnection | None): Connexion à réutiliser.

    Returns:
        DataFrame: Colonnes `poi_id`, `poi_name`, `dist` (mètres depuis A), `az`,
            triées par distance croissante.

    Note:
        Distance euclidienne et azimut planaire, sans correction de latitude :
        c'est la forme correcte en CRS projeté, et le modèle à recopier dans
        `pack_by_direction`.
    """
    if con is None:
        con = duckdb.connect()
        con.sql("INSTALL spatial; LOAD spatial;")
    flat = pd.DataFrame(df.drop(columns="geometry")).assign(geom_wkb=df.geometry.to_wkb())
    con.register("poi", flat)

    return con.execute("""
        WITH ab AS (
            SELECT (degrees(atan2($bx - $ax, $by - $ay)) + 360) % 360 AS center
        ), g AS (
            SELECT poi_id, poi_name, ST_GeomFromWKB(geom_wkb) AS geom
            FROM poi 
        ), d AS (
            SELECT poi_id, poi_name,
                   sqrt(pow(ST_X(geom) - $ax, 2) + pow(ST_Y(geom) - $ay, 2)) AS dist,
                   (degrees(atan2(ST_X(geom) - $ax, ST_Y(geom) - $ay)) + 360) % 360 AS az
            FROM g
        )
        SELECT d.poi_id, d.poi_name, d.dist, d.az,
               row_number() OVER (ORDER BY d.dist) AS rank
        FROM d CROSS JOIN ab
        WHERE abs(((d.az - ab.center + 540)::DOUBLE % 360) - 180) <= $half
        ORDER BY d.dist
        LIMIT $k
    """, {"ax": ax, "ay": ay, "bx": bx, "by": by, "k": k,
          "half": half_width}).df()

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
    xy = np.column_stack([df_osm.x.values, df_osm.y.values])   # (n, 2)
    tree = cKDTree(xy)
    dic_benchmark = defaultdict(list)
    rng = np.random.default_rng(seed)
    list_cat = df_osm["category"].unique()
    pid = df_osm["poi_id"].to_numpy()
    for k in range(nb_q):
        cat_anc = list_cat[k % len(list_cat)]
        sub = df_osm[df_osm["category"] == cat_anc]
        anchor = sub.sample(1, random_state=rng).iloc[0]
        index_b = rng.choice([k for k in tree.query_ball_point([anchor.x, anchor.y], 1000)
                                        if pid[k] != anchor.poi_id])
        anchor_b = df_osm.iloc[index_b]
        results = towards_b_sql(df_osm, anchor.x, anchor.y, anchor_b.x, anchor_b.y, half_width=half_width)
        results = results[results["poi_id"] != anchor.poi_id]
        results = results[results["poi_id"] != anchor_b.poi_id]

        dic_benchmark["query"].append(f"pois near {anchor.poi_name} towards {anchor_b.poi_name}")
        dic_benchmark["anchor_index"].append(anchor.poi_id)
        dic_benchmark["anchor_name"].append(anchor.poi_name)
        dic_benchmark["anchor_category"].append(anchor.category)
        dic_benchmark["anchor_x"].append(anchor.x)
        dic_benchmark["anchor_y"].append(anchor.y)
        dic_benchmark["anchor_b_index"].append(anchor_b.poi_id)
        dic_benchmark["anchor_b_name"].append(anchor_b.poi_name)
        dic_benchmark["anchor_b_category"].append(anchor_b.category)
        dic_benchmark["anchor_b_x"].append(anchor_b.x)
        dic_benchmark["anchor_b_y"].append(anchor_b.y)
        dic_benchmark["function"].append("towards_b_sql")
        dic_benchmark["results_poi_id"].append(list(results.poi_id))
        dic_benchmark["results_poi_name"].append(list(results.poi_name))
        dic_benchmark["results_poi_dist"].append(list(results.dist))
        dic_benchmark["results_poi_rank"].append(list(range(1, len(results) + 1)))
    return pd.DataFrame(dic_benchmark)