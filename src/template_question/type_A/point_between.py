from scipy.spatial import cKDTree
from scipy.spatial import cKDTree
import numpy as np
from collections import defaultdict
import duckdb
import pandas as pd

from src.config import *
from src.template_question.registry import template


def between_ab_sql(df, ax, ay, bx, by, cat, k=100,
                   corridor_m=200.0, con=None):
    """Retourne les POIs d'une catégorie situés dans le corridor reliant deux points.

    Projette chaque POI dans le repère du segment A→B : `along` est l'abscisse
    curviligne le long du segment, `cross_m` l'écart perpendiculaire. Un POI est
    retenu si sa projection tombe entre A et B et si son écart latéral n'excède
    pas `corridor_m`. Le classement privilégie la proximité à l'axe, puis
    l'avancement le long du segment.

    Args:
        df (GeoDataFrame): POIs candidats en EPSG:2154.
        ax (float): Abscisse Lambert-93 du point A.
        ay (float): Ordonnée Lambert-93 du point A.
        bx (float): Abscisse Lambert-93 du point B.
        by (float): Ordonnée Lambert-93 du point B.
        cat (str): Catégorie de POI recherchée.
        k (int): Nombre maximal de résultats retournés.
        corridor_m (float): Demi-largeur du corridor, en mètres.
        con (duckdb.DuckDBPyConnection | None): Connexion à réutiliser.

    Returns:
        DataFrame: Colonnes `poi_id`, `poi_name`, `dist_a`, `dist_b`, `along`,
            `cross_m`, triées par écart latéral puis avancement.

    TODO: `ab_len` vaut 0 si A et B coïncident, ce qui produit une division par
        zéro et des NaN silencieux ; garder le cas ou l'écarter en amont.
    """
    if con is None:
        con = duckdb.connect()
        con.sql("INSTALL spatial; LOAD spatial;")
    flat = pd.DataFrame(df.drop(columns="geometry")).assign(geom_wkb=df.geometry.to_wkb())
    con.register("poi", flat)

    return con.execute("""
        WITH ab AS (
            SELECT $bx - $ax AS abx, $by - $ay AS aby,
                   sqrt(pow($bx - $ax, 2) + pow($by - $ay, 2)) AS ab_len
        ), g AS (
            SELECT poi_id, poi_name, ST_GeomFromWKB(geom_wkb) AS geom
            FROM poi WHERE category = $cat
        ), p AS (
            SELECT poi_id, poi_name,
                   ((ST_X(geom) - $ax) * ab.abx + (ST_Y(geom) - $ay) * ab.aby) / ab.ab_len AS along,
                   abs((ST_X(geom) - $ax) * ab.aby - (ST_Y(geom) - $ay) * ab.abx) / ab.ab_len AS cross_m,
                   sqrt(pow(ST_X(geom) - $ax, 2) + pow(ST_Y(geom) - $ay, 2)) AS dist_a,
                   sqrt(pow(ST_X(geom) - $bx, 2) + pow(ST_Y(geom) - $by, 2)) AS dist_b,
                   ab.ab_len
            FROM g CROSS JOIN ab
        )
        SELECT poi_id, poi_name, dist_a, dist_b, along, cross_m,
               row_number() OVER (ORDER BY cross_m, along) AS rank
        FROM p
        WHERE along BETWEEN 0 AND ab_len
          AND cross_m <= $corr
        ORDER BY cross_m, along
        LIMIT $k
    """, {"ax": ax, "ay": ay, "bx": bx, "by": by, "cat": cat, "k": k,
          "corr": corridor_m}).df()

@template("make_question_point_between")
def make_question_point_between(df_osm, corridor_m=200.0, nb_q=110, seed=42):
    """Génère les questions d'entre-deux « X entre A et B ».

    Même tirage du point B que `make_question_towardsb` : un POI à moins de
    1 000 m de l'ancre. La vérité terrain est ordonnée par distance à l'axe A-B,
    et non par distance à l'ancre.

    Args:
        df_osm (GeoDataFrame): POIs servant d'ancres, de points B et de cibles.
        corridor_m (float): Demi-largeur du corridor, en mètres.
        nb_q (int): Nombre de questions visé, stratifié par catégorie d'ancre.
        seed (int): Graine du générateur aléatoire.

    Returns:
        DataFrame: Une ligne par question. Mêmes colonnes que
            `make_question_towardsb`, la distance rapportée étant l'écart latéral.

    """
    xy = np.column_stack([df_osm.x.values, df_osm.y.values])   # (n, 2)
    tree = cKDTree(xy)
    dic_benchmark = defaultdict(list)
    rng = np.random.default_rng(seed)
    list_cat = df_osm["category"].unique()
    n_queries_per_stratum = nb_q // len(list_cat)
    pid = df_osm["poi_id"].to_numpy()  
    for cat_anc in list_cat:
        sub = df_osm[df_osm['category'] == cat_anc]
        anchors = sub.sample(n=min(n_queries_per_stratum, len(sub)), random_state=rng)
        for anchor in anchors.itertuples():

            index_b = rng.choice([k for k in tree.query_ball_point([anchor.x, anchor.y], 1000)
                                if pid[k] != anchor.poi_id])
            point_b = df_osm.iloc[index_b]
            cat_q = rng.choice(list_cat)
            
            results = between_ab_sql(df_osm, anchor.x, anchor.y, point_b.x, point_b.y, cat_q, corridor_m=corridor_m)
            if cat_anc == cat_q:
                results = results[results["poi_id"] != anchor.poi_id]
            if cat_q == point_b.category:
                results = results[results["poi_id"] != point_b.poi_id]

            dic_benchmark["query"].append(f"{cat_q} between {anchor.poi_name} and {point_b.poi_name}")
            dic_benchmark["anchor_index"].append(anchor.poi_id)
            dic_benchmark["anchor_name"].append(anchor.poi_name)
            dic_benchmark["anchor_category"].append(anchor.category)
            dic_benchmark["anchor_x"].append(anchor.x)
            dic_benchmark["anchor_y"].append(anchor.y)
            dic_benchmark["point_b_index"].append(point_b.poi_id)
            dic_benchmark["point_b_name"].append(point_b.poi_name)
            dic_benchmark["point_b_category"].append(point_b.category)
            dic_benchmark["point_b_x"].append(point_b.x)
            dic_benchmark["point_b_y"].append(point_b.y)
            dic_benchmark["category_query"].append(cat_q)
            dic_benchmark["same_cat"].append(cat_anc==cat_q)
            dic_benchmark["function"].append("between_ab_sql")
            dic_benchmark["results_poi_id"].append(list(results.poi_id))
            dic_benchmark["results_poi_name"].append(list(results.poi_name))

    return pd.DataFrame(dic_benchmark)