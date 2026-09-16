from scipy.spatial import cKDTree
from scipy.spatial import cKDTree
import numpy as np
from collections import defaultdict
import duckdb
import pandas as pd

from src.config import *
from src.template_question.registry import template
from src.template_question.ratio import allocate


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
def make_question_point_between(df_osm, ratio_cat_q=None, corridor_m=200.0, nb_q=110, seed=42, max_try=200):
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
    list_cat = df_osm["category"].unique()
    if ratio_cat_q:
        balance = allocate(nb_q, ratio_cat_q)
    else: balance = allocate(nb_q, {cat: 1 for cat in list_cat})

    xy = np.column_stack([df_osm.x.values, df_osm.y.values])   # (n, 2)
    tree = cKDTree(xy)
    dic_benchmark = defaultdict(list)
    rng = np.random.default_rng(seed)
  
    pid = df_osm["poi_id"].to_numpy()  
    for cat_q, nb_q_cat in balance.items():
        n=0
        nb_try = 0
        while n < nb_q_cat and nb_try < max_try:
            poi_answer = df_osm[df_osm['category'] == cat_q].sample(1, random_state=rng).iloc[0]
            for cat_anc in list_cat:
                #sub = df_osm[df_osm['category'] == cat_anc]
                id_anchor_a = rng.choice([k for k in tree.query_ball_point([poi_answer.x, poi_answer.y], 1000)
                                        if pid[k] != poi_answer.poi_id])
                anchor_a = df_osm.iloc[id_anchor_a]

                x_b, y_b = -(anchor_a.geometry.x - 2*poi_answer.geometry.x), -(anchor_a.geometry.y - 2*poi_answer.geometry.y)
                list_id_anchor_b = [k for k in tree.query_ball_point([x_b, y_b], 400) if pid[k] != anchor_a.poi_id]
                if len(list_id_anchor_b) > 0:
                    id_anchors_b = rng.choice(list_id_anchor_b)
                anchor_b = df_osm.iloc[id_anchors_b]   

                results = between_ab_sql(df_osm, anchor_a.x, anchor_a.y, anchor_b.x, anchor_b.y, cat_q, corridor_m=corridor_m)
                if len(results) >0:
                    if cat_anc == cat_q:
                        results = results[results["poi_id"] != anchor_a.poi_id]
                    if cat_q == anchor_b.category:
                        results = results[results["poi_id"] != anchor_b.poi_id]

                    dic_benchmark["query"].append(f"{cat_q} between {anchor_a.poi_name} and {anchor_b.poi_name}")
                    dic_benchmark["anchor_a_index"].append(anchor_a.poi_id)
                    dic_benchmark["anchor_a_name"].append(anchor_a.poi_name)
                    dic_benchmark["anchor_a_category"].append(anchor_a.category)
                    dic_benchmark["anchor_a_x"].append(anchor_a.x)
                    dic_benchmark["anchor_a_y"].append(anchor_a.y)
                    dic_benchmark["anchor_b_index"].append(anchor_b.poi_id)
                    dic_benchmark["anchor_b_name"].append(anchor_b.poi_name)
                    dic_benchmark["anchor_b_category"].append(anchor_b.category)
                    dic_benchmark["anchor_b_x"].append(anchor_b.x)
                    dic_benchmark["anchor_b_y"].append(anchor_b.y)
                    dic_benchmark["category_query"].append(cat_q)
                    dic_benchmark["same_cat"].append(cat_anc==cat_q)
                    dic_benchmark["function"].append("between_ab_sql")
                    dic_benchmark["results_poi_id"].append(list(results.poi_id))
                    dic_benchmark["results_poi_name"].append(list(results.poi_name))
                    dic_benchmark["results_poi_dist"].append(list(results.cross_m))
                    dic_benchmark["results_poi_rank"].append(list(range(1, len(results) + 1)))
                    n += 1
                    if n > nb_q_cat:
                        break
                else: nb_try += 1

    return pd.DataFrame(dic_benchmark)