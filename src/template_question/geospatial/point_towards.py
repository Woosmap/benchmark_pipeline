from scipy.spatial import cKDTree
import numpy as np

from src.template_question.ratio import allocate
from collections import defaultdict
import duckdb
import pandas as pd

from src.config import *
from src.template_question.registry import template


def towards_b_sql(df, ax, ay, bx, by, cat, k=100,
                  half_width=70.0, con=None):
    """Retourne les POIs d'une catégorie situés dans la direction d'un point B depuis un point A.

    Généralise `cardinal_azimuth_sql` en remplaçant la direction cardinale par
    l'azimut du segment A→B. Le secteur est centré sur cet azimut ; les distances
    restent mesurées depuis A, si bien qu'un POI au-delà de B est retenu tant
    qu'il reste dans le cône.

    Args:
        df (GeoDataFrame): POIs candidats en EPSG:2154.
        ax (float): Abscisse Lambert-93 du point A, origine du cône.
        ay (float): Ordonnée Lambert-93 du point A.
        bx (float): Abscisse Lambert-93 du point B, qui donne la direction.
        by (float): Ordonnée Lambert-93 du point B.
        cat (str): Catégorie de POI recherchée.
        k (int): Nombre maximal de résultats retournés.
        half_width (float): Demi-ouverture du cône, en degrés.
        con (duckdb.DuckDBPyConnection | None): Connexion à réutiliser.

    Returns:
        DataFrame: Colonnes `poi_id`, `poi_name`, `dist` (mètres depuis A), `az`,
            triées par distance croissante.

    Note:
        Distance euclidienne et azimut planaire, sans correction de latitude :
        c'est la forme correcte en CRS projeté, et le modèle à recopier dans
        `cardinal_azimuth_sql`.
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
            FROM poi WHERE category = $cat
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
    """, {"ax": ax, "ay": ay, "bx": bx, "by": by, "cat": cat, "k": k,
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
            `make_question_nearsql`, plus `point_b_index`, `point_b_name`,
            `point_b_category`, `point_b_x`, `point_b_y`.
    """
    xy = np.column_stack([df_osm.x.values, df_osm.y.values])   # (n, 2)
    tree = cKDTree(xy)
    dic_benchmark = defaultdict(list)
    rng = np.random.default_rng(seed)
    list_cat = df_osm["category"].unique()
    pid = df_osm["poi_id"].to_numpy()
    # Une strate par CELLULE (catégorie d'ancre × catégorie interrogée) : les
    # deux marges sont donc équilibrées, et la somme des cellules vaut nb_q.
    # Allouer sur les cellules plutôt que diviser nb_q évite le plancher du
    # produit cartésien, qui ne peut pas descendre sous une question par cellule.
    # Les cellules sont rangées en diagonales cycliques : chaque diagonale
    # touche une fois chaque catégorie d'ancre ET une fois chaque catégorie
    # interrogée. Le reste de la division d'`allocate`, qui va toujours aux
    # premières clés, se répartit donc également sur les deux marges — rangées
    # par `product`, elles retomberaient toutes sur les deux mêmes ancres.
    n_cat = len(list_cat)
    cellules = [(list_cat[i], list_cat[(i + j) % n_cat])
                for j in range(n_cat) for i in range(n_cat)]
    poids = {c: (ratio or {}).get(c[0], 1) for c in cellules}
    for (cat_anc, cat_q), nb_cell in allocate(nb_q, poids).items():
        if not nb_cell:
            continue
        sub = df_osm[df_osm["category"] == cat_anc]
        anchors = sub.sample(n=min(nb_cell, len(sub)), random_state=rng)
        for anchor in anchors.itertuples():
            index_b = rng.choice([k for k in tree.query_ball_point([anchor.x, anchor.y], 1000)
                                            if pid[k] != anchor.poi_id])
            point_b = df_osm.iloc[index_b]
            
            results = towards_b_sql(df_osm, anchor.x, anchor.y, point_b.x, point_b.y, cat_q, half_width=half_width)
            if cat_anc == cat_q:
                results = results[results["poi_id"] != anchor.poi_id]
            if cat_q == point_b.category:
                results = results[results["poi_id"] != point_b.poi_id]

            dic_benchmark["query"].append(f"{cat_q} near {anchor.poi_name} towards {point_b.poi_name}")
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
            dic_benchmark["function"].append("towards_b_sql")
            dic_benchmark["results_poi_id"].append(list(results.poi_id))
            dic_benchmark["results_poi_name"].append(list(results.poi_name))
            dic_benchmark["results_poi_dist"].append(list(results.dist))
            dic_benchmark["results_poi_rank"].append(list(range(1, len(results) + 1)))

    return pd.DataFrame(dic_benchmark)