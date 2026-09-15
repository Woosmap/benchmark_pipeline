import numpy as np
from collections import defaultdict
import duckdb
import pandas as pd

from src.config import *
from src.template_question.registry import template


def near_sql(df, x, y, cat, k=100):
    """Retourne les `k` POIs d'une catégorie les plus proches d'un point.

    Exécute une recherche par distance croissante dans DuckDB avec l'extension
    spatiale. Les coordonnées et les géométries sont en Lambert-93 (EPSG:2154),
    donc les distances sont euclidiennes et exprimées en mètres.

    Args:
        df (GeoDataFrame): POIs candidats, avec `poi_id`, `poi_name`, `category`
            et une colonne `geometry` en EPSG:2154.
        x (float): Abscisse Lambert-93 du point de référence.
        y (float): Ordonnée Lambert-93 du point de référence.
        cat (str): Catégorie de POI recherchée.
        k (int): Nombre maximal de résultats retournés.

    Returns:
        DataFrame: Colonnes `poi_id`, `poi_name`, `dist` (mètres), triées par
            distance croissante.

    TODO: réutiliser une connexion DuckDB unique au lieu d'en ouvrir une et de
        rejouer `INSTALL spatial` à chaque appel.
    """
    con = duckdb.connect()
    con.sql("INSTALL spatial; LOAD spatial;")
    df = pd.DataFrame(df.drop(columns="geometry")).assign(geom_wkb=df.geometry.to_wkb())
    results = con.execute("""
        SELECT poi_id, poi_name,
                ST_Distance(ST_GeomFromWKB(geom_wkb), ST_Point($x, $y))::DOUBLE AS dist
        FROM df
        WHERE category = $cat
        ORDER BY dist
        LIMIT $k
    """, {"x": x, "y": y, "cat": cat, "k": k}).df()
    return results

@template("make_question_point_near")
def make_question_point_near(df_osm, nb_q=110, seed=42):
    """Génère les questions de proximité simple « X près de Y ».

    Pour chaque catégorie d'ancre, tire un quota d'ancres puis, pour chacune, une
    catégorie cible au hasard. La vérité terrain est le classement des POIs de la
    catégorie cible par distance à l'ancre. L'ancre est retirée de ses propres
    résultats lorsque les deux catégories coïncident.

    Args:
        df_osm (GeoDataFrame): POIs servant à la fois d'ancres et de cibles.
        nb_q (int): Nombre total de questions visé, stratifié par catégorie d'ancre.
        seed (int): Graine du générateur aléatoire.

    Returns:
        DataFrame: Une ligne par question. Contexte `query`, `anchor_index`,
            `anchor_name`, `anchor_category`, `anchor_x`, `anchor_y`,
            `category_query`, `same_cat`, `function` ; vérité terrain
            `results_poi_id`, `results_poi_name`, `results_poi_dist`,
            `results_poi_rank`, listes parallèles ordonnées par pertinence
            décroissante.

    """
    dic_benchmark = defaultdict(list)
    rng = np.random.default_rng(seed)
    list_cat = df_osm["category"].unique()
    n_queries_per_stratum = nb_q // len(list_cat)
    for cat_anc in list_cat:
        sub = df_osm[df_osm['category'] == cat_anc]
        anchors = sub.sample(n=min(n_queries_per_stratum, len(sub)), random_state=rng)
        for anchor in anchors.itertuples():
            cat_q = rng.choice(list_cat)
            results = near_sql(df_osm, anchor.x, anchor.y, cat_q)
            if cat_anc == cat_q:
                results = results[results["poi_id"] != anchor.poi_id]

            dic_benchmark["query"].append(f"{cat_q} near {anchor.poi_name}")
            dic_benchmark["anchor_index"].append(anchor.poi_id)
            dic_benchmark["anchor_name"].append(anchor.poi_name)
            dic_benchmark["anchor_category"].append(anchor.category)
            dic_benchmark["anchor_x"].append(anchor.x)
            dic_benchmark["anchor_y"].append(anchor.y)
            dic_benchmark["category_query"].append(cat_q)
            dic_benchmark["same_cat"].append(cat_anc==cat_q)
            dic_benchmark["function"].append("near_sql")
            dic_benchmark["results_poi_id"].append(list(results.poi_id))
            dic_benchmark["results_poi_name"].append(list(results.poi_name))
            dic_benchmark["results_poi_dist"].append(list(results.dist))
            dic_benchmark["results_poi_rank"].append(list(range(1, len(results) + 1)))
    return pd.DataFrame(dic_benchmark)
