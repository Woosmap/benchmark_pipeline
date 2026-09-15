from scipy.spatial import cKDTree
from scipy.spatial import cKDTree
import numpy as np
from collections import defaultdict
import duckdb
import pandas as pd

from src.config import *
from src.template_question.registry import template


def touching_streets(df_streets, street, tol=1.0):
    """Retourne les rues en contact avec une rue donnée.

    Le critère est métrique et non topologique : toute rue dont la géométrie passe
    à moins de `tol` mètres est retenue, ce qui couvre les nœuds partagés comme
    les extrémités jointives imparfaitement numérisées.

    Args:
        df_streets (GeoDataFrame): Rues en EPSG:2154.
        id_street: Index de la rue de référence dans `df_streets`.
        tol (float): Tolérance de contact, en mètres.

    Returns:
        GeoDataFrame: Sous-ensemble de `df_streets` en contact, la rue de
            référence exclue.
    """
    
    geom = street.geometry
    
    d = df_streets.geometry.distance(geom)
    return df_streets[(d <= tol) & (df_streets['id_street'] != float(street.id_street))]

def cross_streets(df, street_geom_a, street_geom_b, cat, k=100):
    """Retourne les POIs d'une catégorie les plus proches du croisement de deux rues.

    Le croisement est l'intersection géométrique des deux rues.

    Args:
        df (GeoDataFrame): POIs candidats en EPSG:2154.
        street_geom_a (LineString | MultiLineString): Première rue.
        street_geom_b (LineString | MultiLineString): Seconde rue.
        cat (str): Catégorie de POI recherchée.
        k (int): Nombre maximal de résultats retournés.

    Returns:
        DataFrame: Colonnes `poi_id`, `poi_name`, `dist` (mètres au croisement),
            `rank` (1-based), triées par distance croissante.
    """
    inter_point = street_geom_a.intersection(street_geom_b)
    sub = df[df["category"] == cat].copy()
    sub["dist"] = sub.geometry.distance(inter_point)
    out = sub.nsmallest(k, "dist")[["poi_id", "poi_name", "dist"]].reset_index(drop=True)
    out["rank"] = out.index + 1
    return out

@template("make_question_street_cross")
def make_question_street_cross(df_osm, df_streets, nb_q=110, seed=42):
    """Génère les questions de carrefour « X au croisement de R1 et R2 ».

    Pour chaque catégorie cible, tire une rue puis une rue sécante parmi celles en
    contact avec elle. Les rues sans sécante sont ignorées, si bien que le nombre
    de questions effectivement produites peut rester sous le quota.

    Args:
        df_osm (GeoDataFrame): POIs cibles en EPSG:2154.
        df_streets (GeoDataFrame): Rues, avec `name` et `geometry` en EPSG:2154.
        nb_q (int): Nombre total de questions visé, stratifié par catégorie cible.
        seed (int): Graine du générateur aléatoire.

    Returns:
        DataFrame: Une ligne par question. Colonnes `query`, `category_query`,
            `street_a_index`, `street_a_name`, `street_a_geometry`,
            `street_b_index`, `street_b_name`, `street_b_geometry`, `function`,
            plus les quatre colonnes `results_poi_*`.
    """
    dic_benchmark = defaultdict(list)
    rng = np.random.default_rng(seed)
    list_cat = df_osm["category"].unique()
    n_queries_per_stratum = nb_q // len(list_cat)
    for cat_q in list_cat:
        for _ in range(n_queries_per_stratum):
            id_street_a = rng.choice(df_streets['id_street'])
            street_a = df_streets.loc[id_street_a]
            candidates = touching_streets(df_streets, street_a)
            if len(candidates) > 0:
                id_street_b = rng.choice(candidates.index)
                street_b = df_streets.loc[id_street_b]
                results = cross_streets(df_osm, street_a.geometry, street_b.geometry, cat_q)

                dic_benchmark["query"].append(f"{cat_q} at the intersection of  {street_a['street_name']} and {street_b['street_name']}")
                dic_benchmark["category_query"].append(cat_q)
                dic_benchmark["street_a_index"].append(id_street_a)
                dic_benchmark["street_a_name"].append(street_a["street_name"])
                dic_benchmark["street_a_geometry"].append(street_a.geometry)
                dic_benchmark["street_b_index"].append(id_street_b)
                dic_benchmark["street_b_name"].append(street_b["street_name"])
                dic_benchmark["street_b_geometry"].append(street_b.geometry)
                dic_benchmark["function"].append("cross_streets")
                dic_benchmark["results_poi_id"].append(list(results.poi_id))
                dic_benchmark["results_poi_name"].append(list(results.poi_name))
                dic_benchmark["results_poi_dist"].append(list(results.dist))
                dic_benchmark["results_poi_rank"].append(list(results["rank"]))
            else: continue

    return pd.DataFrame(dic_benchmark)

