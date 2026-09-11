from scipy.spatial import cKDTree
from scipy.spatial import cKDTree
from scipy.spatial import cKDTree
import numpy as np
from collections import defaultdict
import duckdb
import pandas as pd

from src.config import *


def along_street(df, street_geom, cat, k=100):
    """Retourne les POIs d'une catégorie les plus proches d'une rue.

    La distance est mesurée du POI à la géométrie de la rue, pas à un point : pour
    une rue en plusieurs tronçons, c'est la distance au tronçon le plus proche qui
    est retenue.

    Args:
        df (GeoDataFrame): POIs candidats en EPSG:2154.
        street_geom (LineString | MultiLineString): Géométrie de la rue.
        cat (str): Catégorie de POI recherchée.
        k (int): Nombre maximal de résultats retournés.

    Returns:
        DataFrame: Colonnes `poi_id`, `poi_name`, `dist` (mètres), `rank`
            (1-based), triées par distance croissante.

    TODO: aucun rayon maximal. « Le long de la rue » implique un couloir étroit ;
        sans borne, les 100 plus proches peuvent être à des centaines de mètres,
        surtout pour les catégories rares. Ajouter `max_dist` (~100 m).
    """
    sub = df[df["category"] == cat].copy()
    sub["dist"] = sub.geometry.distance(street_geom)
    out = sub.nsmallest(k, "dist")[["poi_id", "poi_name", "dist"]].reset_index(drop=True)
    out["rank"] = out.index + 1
    return out

def make_question_street_along(df_osm, df_streets, min_length=500, nb_q=110, seed=42):
    """Génère les questions de linéaire « X le long de la rue R ».

    Contrairement aux générateurs à ancre POI, la stratification porte ici sur la
    catégorie cible : pour chaque catégorie, un quota de rues est tiré au hasard.

    Args:
        df_osm (GeoDataFrame): POIs cibles en EPSG:2154.
        df_streets (GeoDataFrame): Rues, avec `name` et `geometry` en EPSG:2154.
        nb_q (int): Nombre total de questions visé, stratifié par catégorie cible.
        seed (int): Graine du générateur aléatoire.

    Returns:
        DataFrame: Une ligne par question. Colonnes `query`, `category_query`,
            `street_index`, `street_name`, `street_geometry`, `function`, plus les
            quatre colonnes `results_poi_*`.

    TODO: la question n'est résolvable que si le nom de rue désigne une voie
        unique dans l'emprise. Tirer uniquement des rues dont le libellé est
        unique, ou lever l'ambiguïté en nommant la commune dans la question.
    TODO: tirage avec remise (`rng.choice` sur l'index à chaque tour) : une même
        rue peut produire deux questions identiques dans la même strate.
    TODO: aucun filtre sur les résultats vides ou quasi vides ; les catégories
        rares produisent des questions à un seul POI très lointain.
    """
    dic_benchmark = defaultdict(list)
    rng = np.random.default_rng(seed)
    df_streets = df_streets[df_streets.geometry.length > min_length]
    list_cat = df_osm["category"].unique()
    n_queries_per_stratum = nb_q // len(list_cat)
    for cat_q in list_cat:
        for _ in range(n_queries_per_stratum):
            id_street = rng.choice(df_streets.index)
            street = df_streets.loc[id_street]
            results = along_street(df_osm, street.geometry, cat_q)

            dic_benchmark["query"].append(f"{cat_q} following {street['street_name']}")
            dic_benchmark["category_query"].append(cat_q)
            dic_benchmark["street_index"].append(id_street)
            dic_benchmark["street_name"].append(street["name"])
            dic_benchmark["street_geometry"].append(street.geometry)
            dic_benchmark["function"].append("along_street")
            dic_benchmark["results_poi_id"].append(list(results.poi_id))
            dic_benchmark["results_poi_name"].append(list(results.poi_name))
            dic_benchmark["results_poi_dist"].append(list(results.dist))
            dic_benchmark["results_poi_rank"].append(list(results["rank"]))

    return pd.DataFrame(dic_benchmark)