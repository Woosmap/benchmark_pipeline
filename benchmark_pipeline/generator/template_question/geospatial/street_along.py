from collections import defaultdict

import numpy as np
import pandas as pd

from benchmark_pipeline.config import *
from benchmark_pipeline.generator.template_question.registry import template


def along_street(df, street_geom, max_dist, k=100):
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
    """
    sub = df.copy()
    sub["dist"] = sub.geometry.distance(street_geom)
    sub = sub[sub["dist"] < max_dist]
    out = sub.nsmallest(k, "dist")[["poi_id", "poi_name", "dist"]].reset_index(drop=True)
    out["rank"] = out.index + 1
    return out

@template("make_question_street_along")
def make_question_street_along(df_osm, df_streets, max_dist=50, min_length=500, nb_q=110, seed=42):
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
    """
    dic_benchmark = defaultdict(list)
    rng = np.random.default_rng(seed)
    df_streets = df_streets[df_streets.geometry.length > min_length]
    id_streets = rng.choice(df_streets.index, nb_q)
    for id_street in id_streets:
        street = df_streets.loc[id_street]
        results = along_street(df_osm, street.geometry, max_dist)
        dic_benchmark["query"].append(f"pois following {street['street_name']}")
        dic_benchmark["street_index"].append(id_street)
        dic_benchmark["street_name"].append(street["street_name"])
        dic_benchmark["street_geometry"].append(street.geometry)
        dic_benchmark["function"].append("along_street")
        dic_benchmark["results_poi_id"].append(list(results.poi_id))
        dic_benchmark["results_poi_name"].append(list(results.poi_name))
        dic_benchmark["results_poi_dist"].append(list(results.dist))
        dic_benchmark["results_poi_rank"].append(list(results["rank"]))
    return pd.DataFrame(dic_benchmark)