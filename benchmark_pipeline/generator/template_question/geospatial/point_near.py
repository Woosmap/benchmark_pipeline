import numpy as np

from benchmark_pipeline.generator.template_question.ratio import allocate
from collections import defaultdict
import pandas as pd
from shapely.geometry import Point

from benchmark_pipeline.config import *
from benchmark_pipeline.generator.template_question.registry import template


def near_sql(df, x, y, cat=None, k=100):
    """Retourne les `k` POIs les plus proches d'un point.

    Calcule les distances directement sur la colonne `geometry`, sans passer par
    DuckDB. Les coordonnées et les géométries sont en Lambert-93 (EPSG:2154),
    donc les distances sont euclidiennes et exprimées en mètres.

    Args:
        df (GeoDataFrame): POIs candidats, avec `poi_id`, `poi_name`, `category`
            et une colonne `geometry` en EPSG:2154.
        x (float): Abscisse Lambert-93 du point de référence.
        y (float): Ordonnée Lambert-93 du point de référence.
        cat (str | None): Catégorie de POI recherchée ; `None` ne filtre pas.
        k (int): Nombre maximal de résultats retournés.

    Returns:
        DataFrame: Colonnes `poi_id`, `poi_name`, `dist` (mètres), triées par
            distance croissante.

    """
    candidats = df if cat is None else df[df["category"] == cat]
    results = pd.DataFrame({
        "poi_id": candidats["poi_id"],
        "poi_name": candidats["poi_name"],
        "dist": candidats.geometry.distance(Point(x, y)),
    })
    return results.nsmallest(k, "dist").reset_index(drop=True)

@template("make_question_point_near")
def make_question_point_near(df_osm, ratio=None, nb_q=110, seed=42):
    """Génère les questions de proximité simple « X près de Y ».

    Répartit le quota sur la catégorie de l'ancre, de sorte que les classes
    d'ancres soient équilibrées. La vérité terrain est le classement de tous les
    POIs par distance à l'ancre, sans contrainte de catégorie sur les réponses ;
    l'ancre est retirée de ses propres résultats. `category_query` vaut donc
    constamment `"pois"`, et `same_cat` constamment `True`.

    Args:
        df_osm (GeoDataFrame): POIs servant à la fois d'ancres et de cibles.
        ratio (dict[str, float] | None): Poids par catégorie d'ancre ; uniforme si None.
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
    for k in range(nb_q):
        cat_anc = list_cat[k % len(list_cat)]
        sub = df_osm[df_osm["category"] == cat_anc]
        anchor = sub.sample(1, random_state=rng).iloc[0]
        results = near_sql(df_osm, anchor.x, anchor.y)
        results = results[results["poi_id"] != anchor.poi_id]
        
        dic_benchmark["query"].append(f"pois near {anchor.poi_name}")
        dic_benchmark["anchor_index"].append(anchor.poi_id)
        dic_benchmark["anchor_name"].append(anchor.poi_name)
        dic_benchmark["anchor_category"].append(anchor.category)
        dic_benchmark["anchor_x"].append(anchor.x)
        dic_benchmark["anchor_y"].append(anchor.y)
        dic_benchmark["function"].append("near_sql")
        dic_benchmark["results_poi_id"].append(list(results.poi_id))
        dic_benchmark["results_poi_name"].append(list(results.poi_name))
        dic_benchmark["results_poi_dist"].append(list(results.dist))
        dic_benchmark["results_poi_rank"].append(list(range(1, len(results) + 1)))
    return pd.DataFrame(dic_benchmark)
