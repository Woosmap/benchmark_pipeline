from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from benchmark_pipeline.config import *
from benchmark_pipeline.generator.template_question.ratio import allocate
from benchmark_pipeline.generator.template_question.registry import template


def between_ab_sql(df, ax, ay, bx, by, cat, corridor_m=200.0):
    """Retourne les POIs d'une catégorie situés dans le corridor reliant deux points.

    Projette chaque POI dans le repère du segment A→B : `along` est l'abscisse
    curviligne le long du segment, `cross_m` l'écart perpendiculaire. Un POI est
    retenu si sa projection tombe entre A et B et si son écart latéral n'excède
    pas `corridor_m`. Le classement privilégie la proximité à l'axe, puis
    l'avancement le long du segment.

    Args:
        df (GeoDataFrame): POIs candidats, avec `poi_id`, `poi_name`, `x` et `y`
            en EPSG:2154.
        ax (float): Abscisse Lambert-93 du point A.
        ay (float): Ordonnée Lambert-93 du point A.
        bx (float): Abscisse Lambert-93 du point B.
        by (float): Ordonnée Lambert-93 du point B.
        cat (str | None): Catégorie de POI recherchée ; `None` ne filtre pas.
        corridor_m (float): Demi-largeur du corridor, en mètres.

    Returns:
        DataFrame: Colonnes `poi_id`, `poi_name`, `dist_a`, `dist_b`, `along`,
            `cross_m`, `rank`, triées par écart latéral puis avancement.
    """
    candidates = df if cat is None else df[df["category"] == cat]
    abx, aby = bx - ax, by - ay
    ab_len = np.sqrt(abx ** 2 + aby ** 2)
    px, py = candidates["x"].to_numpy() - ax, candidates["y"].to_numpy() - ay
    along = (px * abx + py * aby) / ab_len
    cross_m = np.abs(px * aby - py * abx) / ab_len
    kept = np.flatnonzero((along >= 0) & (along <= ab_len) & (cross_m <= corridor_m))
    # np.lexsort sorts on the last key first: cross_m, then along
    kept = kept[np.lexsort((along[kept], cross_m[kept]))]
    return pd.DataFrame({
        "poi_id": candidates["poi_id"].to_numpy()[kept],
        "poi_name": candidates["poi_name"].to_numpy()[kept],
        "dist_a": np.hypot(px, py)[kept],
        "dist_b": np.hypot(px - abx, py - aby)[kept],
        "along": along[kept],
        "cross_m": cross_m[kept],
        "rank": np.arange(1, len(kept) + 1),
    })

@template("make_question_point_between")
def make_question_point_between(df_osm, ratio_cat_anchor=None, corridor_m=200.0,
                                nb_q=110, distance_max=1000.0, seed=42,
                                nb_ancres_candidates=300):
    """Génère les questions d'entre-deux « pois entre A et B ».

    Les deux ancres sont des POIs quelconques, B étant tiré dans le voisinage de A.
    La vérité terrain est le corridor A-B tout entier, sans contrainte de catégorie
    sur les réponses ; le quota est réparti sur la catégorie de l'ancre A, de sorte
    que les classes d'ancres soient équilibrées.

    Args:
        df_osm (GeoDataFrame): POIs servant d'ancres et de cibles.
        ratio_cat_anchor (dict[str, float] | None): Poids par catégorie d'ancre.
            `None` répartit également sur toutes les catégories présentes.
        corridor_m (float): Demi-largeur du corridor, en mètres.
        nb_q (int): Nombre total de questions visé.
        distance_max (float): Rayon de tirage de B autour de A, en mètres.
        seed (int): Graine du générateur aléatoire.
        nb_ancres_candidates (int): Ancres A échantillonnées par catégorie pour
            énumérer les couples à portée. Les borner évite une énumération
            quadratique sur les catégories denses.

    Returns:
        DataFrame: Une ligne par question. Colonnes `query`, `anchor_a_*`,
            `anchor_b_*`, `function`, `results_poi_*` ; la distance rapportée est
            l'écart latéral à l'axe A-B.
    """
    list_cat = df_osm["category"].unique()
    balance = allocate(nb_q, ratio_cat_anchor or {cat: 1 for cat in list_cat})

    coordonnees = np.column_stack([df_osm.x.values, df_osm.y.values])
    arbre = cKDTree(coordonnees)
    poi_ids = df_osm["poi_id"].to_numpy()
    categories = df_osm["category"].to_numpy()
    dic_benchmark = defaultdict(list)
    rng = np.random.default_rng(seed)

    for cat_ancre_a, quota in balance.items():
        positions_ancre = np.flatnonzero((df_osm["category"] == cat_ancre_a).to_numpy())
        if len(positions_ancre) == 0:
            continue

        positions_ancre = rng.permutation(positions_ancre)[:nb_ancres_candidates]
        couples = defaultdict(list)
        for position_a, voisins in zip(positions_ancre,
                                       arbre.query_ball_point(coordonnees[positions_ancre], distance_max)):
            par_categorie = defaultdict(list)
            for k in voisins:
                if poi_ids[k] != poi_ids[position_a]:
                    par_categorie[categories[k]].append(k)
            for cat_voisine, positions_b in par_categorie.items():
                couples[cat_voisine].append((position_a, positions_b))
        if not couples:
            continue

        for cat_ancre_b, part in allocate(quota, {cat: 1 for cat in couples}).items():
            obtenues = 0
            for rang in rng.permutation(len(couples[cat_ancre_b])):
                if obtenues == part:
                    break
                position_a, positions_b = couples[cat_ancre_b][rang]
                anchor_a = df_osm.iloc[position_a]
                anchor_b = df_osm.iloc[rng.choice(positions_b)]

                results = between_ab_sql(df_osm, anchor_a.x, anchor_a.y, anchor_b.x, anchor_b.y,
                                         cat=None, corridor_m=corridor_m)
                results = results[~results["poi_id"].isin([anchor_a.poi_id, anchor_b.poi_id])]
                if results.empty:
                    continue
                obtenues += 1

                dic_benchmark["query"].append(f"pois between {anchor_a.poi_name} and {anchor_b.poi_name}")
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
                dic_benchmark["function"].append("between_ab_sql")
                dic_benchmark["results_poi_id"].append(list(results.poi_id))
                dic_benchmark["results_poi_name"].append(list(results.poi_name))
                dic_benchmark["results_poi_dist"].append(list(results.cross_m))
                dic_benchmark["results_poi_rank"].append(list(range(1, len(results) + 1)))
                dic_benchmark["anchors_type"].append(["point", "point"])
                dic_benchmark["anchors_name"].append([anchor_a.poi_name, anchor_b.poi_name])
                dic_benchmark["anchors_id"].append([anchor_a.poi_id, anchor_b.poi_id])
    return pd.DataFrame(dic_benchmark)