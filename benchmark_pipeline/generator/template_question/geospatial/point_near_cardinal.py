from collections import defaultdict

import numpy as np
import pandas as pd

from benchmark_pipeline.config import *
from benchmark_pipeline.generator.template_question.registry import template

CARDINAL_AZ = {
    "north": 0, "east": 90,
    "south": 180, "west": 270,
}

#: Directions rangées par azimut croissant : `SECTEURS[i]` est le secteur de 90°
#: centré sur `i * 90`, ce qui permet de l'indexer directement par l'azimut.
SECTEURS = np.array(sorted(CARDINAL_AZ, key=CARDINAL_AZ.get))


def pack_by_direction(df, x, y, distance_max=DISTANCE_MAX):
    """Classe les POIs dans les quatre secteurs cardinaux autour d'un point.

    Args:
        df (GeoDataFrame): POIs candidats, avec `poi_id`, `poi_name`, `x` et `y`
            en EPSG:2154.
        x (float): Abscisse Lambert-93 du point de référence.
        y (float): Ordonnée Lambert-93 du point de référence.
        distance_max (float | None): Max distance to the point, in meters; `None` keeps every POI.

    Returns:
        DataFrame: Colonnes `poi_id`, `poi_name`, `dist` (mètres), `az` (degrés),
            `direction`, triées par distance croissante. Les quatre secteurs de
            90° pavent le tour complet : chaque POI tombe dans exactement un.
    """
    dx = df["x"].to_numpy() - x
    dy = df["y"].to_numpy() - y
    # Azimut horaire depuis le nord : `arctan2(dx, dy)`, et non l'ordre
    # trigonométrique habituel `arctan2(dy, dx)`.
    az = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    resultats = pd.DataFrame({
        "poi_id": df["poi_id"].to_numpy(),
        "poi_name": df["poi_name"].to_numpy(),
        "dist": np.hypot(dx, dy),
        "az": az,
        "direction": SECTEURS[(((az + 45) % 360) // 90).astype(int)],
    })
    if distance_max is not None:
        resultats = resultats[resultats["dist"] <= distance_max]
    return resultats.sort_values("dist").reset_index(drop=True)


@template("make_question_point_near_cardinal")
def make_question_point_near_cardinal(df_osm, ratio=None, list_direction=None, nb_q=110, seed=42):
    """Génère les questions de direction cardinale « X au nord de Y ».

    Args:
        df_osm (GeoDataFrame): POIs servant d'ancres et de cibles.
        ratio (dict | None): Inutilisé ; conservé pour l'uniformité des signatures.
        list_direction (list[str]): Directions à décliner, clés de `CARDINAL_AZ`.
        nb_q (int): Nombre d'ancres tirées, en alternant les catégories.
        seed (int): Graine du générateur aléatoire.

    Returns:
        DataFrame: Une ligne par (ancre, direction). Mêmes colonnes que
            `make_question_nearsql`, plus `direction`.
    """
    if list_direction is None:
        list_direction = ["north", "east", "west", "south"]
    dic_benchmark = defaultdict(list)
    rng = np.random.default_rng(seed)
    list_cat = df_osm["category"].unique()
    # Sampling without replacement: one shuffled pool of distinct anchors per category
    anchor_pools = {cat: df_osm[df_osm["category"] == cat].sample(frac=1, random_state=rng) for cat in list_cat}
    for k in range(nb_q // len(list_direction)):
        cat_anc = list_cat[k % len(list_cat)]
        rank_in_pool = k // len(list_cat)
        if rank_in_pool >= len(anchor_pools[cat_anc]):
            continue
        anchor = anchor_pools[cat_anc].iloc[rank_in_pool]
        voisins = pack_by_direction(df_osm, anchor.x, anchor.y)
        voisins = voisins[voisins["poi_id"] != anchor.poi_id]
        for d in list_direction:
            results = voisins[voisins["direction"] == d]
            # Une question sans réponse est inévaluable : un secteur cardinal
            # peut ne contenir aucun POI, et la publier revient à noter un
            # modèle sur une question qui n'a pas de bonne réponse.
            if results.empty:
                continue
            dic_benchmark["query"].append(f"pois to the {d} of {anchor.poi_name}")
            dic_benchmark["anchor_index"].append(anchor.poi_id)
            dic_benchmark["anchor_name"].append(anchor.poi_name)
            dic_benchmark["anchor_category"].append(anchor.category)
            dic_benchmark["anchor_x"].append(anchor.x)
            dic_benchmark["anchor_y"].append(anchor.y)
            dic_benchmark["direction"].append(d)
            dic_benchmark["function"].append("pack_by_direction")
            dic_benchmark["results_poi_id"].append(list(results.poi_id))
            dic_benchmark["results_poi_name"].append(list(results.poi_name))
            dic_benchmark["results_poi_dist"].append(list(results.dist))
            dic_benchmark["results_poi_rank"].append(list(range(1, len(results) + 1)))
            dic_benchmark["anchors_type"].append(["point"])
            dic_benchmark["anchors_name"].append([anchor.poi_name])
            dic_benchmark["anchors_id"].append([anchor.poi_id])
            
    return pd.DataFrame(dic_benchmark)