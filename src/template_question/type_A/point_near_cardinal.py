import numpy as np
from collections import defaultdict
import duckdb
import pandas as pd

from src.config import *


CARDINAL_AZ = {
    "north": 0, "east": 90,
    "south": 180, "west": 270,
}

def cardinal_azimuth_sql(df, x, y, cat, cardinal_dir, k=100,
                         half_width=70.0, con=None):
    """Retourne les POIs d'une catégorie situés dans un secteur cardinal autour d'un point.

    L'azimut de chaque POI est mesuré dans le plan Lambert-93, en degrés horaires
    depuis le nord — d'où l'ordre `atan2(dx, dy)`, inversé par rapport à la
    convention mathématique. Un POI est retenu si son écart angulaire à la
    direction visée, ramené dans (-180°, 180°], ne dépasse pas `half_width`.

    Args:
        df (GeoDataFrame): POIs candidats en EPSG:2154.
        x (float): Abscisse Lambert-93 du point de référence.
        y (float): Ordonnée Lambert-93 du point de référence.
        cat (str): Catégorie de POI recherchée.
        cardinal_dir (str): Direction visée, clé de `CARDINAL_AZ`.
        k (int): Nombre maximal de résultats retournés.
        half_width (float): Demi-ouverture du secteur, en degrés.
        con (duckdb.DuckDBPyConnection | None): Connexion à réutiliser ; une
            connexion jetable est créée si None.

    Returns:
        DataFrame: Colonnes `poi_id`, `poi_name`, `dist`, `az` (degrés), triées
            par distance croissante.

    Note:
        Distance euclidienne et azimut planaire, sans correction de latitude :
        c'est la forme correcte en CRS projeté. Le nord de la grille Lambert-93
        s'écarte du nord géographique d'environ 0,5° à Paris (convergence des
        méridiens), négligeable devant `half_width`.

    TODO: une connexion DuckDB jetable est ouverte à chaque appel quand `con`
        vaut None, et `INSTALL spatial` est rejoué 440 fois sur une génération
        complète. Passer une connexion unique depuis l'appelant.
    """
    center = CARDINAL_AZ[cardinal_dir]
    if con is None:
        con = duckdb.connect()
        con.sql("INSTALL spatial; LOAD spatial;")
    flat = pd.DataFrame(df.drop(columns="geometry")).assign(geom_wkb=df.geometry.to_wkb())
    con.register("poi", flat)

    return con.execute("""
        WITH g AS (
            SELECT poi_id, poi_name, ST_GeomFromWKB(geom_wkb) AS geom
            FROM poi WHERE category = $cat
        ), d AS (
            SELECT poi_id, poi_name,
                   ST_Distance(geom, ST_Point($x, $y))::DOUBLE AS dist,
                   (degrees(atan2(ST_X(geom) - $x, ST_Y(geom) - $y)) + 360) % 360 AS az
            FROM g
        )
        SELECT poi_id, poi_name, dist, az
        FROM d
        WHERE abs(((az - $center + 540)::DOUBLE % 360) - 180) <= $half
        ORDER BY dist
        LIMIT $k
    """, {"x": x, "y": y, "cat": cat, "k": k,
          "center": center, "half": half_width}).df()


def make_question_point_near_cardinal(df_osm, list_direction=["north", "east", "west", "south"], half_width=70.0, nb_q=110, seed=42):
    """Génère les questions de direction cardinale « X au nord de Y ».

    Décline chaque ancre tirée sur les quatre directions demandées.

    Args:
        df_osm (GeoDataFrame): POIs servant d'ancres et de cibles.
        list_direction (list[str]): Directions à décliner, clés de `CARDINAL_AZ`.
        half_width (float): Demi-ouverture du secteur, en degrés.
        nb_q (int): Nombre d'ancres visé, stratifié par catégorie.
        seed (int): Graine du générateur aléatoire.

    Returns:
        DataFrame: Une ligne par (ancre, direction). Mêmes colonnes que
            `make_question_nearsql`, plus `direction`.

    TODO: avec `half_width=70` les secteurs se recouvrent (nord couvre 290°→70°,
        est couvre 20°→160°), donc un POI au nord-est appartient à deux vérités
        terrain de la même ancre. `half_width=45` pave le cercle sans recouvrement.
    TODO: `nb_q * len(list_direction)` lignes produites, pas `nb_q`.
    TODO: `results_poi_rank` 0-based et troué, cf. `make_question_nearsql`.
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
            for d in list_direction:
                results = cardinal_azimuth_sql(df_osm, anchor.x, anchor.y, cat_q, cardinal_dir=d, half_width=half_width)
                if cat_anc == cat_q:
                    results = results[results["poi_id"] != anchor.poi_id]

                dic_benchmark["query"].append(f"{cat_q} to the {d} of {anchor.poi_name}")
                dic_benchmark["anchor_index"].append(anchor.poi_id)
                dic_benchmark["anchor_name"].append(anchor.poi_name)
                dic_benchmark["anchor_category"].append(anchor.category)
                dic_benchmark["anchor_x"].append(anchor.x)
                dic_benchmark["anchor_y"].append(anchor.y)
                dic_benchmark["category_query"].append(cat_q)
                dic_benchmark["same_cat"].append(cat_anc==cat_q)
                dic_benchmark["direction"].append(d)
                dic_benchmark["function"].append("cardinal_azimuth_sql")
                dic_benchmark["results_poi_id"].append(list(results.poi_id))
                dic_benchmark["results_poi_name"].append(list(results.poi_name))
                dic_benchmark["results_poi_dist"].append(list(results.dist))
                dic_benchmark["results_poi_rank"].append(list(range(1, len(results) + 1)))

    return pd.DataFrame(dic_benchmark)