import numpy as np
from collections import defaultdict
import duckdb
import pandas as pd
import shapely

from benchmark_pipeline.config import *
from benchmark_pipeline.generator.template_question.registry import template


def side_of_street(street_geom, pt):
    """Détermine de quel côté d'une rue se trouve un point.

    Estime la tangente à la rue au droit du point, par deux points d'appui pris à
    ±5 m de son projeté, puis prend le signe du produit vectoriel 2D entre cette
    tangente et le vecteur allant du premier point d'appui vers le point. Le signe
    n'a de valeur que relative : deux points de signes opposés sont de part et
    d'autre de la rue.

    Args:
        street_geom (LineString | MultiLineString): Géométrie de la rue.
        pt (Point): Point à situer.

    Returns:
        float: +1.0 ou -1.0 selon le côté.

    TODO: `street_geom.geoms[0]` retient silencieusement le premier tronçon d'une
        `MultiLineString` — pour une rue en deux morceaux, le côté est calculé par
        rapport au mauvais tronçon. Projeter sur le tronçon le plus proche de `pt`.
    """
    s = street_geom.geoms[0] if street_geom.geom_type == "MultiLineString" else street_geom
    t = s.project(pt)                                    # abscisse curviligne du projeté
    p0 = s.interpolate(max(t - 5.0, 0))                  # deux points encadrant, pour la tangente
    p1 = s.interpolate(min(t + 5.0, s.length))
    cross = ((p1.x - p0.x) * (pt.y - p0.y)
             - (p1.y - p0.y) * (pt.x - p0.x))            # produit vectoriel 2D
    return 1.0 if cross >= 0 else -1.0

def opposite_side(df, street_geom, poi_y_geom, cat, k=100,
                  max_along=80.0, max_cross=40.0):
    """Retourne les POIs d'une catégorie situés en face d'un POI de référence, de l'autre côté d'une rue.

    Un candidat est retenu s'il est du côté opposé au POI de référence, à moins de
    `max_cross` mètres de la rue, et décalé d'au plus `max_along` mètres le long de
    la rue. Le classement privilégie le vis-à-vis, c'est-à-dire le plus faible
    écart d'abscisse curviligne avec le POI de référence.

    Args:
        df (GeoDataFrame): POIs candidats en EPSG:2154.
        street_geom (LineString | MultiLineString): Rue traversée.
        poi_y_geom (Point): POI de référence, sur l'autre rive.
        cat (str): Catégorie de POI recherchée.
        k (int): Nombre maximal de résultats retournés.
        max_along (float): Décalage longitudinal maximal admis, en mètres.
        max_cross (float): Éloignement maximal à la rue, en mètres.

    Returns:
        DataFrame: Colonnes `poi_id`, `poi_name`, `cross`, `along`, `score` et
            `rank` (1-based).

    TODO: même troncature `geoms[0]` que `side_of_street`.
    TODO: `sub.geometry.apply(...)` évalue `side_of_street` et `project` POI par
        POI, en Python, à chaque question. Vectoriser avec
        `GeoSeries.project`/`shapely.line_locate_point`.
    """
    s = street_geom.geoms[0] if street_geom.geom_type == "MultiLineString" else street_geom
    side_y = side_of_street(s, poi_y_geom)
    t_y = s.project(poi_y_geom)

    # L'index spatial écarte d'emblée les POIs trop éloignés de la rue : le calcul
    # de côté ne porte plus que sur les candidats admissibles, pas sur toute la
    # catégorie. `dwithin` donne exactement le filtre `cross <= max_cross`.
    sub = df[df["category"] == cat]
    proches = np.sort(sub.sindex.query(s, predicate="dwithin", distance=max_cross))
    sub = sub.iloc[proches]
    if sub.empty:
        return pd.DataFrame(columns=["poi_id", "poi_name", "cross", "along", "score", "rank"])

    # Même géométrie que `side_of_street`, mais sur tout le lot en une fois.
    pts = sub.geometry.values
    along = shapely.line_locate_point(s, pts)
    p0 = shapely.line_interpolate_point(s, np.maximum(along - 5.0, 0))
    p1 = shapely.line_interpolate_point(s, np.minimum(along + 5.0, s.length))
    x0, y0 = shapely.get_x(p0), shapely.get_y(p0)
    x1, y1 = shapely.get_x(p1), shapely.get_y(p1)
    px, py = shapely.get_x(pts), shapely.get_y(pts)
    side = np.where((x1 - x0) * (py - y0) - (y1 - y0) * (px - x0) >= 0, 1.0, -1.0)
    cross = shapely.distance(pts, s)

    garde = (side != side_y) & (cross <= max_cross) & (np.abs(along - t_y) <= max_along)
    out = pd.DataFrame({
        "poi_id": sub["poi_id"].values[garde],
        "poi_name": sub["poi_name"].values[garde],
        "cross": cross[garde],
        "along": along[garde],
    })
    out["score"] = (out["along"] - t_y).abs()
    out = out.nsmallest(k, "score").reset_index(drop=True)
    out["rank"] = out.index + 1
    return out

@template("make_question_street_opposite_side")
def make_question_street_opposite_side(df_osm, df_streets, nb_q=110, seed=42, max_tries=200):
    """Génère les questions de vis-à-vis « X en face de Y, de l'autre côté de la rue R ».

    Tire une rue à géométrie simple, puis un POI de référence parmi ceux situés à
    moins de 40 m de cette rue.

    Args:
        df_osm (GeoDataFrame): POIs servant de références et de cibles.
        df_streets (GeoDataFrame): Rues en EPSG:2154 ; seules les `LineString`
            sont retenues.
        nb_q (int): Nombre total de questions visé, stratifié par catégorie cible.
        seed (int): Graine du générateur aléatoire.

    Returns:
        DataFrame: Une ligne par question. Colonnes `query`, `category_query`,
            `street_index`, `street_name`, `street_geometry`, `poi_y_id`,
            `poi_y_name`, `poi_y_geometry`, `function`, plus les quatre colonnes
            `results_poi_*`.
    """
    dic_benchmark = defaultdict(list)
    rng = np.random.default_rng(seed)
    list_cat = df_osm["category"].unique()
    n_queries_per_stratum = nb_q // len(list_cat)

    lines = df_streets[df_streets.geometry.geom_type.isin(["LineString", "MultiString"])]

    for cat_q in list_cat:
        n = 0
        tries = 0
        while n < n_queries_per_stratum and tries < max_tries:
            tries += 1
            street = lines.loc[rng.choice(lines.index)]

            # Sans l'index, cette ligne mesurait la distance des 19 000 POIs à la
            # rue, à chaque essai — jusqu'à 200 fois par catégorie.
            proches = np.sort(df_osm.sindex.query(street.geometry, predicate="dwithin", distance=40))
            near = df_osm.iloc[proches]
            if near.empty:
                continue
            poi_y = near.loc[rng.choice(near.index)]

            results = opposite_side(df_osm, street.geometry, poi_y.geometry, cat_q)
            results = results[results["poi_id"] != poi_y["poi_id"]]
            if results.empty:
                continue

            dic_benchmark["query"].append(f"{cat_q} across {street['street_name']} from {poi_y['poi_name']}")
            dic_benchmark["category_query"].append(cat_q)
            dic_benchmark["street_index"].append(street.name)
            dic_benchmark["street_name"].append(street["street_name"])
            dic_benchmark["street_geometry"].append(street.geometry)
            dic_benchmark["poi_y_id"].append(poi_y["poi_id"])
            dic_benchmark["poi_y_name"].append(poi_y["poi_name"])
            dic_benchmark["poi_y_geometry"].append(poi_y.geometry)
            dic_benchmark["function"].append("opposite_side")
            dic_benchmark["results_poi_id"].append(list(results.poi_id))
            dic_benchmark["results_poi_name"].append(list(results.poi_name))
            dic_benchmark["results_poi_dist"].append(list(results["cross"]))
            dic_benchmark["results_poi_rank"].append(list(results["rank"]))
            n += 1

    return pd.DataFrame(dic_benchmark)