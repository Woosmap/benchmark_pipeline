"""Relations des backward hops : celles autour d'un POI, plus between et towards avec un second POI Y.

Les tests utilisent le monde synthétique de `conftest.py` : pas de cache OSM, pas de réseau.
"""

import pandas as pd

from benchmark_pipeline.config import DIRECTIONS, FEATURES, METRIC_DISTANCES
from benchmark_pipeline.generator.geospatial.point_between import between_ab_sql
from benchmark_pipeline.generator.geospatial.point_towards import towards_b_sql
from benchmark_pipeline.generator.multihop.add_hop import relations_by_key
from benchmark_pipeline.generator.multihop.make_multihop_text import attributes_text, make_text_multihop

CORRIDOR, HALF_WIDTH, DISTANCE_MAX = 200.0, 70.0, 1000.0


def relations(df_osm, anchors, second):
    return relations_by_key(anchors, second, df_osm, DIRECTIONS, METRIC_DISTANCES, 50, 50,
                            DISTANCE_MAX, CORRIDOR, HALF_WIDTH)


def test_between_et_towards_autour_d_une_reponse(df_osm):
    """between et towards gardent les POIs de between_ab_sql et towards_b_sql, sans l'ancre ni Y."""
    anchor = df_osm.iloc[0]
    distances = df_osm.geometry.distance(anchor.geometry)
    second = df_osm.loc[distances[distances <= DISTANCE_MAX].idxmax()]   # Y au bout du voisinage : corridor non vide
    found = relations(df_osm, df_osm.iloc[[0]], second)

    a, b = anchor.geometry, second.geometry
    between = set(between_ab_sql(df_osm, a.x, a.y, b.x, b.y, cat=None, corridor_m=CORRIDOR).poi_id)
    towards = set(towards_b_sql(df_osm, a.x, a.y, b.x, b.y, half_width=HALF_WIDTH, distance_max=DISTANCE_MAX).poi_id)
    excluded = {anchor.poi_id, second.poi_id}
    assert found["between_ab_sql", f"between {{}} and {second.poi_name}"] == between - excluded != set()
    assert found["towards_b_sql", f"near {{}} towards {second.poi_name}"] == towards - excluded != set()


def test_sans_second_poi_seulement_les_relations_autour_d_un_poi(df_osm):
    found = relations(df_osm, df_osm.iloc[[0]], None)
    assert {function for function, _ in found} == {"near_sql", "near_metric_sql", "pack_by_direction"}
    assert all("{}" in relation for _, relation in found)


def test_la_description_place_la_question_dans_between():
    """La question du dessous peut être au milieu de la relation : « cafe between [Q] and Opera »."""
    df_question = pd.DataFrame({"query_texte": ["restaurant near Louvre"],
                                "features": [("category",)], "values": [("restaurant",)]})
    head = attributes_text(("category",), ("cafe",), FEATURES)
    short = attributes_text(("category",), ("restaurant",), FEATURES)
    df_multihop = pd.DataFrame([{"index": 1, "index_origin": 0, "index_lateral": None,
                                 "query_texte": f"{head} between restaurant near Louvre and Opera",
                                 "features": ("category",), "values": ("cafe",)}])

    description = make_text_multihop(df_multihop, df_question, FEATURES).loc[0, "description_textuelle"]
    first_sentence = f"{head} between {short} and Opera"
    assert description.startswith(first_sentence[0].upper() + first_sentence[1:] + ".")
