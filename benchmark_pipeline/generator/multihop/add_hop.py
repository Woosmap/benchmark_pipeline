from collections import defaultdict
import pandas as pd
from shapely.geometry import shape
from shapely.ops import unary_union
import numpy as np

from benchmark_pipeline.generator.features import *
from benchmark_pipeline.generator.composite.generator_qcomposite import make_question_composite, question_geo_semantic
from benchmark_pipeline.generator.geospatial.area_border import pois_near_border
from benchmark_pipeline.generator.geospatial.area_direction import segregate_pois
from benchmark_pipeline.generator.geospatial.area_inside import pois_inside_area
from benchmark_pipeline.generator.geospatial.point_between import between_ab_sql
from benchmark_pipeline.generator.geospatial.point_near import near_sql
from benchmark_pipeline.generator.geospatial.point_near_cardinal import pack_by_direction
from benchmark_pipeline.generator.geospatial.point_near_metric import near_metric_sql
from benchmark_pipeline.generator.geospatial.point_towards import towards_b_sql
from benchmark_pipeline.generator.geospatial.street_along import along_street
from benchmark_pipeline.config import MANDATORY_FEATURES, MULTIHOP_COLUMNS


def find_area(pois):
    """retourne la zone dans laquelle se trouve le pois
    """
    return unary_union([shape(p["geometry"]) for _, p in pois.iterrows()]).convex_hull

def object_inside_area(area, gdf):
    """retourne les pois, streets et area qui se trouve dans la zone area
    """
    idx = gdf.sindex.query(area, predicate="intersects")
    return gdf.iloc[idx]

def next_label(df_question, df_multihop):
    last = df_multihop["index"].max() if len(df_multihop) else -1
    return max(df_question.index.max(), last) + 1

def get_question(id_question, df_question, df_multihop):
    """The MANDATORY_FEATURES fields of a question, whether single-hop (df_question) or multihop (df_multihop).
    """
    if len(df_multihop) and id_question in set(df_multihop["index"]):
        row = df_multihop.set_index("index").loc[id_question]
    else:
        row = df_question.loc[id_question]
    question = {column: row.get(column) for column in list(MANDATORY_FEATURES) + MULTIHOP_COLUMNS}
    question["answers"] = list(question["answers"])
    return question

def anchors_graph(question):
    """`anchors_type` and `anchors_id` of `question` as {level: [anchors of each question of the node]}, like dic_query.

    A single-hop question has the flat lists of its generator: they become its only node.
    """
    if isinstance(question["anchors_type"], dict):
        return question["anchors_type"], question["anchors_id"]
    return {1: [list(question["anchors_type"])], 2: None}, {1: [list(question["anchors_id"])], 2: None}

def node_anchors(question):
    """`anchors_type` and `anchors_id` of the last node of `question` (level 1), as flat lists."""
    types, ids = anchors_graph(question)
    return [t for anchors in types[1] for t in anchors], [i for anchors in ids[1] for i in anchors]

def add_lateral_hop(
        id_question,
        nb_hop,
        df_question,
        df_osm,
        df_street,
        df_area,
        df_multihop,
        *,
        max_new_queries,
        seed,
        area_buffer,
        nb_q_by_temp,
        nb_q_by_feat,
        ratio_features,
        part_with_features,
        list_template,
        template_params,
    ):
    """Create questions made of `nb_hop` questions asked together: `id_question` and `nb_hop - 1` lateral questions.

    Lateral questions use other anchors near the answers of `id_question` and share some of its answers.

    Args:
        id_question: label of the starting question in `df_question`.
        nb_hop (int): number of questions asked in parallel in each new question.
        max_new_queries (int): number of combinations drawn; only those with at least one answer are kept.
        area_buffer (float): meters around the answers of `id_question` where lateral anchors are searched.
        nb_q_by_temp (int): `nb_q` of each template for the lateral questions.
        nb_q_by_feat (int): attribute constraints drawn per lateral question.
        ratio_features, part_with_features, list_template, template_params: passed to `make_question_composite`.

    Returns:
        tuple: `index` of the new questions in `df_multihop`, `df_question` with the lateral questions,
            `df_multihop` with the new questions.
    """
    if nb_hop < 2:
        return [], df_question, df_multihop

    df_all = pd.concat([df_osm, df_street, df_area], ignore_index=True)
    random_generator = np.random.default_rng(seed)
    question = get_question(id_question, df_question, df_multihop)
    answers = set(question["answers"])
    area = find_area(df_osm.loc[question["answers"]]).buffer(area_buffer)
    anchors = object_inside_area(area, df_all)
    df_street = anchors[anchors["street_id"].notna()]
    df_area = anchors[anchors["area_id"].notna()]
    pois = df_osm[df_osm["geometry"].within(area)]["poi_id"]
    lateral_questions = make_question_composite(
        df_osm.loc[pois],
        df_area, df_street,
        nb_q_by_temp=nb_q_by_temp,
        nb_q_by_feat=nb_q_by_feat,
        ratio_features=ratio_features,
        part_avec_features=part_with_features,
        feature_cols=question["features"],
        seed=seed,
        list_template=list_template,
        template_params=template_params,
    )
    if lateral_questions.empty:   # no lateral question around an isolated answer
        return [], df_question, df_multihop
    # every lateral question goes to df_question, under labels after the existing ones
    start = next_label(df_question, df_multihop)
    lateral_questions = lateral_questions.set_axis(list(range(start, start + len(lateral_questions))))
    df_question = pd.concat([df_question, lateral_questions.rename(columns={"results_features_pois_id": "answers"})])

    shares_answers = lateral_questions["results_features_pois_id"].map(
            lambda ids: 0 < len(answers.intersection(ids)) < len(answers)
        )
    lateral_questions = lateral_questions[shares_answers]
    types_graph, ids_graph = anchors_graph(question)
    own_types, own_ids = node_anchors(question)
    new_rows = []
    if len(lateral_questions) >= nb_hop - 1:
        for _ in range(max_new_queries):
            chosen_laterals = random_generator.choice(lateral_questions.index, size=nb_hop - 1, replace=False)
            common_answers = set(answers)
            texts = [question["query_texte"]]
            for lateral_id in chosen_laterals:
                lateral = lateral_questions.loc[lateral_id]
                common_answers &= set(lateral["results_features_pois_id"])
                texts.append(lateral["query"][len("pois "):])
            if common_answers:
                # level 1 of the graph is [question, *laterals], as in dic_query: one list of anchors per question
                laterals = lateral_questions.loc[sorted(chosen_laterals)]
                new_rows.append({"index_origin": id_question,
                                 "index_lateral": sorted(chosen_laterals),
                                 "query_texte":"[" + " and ".join(texts) + "]",
                                 "features": question["features"],
                                 "values": question["values"],
                                 "answers": sorted(common_answers),
                                 "anchors_type": {**types_graph, 1: [own_types] + list(laterals["anchors_type"])},
                                 "anchors_id": {**ids_graph, 1: [own_ids] + list(laterals["anchors_id"])}})
    if not new_rows:
        return [], df_question, df_multihop

    if isinstance(question["template"], dict):
        template = dict(question["template"])
        template[min(template)] += (nb_hop - 1)
    else:
        template = {1: nb_hop, 2: 0}
    dic_query = dict(question["dic_query"] or {})

    df_new_qm = pd.DataFrame(new_rows).drop_duplicates("query_texte")
    # df_question now holds the laterals: new labels must start after theirs
    start = next_label(df_question, df_multihop)
    df_new_qm.insert(0, "index", list(range(start, start + len(df_new_qm))))
    df_new_qm["template"] = [dict(template) for _ in range(len(df_new_qm))]
    df_new_qm["dic_query"] = [{**dic_query, 1: [id_question] + [int(i) for i in laterals]}
                              for laterals in df_new_qm["index_lateral"]]
    df_multihop = pd.concat([df_multihop, df_new_qm], ignore_index=True)
    return list(df_new_qm["index"]), df_question, df_multihop


def relations_around(anchor, df_osm, directions, metric_distances, border_band, street_max_dist, distance_max):
    """Template relations around `anchor`, a POI, an area or a street: (function, relation text, poi_ids)."""
    geometry = anchor.geometry
    if geometry.geom_type in ("Polygon", "MultiPolygon"):
        yield "pois_inside_area", "inside", pois_inside_area(df_osm, geometry).poi_id
        yield "pois_near_border", "at the border of", pois_near_border(df_osm, geometry, border_band).poi_id
        inside = df_osm[df_osm.geometry.within(geometry)]
        for direction, sector in segregate_pois(inside, geometry, directions).items():
            yield "segregate_pois", f"at the {direction} of", sector.poi_id
    elif "LineString" in geometry.geom_type:
        yield "along_street", "following", along_street(df_osm, geometry, street_max_dist).poi_id
    else:
        yield "near_sql", "near", near_sql(df_osm, geometry.x, geometry.y, distance_max=distance_max).poi_id
        for distance, batch in near_metric_sql(df_osm, geometry.x, geometry.y, metric_distances).groupby("distance"):
            yield "near_metric_sql", f"at less than {distance} meters from", batch.poi_id
        for direction, sector in pack_by_direction(df_osm, geometry.x, geometry.y, distance_max=distance_max).groupby("direction"):
            yield "pack_by_direction", f"to the {direction} of", sector.poi_id

def two_anchor_relations(anchor, second, df_osm, corridor_m, half_width, distance_max):
    """Relations between the POI `anchor` and a second POI: (function, relation with {} for the anchor, poi_ids)."""
    a, b = anchor.geometry, second.geometry
    yield ("between_ab_sql", f"between {{}} and {second.poi_name}",
           between_ab_sql(df_osm, a.x, a.y, b.x, b.y, cat=None, corridor_m=corridor_m).poi_id)
    yield ("towards_b_sql", f"near {{}} towards {second.poi_name}",
           towards_b_sql(df_osm, a.x, a.y, b.x, b.y, half_width=half_width, distance_max=distance_max).poi_id)

def relations_by_key(anchors, second, df_osm, directions, metric_distances, border_band, street_max_dist,
                     distance_max, corridor_m, half_width):
    """POIs of each relation of a backward hop around the POIs `anchors`: {(function, relation): poi_ids}.

    `relation` has {} where the anchor goes: "near {}", "between {} and Y". `second` is the POI Y of between
    and towards, None to leave them out. Each anchor, and Y, is left out of its own results.
    """
    found = defaultdict(set)
    for _, anchor in anchors.iterrows():
        for function, relation, ids in relations_around(anchor, df_osm, directions, metric_distances,
                                                        border_band, street_max_dist, distance_max):
            found[function, relation + " {}"] |= set(ids) - {anchor.poi_id}
        if second is not None:
            for function, relation, ids in two_anchor_relations(anchor, second, df_osm, corridor_m, half_width,
                                                                distance_max):
                found[function, relation] |= set(ids) - {anchor.poi_id, second.poi_id}
    return found

def node_pois(question, df_osm):
    """The POI anchors of the last node of `question`; streets and areas are left out, their ids are not poi_ids."""
    return df_osm.loc[[anchor_id for kind, anchor_id in zip(*node_anchors(question)) if kind == "point"]]

def draw_second_anchor(answers, excluded, df_osm, distance_max, seed):
    """Y of between and towards: a named POI less than `distance_max` from one of the POIs `answers`, outside
    `excluded`, drawn as in make_question_point_between and make_question_point_towards; None if there is none."""
    close = df_osm.geometry.distance(unary_union(list(answers.geometry))) <= distance_max
    named = df_osm["poi_name"].notna() & (df_osm["poi_name"] != "")
    candidates = df_osm[close & named & ~df_osm["poi_id"].isin(excluded)]
    if candidates.empty:
        return None
    return candidates.iloc[np.random.default_rng(seed).integers(len(candidates))]

def shift_keys(d, head):
    return {1: head, **{k + 1: v for k, v in d.items()}}

def drop_reachable_from_anchor(df_q2, question, second, df_osm, *relation_params):
    """Drop the rows of `df_q2` whose answers are those of the same question asked around the base anchor.

    Each row is asked again with the same relation, the same Y and the same attributes, around the anchors
    of `question` instead of its answers; when this ground truth equals the answers of the row, the middle
    anchor adds nothing. Only POI anchors give the relations of a backward hop (near, at less than d meters,
    to the north of, between, towards).

    relation_params: directions, metric_distances, border_band, street_max_dist, distance_max, corridor_m,
        half_width, as in `relations_by_key`.
    """
    # POIs around the anchors of the last node of `question`, for each relation: the same for every row
    answers = relations_by_key(node_pois(question, df_osm), second, df_osm, *relation_params)

    to_drop = []
    for index, row in df_q2.iterrows():
        pois = df_osm.loc[sorted(answers[row["function"], row["relation"]])]
        has_features = (pois[list(row["features"])] == list(row["values"])).all(axis=1)
        if set(pois.poi_id[has_features]) == set(row["results_features_pois_id"]):
            to_drop.append(index)
    return df_q2.drop(index=to_drop)


def add_backward_hop(
        id_question,
        df_osm,
        df_question,
        df_multihop,
        nb_q_by_feat,
        seed,
        *,
        feature_cols,
        ratio_features,
        part_with_features,
        directions,
        metric_distances,
        border_band,
        street_max_dist,
        distance_max,
        corridor_m,
        half_width,
    ):
    question = get_question(id_question, df_question, df_multihop)
    answer_pois = df_osm.loc[question["answers"]]
    excluded = set(question["answers"]) | set(node_pois(question, df_osm)["poi_id"])
    second = draw_second_anchor(answer_pois, excluded, df_osm, distance_max, seed)
    relation_params = (directions, metric_distances, border_band, street_max_dist, distance_max, corridor_m, half_width)
    answers = relations_by_key(answer_pois, second, df_osm, *relation_params)

    # the question goes where {} is: "pois near [Q]", "pois between [Q] and Y"
    df_geo = pd.DataFrame([{"function": function, "relation": relation,
                            "query": "pois " + relation.replace("{}", question["query_texte"], 1),
                            "results_poi_id": sorted(ids)} for (function, relation), ids in answers.items()])

    df_q2 = question_geo_semantic(df_geo, df_osm, feature_cols, ratio_features, nb_q_by_feat,
                                  part_with_features, seed=seed)
    df_q2 = drop_reachable_from_anchor(df_q2, question, second, df_osm, *relation_params)
    if df_q2.empty:
        return [], df_multihop
    start = next_label(df_question, df_multihop)
    new_q_index = list(range(start, start + len(df_q2)))
    df_q2 = df_q2.set_axis(new_q_index)

    new_template = shift_keys(question["template"] or {1: 1, 2: 0}, 1)
    base_dic = shift_keys(question["dic_query"] or {}, 0)
    dic_queries = [{**base_dic, 1: k} for k in range(len(df_q2))]
    types_graph, ids_graph = anchors_graph(question)
    anchors_type = shift_keys(types_graph, [["point"] * len(question["answers"])])
    anchors_id = shift_keys(ids_graph, [list(question["answers"])])

    df_new_qm = pd.DataFrame({
        "index": df_q2.index,
        "index_origin": [id_question]*len(df_q2.index),
        "features": df_q2["features"],
        "query_texte": df_q2["query_texte"].to_numpy(),
        "answers": df_q2["results_features_pois_id"].to_numpy(),
        "values": df_q2["values"].to_numpy(),
        "template": [dict(new_template) for _ in range(len(df_q2.index))],
        "dic_query": dic_queries,
        "anchors_type": [dict(anchors_type) for _ in range(len(df_q2.index))],
        "anchors_id": [dict(anchors_id) for _ in range(len(df_q2.index))],
        })
    df_multihop = pd.concat([df_multihop, df_new_qm], ignore_index=True).rename_axis("index")
    return new_q_index, df_multihop
