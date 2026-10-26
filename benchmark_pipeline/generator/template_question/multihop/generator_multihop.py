from collections import defaultdict

from shapely.geometry import shape
from shapely.ops import unary_union
import pandas as pd
import numpy as np

from benchmark_pipeline.generator.features import FEATURES
from benchmark_pipeline.generator.template_question.composite.generator_qcomposite import (
    make_question_composite,
    question_geo_semantic,
)
from benchmark_pipeline.generator.template_question.geospatial.area_border import (
    pois_near_border,
)
from benchmark_pipeline.generator.template_question.geospatial.area_direction import (
    segregate_pois,
)
from benchmark_pipeline.generator.template_question.geospatial.area_inside import (
    pois_inside_area,
)
from benchmark_pipeline.generator.template_question.geospatial.point_near import (
    near_sql,
)
from benchmark_pipeline.generator.template_question.geospatial.point_near_cardinal import (
    pack_by_direction,
)
from benchmark_pipeline.generator.template_question.geospatial.point_near_metric import (
    near_metric_sql,
)
from benchmark_pipeline.generator.template_question.geospatial.street_along import (
    along_street,
)


MULTIHOP_COLUMNS = ["query_texte", "features", "dic_query", "values", "answers", "template"]

list_template = ['make_question_area_border',
                 'make_question_area_inside', 'make_question_area_direction',
                 'make_question_point_between', 'make_question_point_near_cardinal',
                 'make_question_point_near_metric', 'make_question_point_near',
                 'make_question_point_towards', 'make_question_street_along',
                 'make_question_street_cross']

list_template_forward = ['make_question_point_between', 'make_question_point_near_cardinal', 
                         'make_question_point_near_metric', 'make_question_point_near', 
                         'make_question_point_towards']


def make_df_all(df_osm, df_street, df_area):
    return pd.concat([df_osm, df_street, df_area], ignore_index=True)

def format_df_question(df_question):
    return (df_question
             .rename(columns={"results_features_pois_id": "answers"})
             .assign(template=[{1: 1, 2: 0}] * len(df_question))
             .assign(dic_query=[{1: index, 2: None} for index in df_question.index])
             [MULTIHOP_COLUMNS])

def find_area(pois):
    return unary_union([shape(p["geometry"]) for _, p in pois.iterrows()]).convex_hull

def find_anchors(area, gdf):
    idx = gdf.sindex.query(area, predicate="intersects")
    return gdf.iloc[idx]

def next_label(df_question, df_multihop):
    last = df_multihop["index"].max() if len(df_multihop) else -1
    return max(df_question.index.max(), last) + 1

def get_question(id_question, df_question, df_multihop):
    """The MULTIHOP_COLUMNS fields of a question, whether single-hop (df_question) or multihop (df_multihop).
    """
    if len(df_multihop) and id_question in set(df_multihop["index"]):
        row = df_multihop.set_index("index").loc[id_question]
    else:
        row = df_question.loc[id_question]
    question = {column: row.get(column) for column in MULTIHOP_COLUMNS}
    question["answers"] = list(question["answers"])
    return question

def add_lateral_hop(
        id_question,
        nb_hop,
        df_question,
        df_osm,
        df_all,
        df_multihop,
        max_new_questions=50,
        seed=0,
    ):
    """Create questions made of `nb_hop` questions asked together: `id_question` and `nb_hop - 1` lateral questions.

    Lateral questions use other anchors near the answers of `id_question` and share some of its answers.

    Args:
        id_question: label of the starting question in `df_question`.
        nb_hop (int): number of questions asked in parallel in each new question.
        max_new_questions (int): number of combinations drawn; only those with at least one answer are kept.

    Returns:
        tuple: `index` of the new questions in `df_multihop`, `df_question` with the lateral questions,
            `df_multihop` with the new questions.
    """
    if nb_hop < 2:
        return [], df_question, df_multihop
    random_generator = np.random.default_rng(seed)
    question = get_question(id_question, df_question, df_multihop)
    answers = set(question["answers"])
    area = find_area(df_osm.loc[question["answers"]]).buffer(200)
    anchors = find_anchors(area, df_all)
    df_street = anchors[anchors["id_street"].notna()]
    df_area = anchors[anchors["id_area"].notna()]
    pois = df_osm[df_osm["geometry"].within(area)]["poi_id"]
    lateral_questions = make_question_composite(
        df_osm.loc[pois],
        df_area, df_street,
        nb_q_by_temp=100,
        nb_q_by_feat=10,
        list_template=list_template,
        feature_cols=question["features"],
    )
    shares_answers = lateral_questions["results_features_pois_id"].map(lambda ids: not answers.isdisjoint(ids))
    lateral_questions = lateral_questions[shares_answers]
    start = next_label(df_question, df_multihop)
    lateral_questions = lateral_questions.set_axis(list(range(start, start + len(lateral_questions))))
    df_question = pd.concat([df_question, lateral_questions.rename(columns={"results_features_pois_id": "answers"})])
    new_rows = []
    if len(lateral_questions) >= nb_hop - 1:
        for _ in range(max_new_questions):
            chosen_laterals = random_generator.choice(lateral_questions.index, size=nb_hop - 1, replace=False)
            common_answers = set(answers)
            texts = [question["query_texte"]]
            for lateral_id in chosen_laterals:
                lateral = lateral_questions.loc[lateral_id]
                common_answers &= set(lateral["results_features_pois_id"])
                texts.append(lateral["query"][len("pois "):])
            if common_answers:
                new_rows.append({"index_origin": id_question,
                                 "index_lateral": sorted(chosen_laterals),
                                 "query_texte":"[" + " and ".join(texts) + "]",
                                 "features": question["features"],
                                 "values": question["values"],
                                 "answers": sorted(common_answers)})
    if not new_rows:
        return [], df_question, df_multihop

    if isinstance(question["template"], dict):
        template = dict(question["template"])
        template[min(template)] += (nb_hop - 1)
    else:
        template = {1: nb_hop, 2: 0}
    dic_query = dict(question["dic_query"] or {})

    df_new_qm = pd.DataFrame(new_rows).drop_duplicates("query_texte")
    df_new_qm.insert(0, "index", list(range(start, start + len(df_new_qm))))
    df_new_qm["template"] = [dict(template) for _ in range(len(df_new_qm))]
    df_new_qm["dic_query"] = [{**dic_query, 1: [id_question] + [int(i) for i in laterals]}
                              for laterals in df_new_qm["index_lateral"]]
    df_multihop = pd.concat([df_multihop, df_new_qm], ignore_index=True)
    return list(df_new_qm["index"]), df_question, df_multihop


def relations_around(anchor, df_osm):
    """Template relations around `anchor`, a POI, an area or a street: (function, relation text, poi_ids)."""
    geometry = anchor.geometry
    if geometry.geom_type in ("Polygon", "MultiPolygon"):
        yield "pois_inside_area", "inside", pois_inside_area(df_osm, geometry).poi_id
        yield "pois_near_border", "at the border of", pois_near_border(df_osm, geometry, 200).poi_id
        inside = df_osm[df_osm.geometry.within(geometry)]
        for direction, sector in segregate_pois(inside, geometry, ["north", "east", "west", "south"]).items():
            yield "segregate_pois", f"at the {direction} of", sector.poi_id
    elif "LineString" in geometry.geom_type:
        yield "along_street", "following", along_street(df_osm, geometry, 50).poi_id
    else:
        yield "near_sql", "near", near_sql(df_osm, geometry.x, geometry.y).poi_id
        for distance, batch in near_metric_sql(df_osm, geometry.x, geometry.y, [100, 300, 500, 1000]).groupby("distance"):
            yield "near_metric_sql", f"at less than {distance} meters from", batch.poi_id
        for direction, sector in pack_by_direction(df_osm, geometry.x, geometry.y).groupby("direction"):
            yield "pack_by_direction", f"to the {direction} of", sector.poi_id

def shift_keys(d, head):
    return {1: head, **{k + 1: v for k, v in d.items()}}

def add_backward_hop(id_question, df_osm, df_question, df_multihop, nb_q_by_feat=10):
    question = get_question(id_question, df_question, df_multihop)
    answers = defaultdict(set)
    for _, anchor in df_osm.loc[question["answers"]].iterrows():
        for function, relation, ids in relations_around(anchor, df_osm):
            answers[function, relation] |= set(ids) - {anchor.poi_id}
    df_geo = pd.DataFrame([{"function": function, "query": f"pois {relation} {question['query_texte']}",
                            "results_poi_id": sorted(ids)} for (function, relation), ids in answers.items()])
    
    df_q2 = question_geo_semantic(df_geo, df_osm, FEATURES, {1: 1, 2: 1, 3: 1}, nb_q_by_feat)
    start = next_label(df_question, df_multihop)
    new_q_index = list(range(start, start + len(df_q2)))
    df_q2 = df_q2.set_axis(new_q_index)

    new_template = shift_keys(question["template"] or {1: 1, 2: 0}, 1)
    base_dic = shift_keys(question["dic_query"] or {}, 0)
    dic_queries = [{**base_dic, 1: k} for k in range(len(df_q2))]

    df_new_qm = pd.DataFrame({
        "index": df_q2.index,
        "index_origin": [id_question]*len(df_q2.index), 
        "features": df_q2["features"],
        "query_texte": df_q2["query_texte"].to_numpy(),
        "answers": df_q2["results_features_pois_id"].to_numpy(),
        "values": df_q2["values"].to_numpy(),
        "template": [dict(new_template) for _ in range(len(df_q2.index))],
        "dic_query": dic_queries,
        })
    df_multihop = pd.concat([df_multihop, df_new_qm], ignore_index=True).rename_axis("index")
    return new_q_index, df_multihop


def keep_questions(question_index, max_questions, random_generator):
    """At most `max_questions` labels drawn at random from `question_index`."""
    if len(question_index) <= max_questions:
        return list(question_index)
    return list(random_generator.choice(question_index, size=max_questions, replace=False))


def make_multihop_questions(
        id_question, 
        question_template, 
        df_question, 
        df_osm, 
        df_street,
        df_area, 
        max_questions_per_node=3,
        seed=0,
    ):
    """Build multihop questions following `question_template`, with `add_backward_hop` and `add_lateral_hop`.

    Args:
        max_questions_per_node (int): questions of a node the chain continues from; every question
            created by the hops stays in df_multihop.

    Returns:
        tuple: labels of the last node, `df_question`, `df_multihop`.
    """
    df_question = format_df_question(df_question)
    df_all = make_df_all(df_osm, df_street, df_area)

    random_generator = np.random.default_rng(seed)
    df_multihop = pd.DataFrame()
    nodes = sorted(question_template, reverse=True)
    backward_index = [id_question]
    for node in nodes[1:]:
        print(node)
        nb_hop = question_template[node]
        if nb_hop >= 2:
            lateral_index = []
            for index in backward_index:
                new_index, df_question, df_multihop = add_lateral_hop(index, nb_hop, df_question, df_osm, df_all, df_multihop)
                lateral_index += new_index
            backward_index = keep_questions(lateral_index, max_questions_per_node, random_generator)
        next_index = []
        if node > 1:
            for index in backward_index:
                new_index, df_multihop = add_backward_hop(index, df_osm, df_question, df_multihop)
                next_index += new_index
            backward_index = keep_questions(next_index, max_questions_per_node, random_generator)
    
    return backward_index, df_question, df_multihop