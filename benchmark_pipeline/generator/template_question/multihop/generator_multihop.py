import numpy as np
import pandas as pd

from benchmark_pipeline.generator.template_question.multihop.add_hop import add_backward_hop, add_lateral_hop
from benchmark_pipeline.config import MULTIHOP_COLUMNS


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
        df_multihop=None,
    ):
    """Build multihop questions following `question_template`, with `add_backward_hop` and `add_lateral_hop`.

    Args:
        max_questions_per_node (int): questions of a node the chain continues from; every question
            created by the hops stays in df_multihop.
        df_multihop (DataFrame | None): multihop questions of previous calls; the new ones are appended.

    Returns:
        tuple: labels of the last node, `df_question`, `df_multihop` without text
            (call `make_text_multihop` once at the end).
    """
    df_question_light = (df_question
             .rename(columns={"results_features_pois_id": "answers"})
             .assign(template=[{1: 1, 2: 0}] * len(df_question))
             .assign(dic_query=[{1: index, 2: None} for index in df_question.index])
             [MULTIHOP_COLUMNS])
    
    random_generator = np.random.default_rng(seed)
    df_multihop = pd.DataFrame() if df_multihop is None else df_multihop
    nodes = sorted(question_template, reverse=True)
    backward_index = [id_question]
    for node in nodes[1:]:
        nb_hop = question_template[node]
        if nb_hop >= 2:
            lateral_index = []
            for index in backward_index:
                new_index, df_question_light, df_multihop = add_lateral_hop(index, nb_hop, df_question_light, df_osm, df_street, df_area, df_multihop)
                lateral_index += new_index
            backward_index = keep_questions(lateral_index, max_questions_per_node, random_generator)
        next_index = []
        if node > 1:
            for index in backward_index:
                new_index, df_multihop = add_backward_hop(index, df_osm, df_question_light, df_multihop)
                next_index += new_index
            backward_index = keep_questions(next_index, max_questions_per_node, random_generator)
    
    return backward_index, df_question_light, df_multihop