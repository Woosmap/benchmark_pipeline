import numpy as np
import pandas as pd

from benchmark_pipeline.generator.template_question.multihop.add_hop import add_backward_hop, add_lateral_hop
from benchmark_pipeline.config import MULTIHOP_COLUMNS, SEED, MAX_QUERY_NODES, NB_QUERY_FEATURE
from benchmark_pipeline.utils.dicts_to_lists import dicts_to_lists
from benchmark_pipeline.generator.template_question.multihop.make_multihop_text import make_text_multihop
from benchmark_pipeline.path import DATA


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
        max_questions_per_node=MAX_QUERY_NODES,
        seed=SEED,
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
                new_index, df_question_light, df_multihop = add_lateral_hop(index, nb_hop, df_question_light, df_osm, df_street, df_area, df_multihop, seed=seed)
                lateral_index += new_index
            backward_index = keep_questions(lateral_index, max_questions_per_node, random_generator)
        next_index = []
        if node > 1:
            for index in backward_index:
                new_index, df_multihop = add_backward_hop(index, df_osm, df_question_light, df_multihop, NB_QUERY_FEATURE, seed=seed)
                next_index += new_index
            backward_index = keep_questions(next_index, max_questions_per_node, random_generator)
    
    return backward_index, df_question_light, df_multihop

def make_multihop_dataset(
    nb_seed_question: int,
    repo: str,
    template: dict[int, int],
    df_question: pd.DataFrame,
    df_osm: pd.DataFrame,
    df_street: pd.DataFrame,
    df_area: pd.DataFrame,
    seed: int = 0,
) -> None:
    """Build the multi-hop question dataset and save it to parquet.

    Samples ``nb_seed_question`` starting questions from ``df_question`` and, for each
    one, generates a chain of multi-hop questions following ``template``.
    The resulting chains are then turned into natural-language text and
    written, together with the updated question table, to
    ``DATA / repo / "multihop"``.

    Args:
        nb_seed_question: Number of starting questions sampled from ``df_question``.
        repo: repo name for registration
        template: Generation template mapping each hop depth to the value
            used by ``make_multihop_questions`` at that depth.
        df_question: Single-hop question table; it is enriched by each call
            to ``make_multihop_questions``.
        df_osm: OSM POI table.
        df_street: Street geometry table.
        df_area: Area (zone) geometry table.
        seed: Seed of every draw: starting questions, hops and the questions they generate.

    Returns:
        None. Writes two files to ``DATA / repo / "multihop"``:
        ``df_question.parquet`` (restricted to ``MULTIHOP_COLUMNS``) and
        ``df_multihop_15_hops.parquet``.
    """
    df_multihop: pd.DataFrame | None = None
    for id_start in df_question.sample(nb_seed_question, random_state=seed).index:
        _, df_question, df_multihop = make_multihop_questions(
            id_start, template, df_question, df_osm, df_street, df_area,
            seed=seed, df_multihop=df_multihop,
        )
    df_multihop = make_text_multihop(df_multihop, df_question)

    multihop_dir = DATA / repo / "multihop"
    multihop_dir.mkdir(parents=True, exist_ok=True)
    dicts_to_lists(df_question[MULTIHOP_COLUMNS]).to_parquet(multihop_dir / "df_question.parquet")
    dicts_to_lists(df_multihop).to_parquet(multihop_dir / "df_multihop.parquet")