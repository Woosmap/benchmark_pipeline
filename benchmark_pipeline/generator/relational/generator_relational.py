import random

import pandas as pd

from benchmark_pipeline.generator.relational.graph import path_subgraph, shortest_path
from benchmark_pipeline.config import FORMAT_DIRECTION, RATIO_TEST
from benchmark_pipeline.generator.relational.relation import compute_relation, direction_text
from benchmark_pipeline.path import DATA


def build_graph(area_pois, nb_pois, nb_relations):
    """Graph of `nb_pois` POIs drawn at random in the area, linked by `nb_relations` relations.

    The relations start with a loop through all the POIs: each POI has a relation
    to another POI, and any POI can be reached from any other.
    At most one relation is kept per pair of POIs, since the other direction follows (same distance, opposite direction).
    """
    if nb_relations < nb_pois + 2:
        raise ValueError(f"at least nb_pois + 2 = {nb_pois + 2} relations are needed")
    if nb_relations > nb_pois * (nb_pois - 1) // 2:
        raise ValueError(f"{nb_pois} POIs only have {nb_pois * (nb_pois - 1) // 2} possible pairs")

    # the POIs: an id (name and category) and coordinates in meters
    pois = []
    for position in random.sample(range(len(area_pois)), nb_pois):
        row = area_pois.iloc[position]
        pois.append({"id": f"{row['poi_name']} ({row['category']})",
                     "x": row.geometry.x, "y": row.geometry.y})

    # every possible relation, from one POI to another
    all_relations = []
    for start in pois:
        for end in pois:
            if start["id"] != end["id"]:
                distance, direction_degrees, direction_cardinal = compute_relation(
                    start["x"], start["y"], end["x"], end["y"])
                all_relations.append({"from": start["id"], "to": end["id"], "distance": distance,
                                      "direction_degrees": direction_degrees,
                                      "direction_cardinal": direction_cardinal})

    # first the loop: each POI has a relation to the next POI, and the last one to the first
    relations = []
    taken_pairs = []
    for i in range(nb_pois):
        start = pois[i]["id"]
        end = pois[(i + 1) % nb_pois]["id"]
        for relation in all_relations:
            if relation["from"] == start and relation["to"] == end:
                relation["in_loop"] = True
                relations.append(relation)
        taken_pairs.append(sorted([start, end]))

    # then relations drawn at random, up to nb_relations
    random.shuffle(all_relations)
    for relation in all_relations:
        pair = sorted([relation["from"], relation["to"]])
        if len(relations) < nb_relations and pair not in taken_pairs:
            relation["in_loop"] = False
            relations.append(relation)
            taken_pairs.append(pair)

    random.shuffle(relations)   # so that the order of the sentences does not reveal the loop
    coordinates = {}
    for poi in pois:
        coordinates[poi["id"]] = [poi["x"], poi["y"]]
    return {"pois": [poi["id"] for poi in pois], "coordinates": coordinates, "relations": relations}


def pick_answer(graph, path_length):
    """Draws the relation to find and removes it from the graph.

    It is drawn outside the loop: without it, any POI can still be reached from any other,
    so the answer can be deduced from the remaining relations.
    Without it, its two POIs must be linked by a shortest path of `path_length` relations.
    Returns the graph without the answer, the answer and the subgraph of the shortest path,
    or None if no relation of the graph fits.
    """
    candidates = []
    paths = []
    for relation in graph["relations"]:
        if not relation["in_loop"]:
            other_relations = []
            for other in graph["relations"]:
                if other != relation:
                    other_relations.append(other)
            path = shortest_path(other_relations, relation["from"], relation["to"])
            if path is not None and len(path) == path_length:
                candidates.append(relation)
                paths.append(path)
    if not candidates:
        return None
    position = random.randrange(len(candidates))
    drawn = candidates[position]
    path = path_subgraph(paths[position], drawn["from"])

    remaining_relations = []
    for relation in graph["relations"]:
        if relation != drawn:
            remaining_relations.append(relation)

    graph_without_answer = {"pois": graph["pois"], "coordinates": graph["coordinates"],
                            "relations": remaining_relations}
    answer = {"from": drawn["from"], "to": drawn["to"], "distance": drawn["distance"],
              "direction_degrees": drawn["direction_degrees"], "direction_cardinal": drawn["direction_cardinal"]}
    return graph_without_answer, answer, path


def write_question(graph, answer, format_direction=FORMAT_DIRECTION):
    """One sentence per relation of the graph, then the question about the relation to find."""
    sentences = []
    for relation in graph["relations"]:
        sentences.append(f"From the poi {relation['from']}, the poi {relation['to']} is {relation['distance']} m "
                         f"away in direction {direction_text(relation, format_direction)}.")
    sentences.append(f"How far and in which direction is the poi {answer['to']} "
                     f"from the poi {answer['from']}?")
    return "\n".join(sentences)


def make_relational_dataset(repo, area_pois, nb_questions, nb_pois, nb_relations, path_length, seed,
                            ratio_test=RATIO_TEST, format_direction=FORMAT_DIRECTION):
    """The benchmark table: one row per question, with its graph, the subgraph of the shortest path,
    its text, its answer and its two labels (direction as a cardinal point and in degrees).

    The two POIs of each answer are linked by a shortest path of `path_length` relations.
    When a graph has no such relation, another one is drawn.
    Each question has its own graph: the first `ratio_test` of them are written to
    `DATA / repo / "question" / "test" / "relational"`, the others to `.../"train"/...`.
    """
    if path_length < 2:
        raise ValueError("path_length must be at least 2: the direct relation is hidden")
    random.seed(seed)
    rows = []
    attempts = 0
    while len(rows) < nb_questions:
        attempts += 1
        if attempts > 100 * nb_questions:
            raise ValueError(f"too many graphs without a path of {path_length} relations: "
                             "lower path_length or change nb_pois and nb_relations")
        graph = build_graph(area_pois, nb_pois, nb_relations)
        result = pick_answer(graph, path_length)
        if result is None:
            continue
        graph_without_answer, answer, path = result
        question = write_question(graph_without_answer, answer, format_direction)
        rows.append({"graph": graph_without_answer, "path": path, "questions": question, "answer": answer,
                     "label_cardinal": {"distance": answer["distance"], "direction": answer["direction_cardinal"]},
                     "label_degrees": {"distance": answer["distance"], "direction": answer["direction_degrees"]}})
    n_test = int(len(rows) * ratio_test)
    for split, split_rows in (("test", rows[:n_test]), ("train", rows[n_test:])):
        output_dir = DATA / repo / "question" / split / "relational"
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(split_rows).to_parquet(output_dir / "df_relational.parquet")
