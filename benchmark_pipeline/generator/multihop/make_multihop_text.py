from benchmark_pipeline.generator.composite.generator_qcomposite import question_en_texte
from benchmark_pipeline.generator.features import *


def attributes_text(features, values, feature_cols):
    """« tapas food with outdoor seating »: the attributes of a question, as in its text."""
    return question_en_texte("pois", tuple(features), tuple(values), feature_cols)


def make_text_multihop(df_multihop, df_question, feature_cols):
    """Add `description_textuelle` to df_multihop: each question as short sentences, one per node,
    from the final question to the starting one.

    `feature_cols` are the features whose `to_text` writes the attributes, as in `question_en_texte`.
    """
    multihop = df_multihop.set_index("index")
    nodes_cache = {}

    def text_of(id_query):
        if id_query in multihop.index:
            return multihop.loc[id_query, "query_texte"]
        return df_question.loc[id_query, "query_texte"]

    def nodes_of(id_query):
        """[{sentence, short}] from the node of `id_query` down to the starting question."""
        if id_query in nodes_cache:
            return nodes_cache[id_query]
        if id_query not in multihop.index:
            question = df_question.loc[id_query]
            nodes = [{"sentence": question["query_texte"],
                      "short": attributes_text(question["features"][:1], question["values"][:1], feature_cols)}]
        else:
            row = multihop.loc[id_query]
            below = nodes_of(row["index_origin"])
            origin_text = text_of(row["index_origin"])
            if isinstance(row.get("index_lateral"), list):
                relations = row["query_texte"][len("[" + origin_text + " and "):-1]
                nodes = [{**below[0], "sentence": below[0]["sentence"] + " and " + relations}] + below[1:]
            else:
                head = attributes_text(row["features"], row["values"], feature_cols)
                relation = row["query_texte"][len(head) + 1:-len(origin_text) - 1]
                nodes = [{"sentence": f"{head} {relation} {below[0]['short']}",
                          "short": attributes_text(row["features"][:1], row["values"][:1], feature_cols)}] + below
        nodes_cache[id_query] = nodes
        return nodes

    def description(id_query):
        sentences = [" ".join(node["sentence"].split()) for node in nodes_of(id_query)]
        return " ".join(sentence[0].upper() + sentence[1:] + "." for sentence in sentences)

    df_multihop = df_multihop.copy()
    df_multihop["description_textuelle"] = [description(id_query) for id_query in df_multihop["index"]]
    return df_multihop
