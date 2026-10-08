import re

import pandas as pd

#: Colonnes qui portent le libellé d'une ancre, c'est-à-dire un nom que
#: l'énoncé cite et que le masque doit donc couvrir. `poi_y_name` en fait
#: partie : `street_opposite_side` y range son POI de référence, que son énoncé
#: nomme (« X across rue Y from Z ») — l'omettre laissait ce nom non masqué.
ANCHOR_COL = re.compile(r"(anchor|area|street)(_[ab])?_name|poi_y_name")

def get_anchors(row: pd.Series) -> list[str]:
    return [row[col] for col in row.index
            if ANCHOR_COL.fullmatch(col) and pd.notna(row[col])]

def make_mask(tokens_query, tokens_anchors):
    """Étiquette les tokens d'une question selon qu'ils appartiennent à une ancre.

    Args:
        tokens_query (list[int]): Tokens d'une question.
        tokens_anchors (list[list[int]]): Tokens de chacune de ses ancres, produits
            avec la même convention de tokens spéciaux que `tokens_query`.

    Returns:
        list[int]: Même longueur que `tokens_query`. 0 hors ancre, 1 sur le premier
            token d'une ancre, 2 sur les suivants.
    """
    mask = [0] * len(tokens_query)
    for tokens_ancre in tokens_anchors:
        longueur = len(tokens_ancre)
        if longueur == 0:
            continue
        for debut in range(len(tokens_query) - longueur + 1):
            if tokens_query[debut:debut + longueur] != tokens_ancre:
                continue
            if any(mask[debut:debut + longueur]):
                continue
            mask[debut] = 1
            mask[debut + 1:debut + longueur] = [2] * (longueur - 1)
    return mask

def add_anchors_mask(df_question, sentence_model):
    list_anchors = []
    for _, row in df_question.iterrows():
        list_anchors.append(get_anchors(row))
    list_tokens_query = sentence_model.tokenizer(list(df_question["query"]))["input_ids"]
    list_tokens_anchors = [sentence_model.tokenizer(anchors, add_special_tokens=False)["input_ids"] if len(anchors) else []
                           for anchors in list_anchors]
    list_mask = [make_mask(tokens_query, tokens_anchors)
                 for tokens_query, tokens_anchors in zip(list_tokens_query, list_tokens_anchors)]
    df_question["anchors"] = list_anchors
    df_question["mask_token_anchors"] = list_mask
    return df_question
