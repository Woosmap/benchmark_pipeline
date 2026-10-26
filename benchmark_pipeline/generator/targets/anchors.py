

def add_anchors(df_question):
    list_id_anchors = df_question['anchors_id']
    list_name_anchors = df_question['anchors_name']
    list_type_anchors = df_question['anchors_type']
    """
    target_anchors = []
    for i, (anc_id, name, anc_type) in enumerate(zip(list_id_anchors, list_name_anchors, list_type_anchors)):
        target_anchors.append({"id": anc_id, "name": name, "type": anc_type})
    df_question["anchors"] = target_anchors
    """
    pairs = [tuple(types) for types in list_type_anchors]
    mapping = {p: k for k, p in enumerate(sorted(set(pairs)))}
    list_target = [mapping[p] for p in pairs]
    df_question['anchors_type_target'] = list_target
    return df_question
