import re

import pandas as pd

def add_anchors(df_question):
    list_id_anchors = df_question['anchors_id']
    list_name_anchors = df_question['anchors_name']
    list_type_anchors = df_question['anchors_type']
    target_anchors = []
    for i, (anc_id, name, anc_type) in enumerate(zip(list_id_anchors, list_name_anchors, list_type_anchors)):
        target_anchors.append({"id": anc_id, "name": name, "type": anc_type})
    df_question["anchors"] = target_anchors
    return df_question
