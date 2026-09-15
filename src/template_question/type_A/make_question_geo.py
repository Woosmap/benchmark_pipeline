from src.template_question.type_A import *
from collections import defaultdict
from itertools import combinations
import pandas as pd

def make_question_geo(df, nb_q, list_template=['area_border', 'area_direction', 'area_inside', 'area_outside', 
                                               'point_beetween', 'point_near_cardinal', 'point_near_metric', 
                                               'point_near', 'point_towards', 'street_along', 'street_cross', 
                                               'street_ooposite_side']
                        ):
    """
    on va juste faire le produit cartesien entre les features et les questions geo
    """
    for func in list_template:
        df_question = func(df)

        df_question["answer"]

def question_geo_semantic(df_question, df_osm, nb_q=10, nb_f=2):
    for question in df_question.iterrows():
        id_poi = question["results_poi_id"]
        sub_df = df_osm[df_osm['poi_id'] in id_poi]
        sub

def feature_agregation(df, nb_feature, feature_cols, dropna=True):
    dic_agg = defaultdict(list)
    ids = df["poi_id"]
    rows = []
    for combo in combinations(feature_cols, nb_feature):
        g = df.groupby(list(combo), dropna=dropna, observed=True)
        for values, idx in g.groups.items():
            values = values if isinstance(values, tuple) else (values,)
            rows.append({
                "features": combo,
                "values":   values,
                "pois":     list(ids.loc[idx]),
                "size":     len(idx),
                **dict(zip(combo, values)),
            })

    return (pd.DataFrame(rows)
              .sort_values("size", ascending=False)
              .reset_index(drop=True))


"""
il y a un ratio pour les template
un ratio pour les features 
donc par template il doit y avoir le ratio de fetures 
"""