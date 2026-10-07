import inspect
from itertools import combinations

import numpy as np
import pandas as pd

from benchmark_pipeline.config import FEATURES
from benchmark_pipeline.generator.template_question import geospatial  # noqa: F401
from benchmark_pipeline.generator.template_question.ratio import allocate
from benchmark_pipeline.generator.template_question.registry import REGISTRY


list_template = ['make_question_point_near']
"""
['make_question_area_border',
                 'make_question_area_inside', 
                 'make_question_point_between', 'make_question_point_near_cardinal',
                 'make_question_point_near_metric', 'make_question_point_near',
                 'make_question_point_towards', 'make_question_street_along',
                 'make_question_street_cross', 'make_question_street_opposite_side']
"""
#'make_question_area_outside', 'make_question_area_direction'

def _groupes_disponibles(reponse, nb_attributs):
    """Contraintes d'attributs portées par les POIs d'une réponse spatiale.

    Args:
        reponse (DataFrame): POIs de la réponse, indexés par `poi_id`, dans
            l'ordre de pertinence spatiale.
        nb_attributs (int): Nombre d'attributs à combiner.

    Returns:
        list[tuple]: `(colonnes, valeurs, poi_ids)`. Les `poi_ids` sont un
            sous-ensemble de la réponse, dans son ordre.
    """
    trouves = []
    for colonnes in combinations(reponse.columns, nb_attributs):
        renseignes = reponse.dropna(subset=list(colonnes))
        # `sort=False` : l'ordre d'apparition est l'ordre de pertinence spatiale.
        groupes = renseignes.groupby(list(colonnes), sort=False, observed=True)
        trouves += [
            (colonnes, valeurs if isinstance(valeurs, tuple) else (valeurs,), list(pois))
            for valeurs, pois in groupes.groups.items()
        ]
    return trouves


def _ligne_composite(question, colonnes, valeurs, pois):
    """Assemble une ligne de sortie : la question spatiale, plus sa contrainte."""
    qualificatif = " ".join(f"{c}={v}" for c, v in zip(colonnes, valeurs))
    mots = question["query"].split(" ")
    return {
        **question,
        "features": colonnes,
        "values": valeurs,
        "results_features_pois_id": pois,
        # Le qualificatif se glisse après le premier mot : « pois cuisine=X near Y ».
        "query_geo_semantic": " ".join(mots[:1] + [qualificatif] + mots[1:])
                              if qualificatif else question["query"],
    }


def question_geo_semantic(df_question, df_osm, feature_cols, ratio_features,
                          nb_q_by_feat, part_avec_features=1.0, seed=42):
    """Croise chaque question spatiale avec des contraintes d'attributs.

    Args:
        df_question (DataFrame): Questions spatiales, avec `query` et
            `results_poi_id`.
        df_osm (DataFrame): Corpus, avec `poi_id` et les colonnes d'attributs.
        feature_cols (list[str] | list[Feature]): Colonnes d'attributs éligibles.
        ratio_features (dict[int, float]): Poids par nombre d'attributs combinés.
        nb_q_by_feat (int): Contraintes tirées par question enrichie.
        part_avec_features (float): Part des questions à enrichir, dans [0, 1].
        seed (int): Graine du tirage.

    Returns:
        DataFrame: Les colonnes de `df_question`, plus `features`, `values`,
            `results_features_pois_id` et `query_geo_semantic`.
    """
    colonnes_attributs = [getattr(c, "COLUMN", c) for c in feature_cols]
    rng = np.random.default_rng(seed)
    attributs = df_osm.set_index("poi_id")[colonnes_attributs]
    budget = allocate(nb_q_by_feat, ratio_features)
    lignes = []
    for (_, question), enrichie in zip(df_question.iterrows(),
                                       rng.random(len(df_question)) < part_avec_features):
        reponse = attributs.reindex(question["results_poi_id"])
        if not enrichie:
            lignes.append(_ligne_composite(question, (), (), list(reponse.index)))
            continue
        for nb_attributs, nb_tires in budget.items():
            candidats = _groupes_disponibles(reponse, nb_attributs)
            tires = rng.choice(len(candidats), min(nb_tires, len(candidats)),
                               replace=False) if candidats else []
            lignes += [_ligne_composite(question, *candidats[i]) for i in tires]
    return pd.DataFrame(lignes)


def make_question_composite(
        df_osm,
        df_area,
        df_streets,
        nb_q_by_temp=100,
        nb_q_by_feat=10,
        list_template=list_template,
        ratio_features=None,
        part_avec_features=1.0,
        feature_cols=FEATURES,
        seed=42,
    ):
    """Produit le jeu complet de questions composites.

    Enchaîne les deux étapes : générer les questions spatiales, puis croiser
    chacune avec les contraintes d'attributs que porte sa propre réponse.

    Args:
        df_osm (GeoDataFrame): Corpus de POIs, avec les colonnes d'attributs.
        df_area (GeoDataFrame | None): Zones, exigées par les templates `area_*`.
        df_streets (GeoDataFrame | None): Rues, exigées par les `street_*`.
        nb_q_by_temp (int): `nb_q` transmis à chaque générateur spatial.
        nb_q_by_feat (int): Contraintes tirées par question spatiale enrichie.
        list_template (list[str]): Clés de `REGISTRY` à exécuter.
        ratio_features (dict[int, float] | None): Poids par nombre d'attributs
            combinés ; équirépartition sur 1, 2 et 3 si None.
        part_avec_features (float): Part des questions à enrichir, dans [0, 1].
        feature_cols (list[str] | list[Feature]): Colonnes d'attributs éligibles.
        seed (int): Graine du tirage sémantique.

    Returns:
        DataFrame: Les questions composites, `query_geo_semantic` portant l'énoncé
            enrichi et `query` conservant l'énoncé spatial seul.
    """
    ratio_features = ratio_features or {1: 1, 2: 1, 3: 1}
    sources = {"df_osm": df_osm, "df_area": df_area, "df_streets": df_streets}
    df_geo_q = pd.concat(
        [REGISTRY[name](nb_q=nb_q_by_temp,
                        **{k: v for k, v in sources.items()
                           if k in inspect.signature(REGISTRY[name]).parameters})
         for name in list_template],
        ignore_index=True)
    return question_geo_semantic(df_geo_q, df_osm, feature_cols, ratio_features,
                                 nb_q_by_feat, part_avec_features, seed)
