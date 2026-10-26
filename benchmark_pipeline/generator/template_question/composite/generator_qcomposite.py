import inspect
from itertools import combinations

import numpy as np
import pandas as pd

from benchmark_pipeline.config import FEATURES
from benchmark_pipeline.generator.template_question.ratio import allocate
from benchmark_pipeline.generator.template_question.registry import get_template


list_template = ['make_question_area_border',
                 'make_question_area_inside', 'make_question_area_direction',
                 'make_question_point_between', 'make_question_point_near_cardinal',
                 'make_question_point_near_metric', 'make_question_point_near',
                 'make_question_point_towards', 'make_question_street_along',
                 'make_question_street_cross']

MOTS_GENERIQUES = {"poi", "pois"}


def _groupes_disponibles(reponse, nb_attributs):
    trouves = []
    for colonnes in combinations(reponse.columns, nb_attributs):
        renseignes = reponse.dropna(subset=list(colonnes))
        groupes = renseignes.groupby(list(colonnes), sort=False, observed=True)
        for valeurs, pois in groupes.groups.items():
            if not isinstance(valeurs, tuple):
                valeurs = (valeurs,)
            trouves.append((colonnes, valeurs, list(pois)))
    return trouves


def _to_text_par_colonne(features):
    return {f.COLUMN: f.to_text for f in features if hasattr(f, "to_text")}


def question_en_texte(query, colonnes, valeurs, features=FEATURES):
    """Énoncé en langue naturelle : le « pois » initial cède la place aux attributs.

    Args:
        query (str): Énoncé spatial, par exemple « pois near Gare du Nord ».
        colonnes (tuple[str]): Colonnes des attributs contraints.
        valeurs (tuple): Valeurs attendues, alignées sur `colonnes`.
        features (list[Feature]): Features connues, pour retrouver leur `to_text`.

    Returns:
        str: Par exemple « category restaurant cooking_type italian near Gare
            du Nord ». `query` inchangée si aucun attribut ne rend de texte.
    """
    to_text = _to_text_par_colonne(features)
    fragments = [to_text[c](v) for c, v in zip(colonnes, valeurs) if c in to_text]
    fragments = [f for f in fragments if f]
    if not fragments:
        return query

    premier_mot, *reste = query.split(" ")
    tete = fragments if premier_mot.lower() in MOTS_GENERIQUES else ["place", *fragments]
    return " ".join([*tete, *reste])


def _ligne_composite(question, colonnes, valeurs, pois, features=FEATURES):
    """Assemble une ligne de sortie : la question spatiale, plus sa contrainte."""
    qualificatif = " ".join(f"{c}={v}" for c, v in zip(colonnes, valeurs))
    premier_mot, *reste = question["query"].split(" ")
    return {
        **question,
        "features": colonnes,
        "values": valeurs,
        "results_features_pois_id": pois,
        # Le qualificatif se glisse après le premier mot : « pois cuisine=X near Y ».
        "query_geo_semantic": " ".join([premier_mot, qualificatif, *reste])
                              if qualificatif else question["query"],
        "query_texte": question_en_texte(question["query"], colonnes, valeurs, features),
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
            `results_features_pois_id`, `query_geo_semantic` (énoncé balisé
            `colonne=valeur`) et `query_texte` (énoncé en langue naturelle).
    """
    colonnes_attributs = [getattr(c, "COLUMN", c) for c in feature_cols]
    rng = np.random.default_rng(seed)
    attributs = df_osm.set_index("poi_id")[colonnes_attributs]
    budget = allocate(nb_q_by_feat, ratio_features)
    a_enrichir = rng.random(len(df_question)) < part_avec_features

    lignes = []
    for (_, question), enrichie in zip(df_question.iterrows(), a_enrichir):
        reponse = attributs.reindex(question["results_poi_id"])
        if not enrichie:
            lignes.append(_ligne_composite(question, (), (), list(reponse.index), feature_cols))
            continue
        for nb_attributs, nb_tires in budget.items():
            candidats = _groupes_disponibles(reponse, nb_attributs)
            if not candidats:
                continue
            tires = rng.choice(len(candidats), min(nb_tires, len(candidats)), replace=False)
            lignes += [_ligne_composite(question, *candidats[i], feature_cols) for i in tires]
    return pd.DataFrame(lignes)


def _appeler_template(nom, nb_q, sources):
    """Appelle un générateur du REGISTRY avec les seules sources qu'il accepte.

    Args:
        nom (str): Clé du REGISTRY.
        nb_q (int): Nombre de questions demandé au générateur.
        sources (dict): `df_osm`, `df_area` et `df_streets` disponibles ; chaque
            template n'en déclare qu'une partie dans sa signature.

    Returns:
        DataFrame: Les questions spatiales produites par ce template.
    """
    generateur = get_template(nom)
    attendues = inspect.signature(generateur).parameters
    return generateur(nb_q=nb_q, **{k: v for k, v in sources.items() if k in attendues})


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
        is_paraphrase=False,
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
        DataFrame: Les questions composites. `query` garde l'énoncé spatial seul,
            `query_geo_semantic` l'énoncé balisé `colonne=valeur`, et
            `query_texte` l'énoncé en langue naturelle.
    """
    ratio_features = ratio_features or {1: 1, 2: 1, 3: 1}
    sources = {"df_osm": df_osm, "df_area": df_area, "df_streets": df_streets}
    df_geo_q = pd.concat(
        [_appeler_template(nom, nb_q_by_temp, sources) for nom in list_template],
        ignore_index=True)

    if is_paraphrase:
        data = question_geo_semantic(df_geo_q, df_osm, feature_cols, ratio_features,
                                 nb_q_by_feat, part_avec_features, seed)
        return make_paraphrase(data)
    else: return question_geo_semantic(df_geo_q, df_osm, feature_cols, ratio_features,
                                 nb_q_by_feat, part_avec_features, seed)
