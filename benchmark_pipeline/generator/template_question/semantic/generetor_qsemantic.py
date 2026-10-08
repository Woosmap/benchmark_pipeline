"""Questions sémantiques : décrire un POI par ses attributs, sans géographie.

Là où `geospatial` interroge une relation spatiale — « le café le plus proche de
X » — ce module interroge des **attributs** : « un restaurant italien avec
terrasse ». Aucune coordonnée n'intervient ; le corpus est lu comme une table
d'attributs, et la difficulté vient du nombre de contraintes combinées.

Les attributs ne sont pas des colonnes brutes mais des objets `Feature`
(cf. `benchmark_pipeline/generator/features.py`), qui savent deux choses : lire la valeur d'un POI
(`check`) et la traduire en fragment de phrase (`to_text`). Cette indirection
permet aux features dérivées — « à moins de 200 m de la Seine » — d'exister à
côté des tags OSM bruts, et laisse `to_text` rendre `None` quand l'attribut
n'est pas renseigné, ce qui écarte le POI au lieu de produire un énoncé bancal.
"""
from collections import defaultdict
from itertools import combinations

from pandas import DataFrame

from benchmark_pipeline.config import *


def generate_question(poi, n_features, nb_question, features=FEATURES):
    """
    Génère toutes les combinaisons de questions possibles pour un POI donné.

    Évalue chaque feature sur le POI, collecte les fragments de texte disponibles,
    puis construit toutes les combinaisons de `n_features` fragments. Chaque
    combinaison forme une question potentielle de benchmark sous la forme d'une
    liste [catégorie, fragment_1, ..., fragment_n], triée alphabétiquement pour
    garantir la reproductibilité.

    Args:
        poi (Series): Ligne d'un GeoDataFrame représentant un point d'intérêt.
        n_features (int): Nombre de features à combiner dans chaque question.
        features (list[Feature]): Liste des features à évaluer sur le POI.

    Returns:
        list[list[str]] | None: Liste de combinaisons [catégorie, *fragments], ou
            None si le POI ne possède pas assez de features renseignées.

    Note:
        Le nombre de fragments peut dépasser `len(features)` : une feature
        multi-valuée rend une liste, aplatie par `extend`. C'est pourquoi la
        garde porte sur `len(available)` et non sur `len(features)`.
    """
    available = []
    for feature in features:
        value = feature.check(poi)
        text = feature.to_text(value)
        if text is None or text == "nan" or text == "":
            continue
        if isinstance(text, list):
            available.extend((t, feature.COLUMN, v) for t, v in zip(text, value))
        else:
            available.append((text, feature.COLUMN, value))

    if len(available) < n_features:
        return None

    question = []
    features = []
    for combo in combinations(available, n_features):
        combo = sorted(combo, key=lambda item: item[0])
        question.append([t for t, _, _ in combo])
        features.append({col: val for _, col, val in combo})
        #out.append({
        #    "question":    [poi["category"]] + [t for t, _, _ in combo],
        #    "features": [(col, val) for _, col, val in combo],
        #})
    return question, features



def make_questions_semantic(df_osm, features, nb_question, rng=None, seed=42):
    """Tire un jeu de questions sémantiques, stratifié par nombre d'attributs.

    La strate n'est pas la catégorie de POI mais le **nombre d'attributs
    combinés** : une question à un attribut (« un restaurant italien ») est plus
    facile qu'une à trois (« un restaurant italien avec terrasse et salle »).
    Répartir le budget entre ces strates règle donc la difficulté du jeu.

    Le tirage procède par rejet : on tire une catégorie, puis un POI de cette
    catégorie, et on garde la question si le POI porte assez d'attributs
    renseignés. Les POIs mal documentés — la majorité dans OSM — sont ainsi
    écartés sans qu'il faille les filtrer en amont.

    Args:
        df_osm (DataFrame): Corpus de POIs, avec `poi_id`, `category` et les
            colonnes que lisent les `features`.
        features (list[Feature]): Features à évaluer sur chaque POI.
        nb_question (int): Nombre de question par nb de features
        ratio (dict[int, int] | None): Nombre de questions par strate, la clé
            étant le nombre d'attributs combinés. Par défaut, équirépartition
            sur 1..len(FEATURES).
        rng (Generator | None): Générateur aléatoire à réutiliser. Créé depuis
            `seed` s'il est absent.
        seed (int): Graine, utilisée seulement si `rng` est absent.

    Returns:
        DataFrame: Une ligne par question, avec `question` (la liste
            [catégorie, *fragments]), `poi` (le `poi_id` qui l'a inspirée) et
            `nb_feature` (la strate).
    """
    dic_question = defaultdict(list)
    for _, row in df_osm.iterrows():
        for nb_feat in range(1, len(features)+1):
            genere = generate_question(row, nb_feat, features, nb_question)
            if not genere:
                continue
            questions, contraintes = genere
            dic_question['poi'].extend([row['poi_id']]*len(questions))
            dic_question['question'].extend(questions)
            dic_question["features"].extend(contraintes)
            dic_question["nb_feature"].extend([nb_feat]*len(questions))
    df_out = DataFrame(dic_question)
    df_out["question_key"] = df_out["question"].map(tuple)
    df_out_group = (df_out
        .groupby("question_key", as_index=False)
        .agg(
            question=("question", "first"),
            nb_feature=("nb_feature", "first"),
            features=("features", "first"),
            pois=("poi", list),
            nb_answer=("poi", "size"),
        ))
    return df_out_group
