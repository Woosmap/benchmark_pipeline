"""Questions composites : une relation spatiale croisée avec des attributs OSM.

Un template `geospatial` produit « cafe near poi5_cafe » ; ce module y ajoute une
contrainte sémantique tirée des colonnes d'attributs du corpus, pour obtenir
« cafe cuisine=italian near poi5_cafe ». La question reste géographique, mais sa
vérité terrain est restreinte aux POIs qui portent aussi les bons attributs.

Le procédé est un produit cartésien en deux temps :

1. `make_question_geo` exécute les douze générateurs de `geospatial` et
   concatène leurs sorties — une banque de questions purement spatiales ;
2. pour chacune, `question_geo_semantic` regroupe les POIs de sa réponse par
   combinaison d'attributs, et croise la question avec chaque groupe obtenu.

Ce module dépend donc de `geospatial` et jamais l'inverse : c'est ce qui justifie
qu'il vive dans `composite/` plutôt qu'au milieu des templates spatiaux, où sa
première version importait le paquet qui la contenait.
"""

from collections import defaultdict
from itertools import combinations
import numpy as np
import pandas as pd
import inspect

from benchmark_pipeline.generator.template_question import geospatial
from benchmark_pipeline.generator.template_question.registry import REGISTRY
from benchmark_pipeline.generator.template_question.ratio import allocate
from benchmark_pipeline.config import FEATURES
from benchmark_pipeline.generator.features import Feature


#: Les douze générateurs spatiaux, désignés par leur clé dans `REGISTRY` — donc
#: par leur nom de fonction, et non par le nom court du registre de `schema.py`.
#: Les deux registres cohabitent : celui-ci est peuplé par le décorateur
#: `@template` à l'import de `geospatial`.

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

def feature_agregation(
        df, 
        nb_feature, 
        nb_question, 
        features_cols, 
        dropna, 
        seed
    ):
    """Regroupe les POIs par combinaison de valeurs d'attributs.

    Parcourt toutes les combinaisons de `nb_feature` colonnes prises parmi
    `features_cols`, et pour chacune produit une ligne par valeur distincte
    rencontrée. Une ligne décrit donc un sous-ensemble homogène du corpus :
    « les POIs dont `cuisine` vaut `italian` », avec la liste de leurs
    identifiants.

    Ces lignes servent de contrainte sémantique : croisées avec une question
    spatiale, elles en restreignent la réponse.

    Args:
        df (DataFrame): POIs candidats, avec `poi_id` et les colonnes
            d'attributs citées par `features_cols`.
        nb_feature (int): Nombre d'attributs combinés — 1 donne « cuisine=X »,
            2 donne « cuisine=X et outdoor_seating=Y ».
        nb_question (int): Nombre maximal de groupes retournés.
        features_cols (list[str]): Colonnes d'attributs éligibles.
        dropna (bool): Passé à `groupby` : si faux, les valeurs manquantes
            forment un groupe à part.
        seed (int): Graine de l'échantillonnage final.

    Returns:
        DataFrame: Une ligne par groupe, avec `features` (le tuple de colonnes),
            `values` (le tuple de valeurs), `number features`,
            `results_features_pois_id` (les `poi_id` du groupe), `size`, plus une
            colonne par attribut de la combinaison.
    """
    features_cols = [getattr(f, "COLUMN", f) for f in features_cols]
    rows = []
    for combo in combinations(features_cols, nb_feature):
        df_full = df.dropna(subset=list(combo))
        poi_ids = df_full["poi_id"].to_numpy()
        g = df_full.groupby(list(combo), dropna=dropna, observed=True)
        for values, positions in g.indices.items():
            values = values if isinstance(values, tuple) else (values,)
            rows.append({
                "features": combo,
                "values": values,
                "number features": nb_feature,
                "results_features_pois_id": (poi_ids, positions),
                "size": len(positions),
                **dict(zip(combo, values)),
            })

    # Aucun groupe : frame vide sans colonnes, comme avant. L'affectation qui suit
    # lèverait un KeyError dessus.
    if not rows:
        return pd.DataFrame(rows)

    out = pd.DataFrame(rows).sample(n=min(nb_question, len(rows)), random_state=seed)
    out["results_features_pois_id"] = [
        list(ids[positions]) for ids, positions in out["results_features_pois_id"]
    ]
    return out

def prepare_feature_groups(df, features_cols, nb_features, dropna):
    """Associe à chaque POI son code de groupe, pour chaque combinaison d'attributs.

    Le regroupement est une propriété du corpus, pas de la question : le calculer
    une seule fois évite de refaire un `dropna` et un `groupby` par question et par
    combinaison.

    Args:
        df (DataFrame): Corpus complet, avec `poi_id` et les colonnes d'attributs.
        features_cols (list[str] | list[Feature]): Colonnes d'attributs éligibles.
        nb_features (iterable[int]): Tailles de combinaison à préparer.
        dropna (bool): Passé à `groupby`, comme dans `feature_agregation`.

    Returns:
        dict: `{combo: (codes, cles)}` — `codes` donne le groupe de chaque ligne de
            `df`, -1 si un attribut de la combinaison est manquant ; `cles[code]`
            donne le tuple de valeurs correspondant.
    """
    features_cols = [getattr(f, "COLUMN", f) for f in features_cols]
    tables = {}
    for nb in set(nb_features):
        for combo in combinations(features_cols, nb):
            # `dropna(subset=...)` retirait déjà ces lignes avant le groupby : on
            # reproduit le masque, sans la copie de frame qu'il entraînait.
            valide = df[list(combo)].notna().all(axis=1).to_numpy()
            groupes = df.loc[valide].groupby(list(combo), dropna=dropna, observed=True)
            codes = np.full(len(df), -1, dtype=np.int64)
            codes[valide] = groupes.ngroup().to_numpy()
            cles = [k if isinstance(k, tuple) else (k,) for k in groupes.indices]
            tables[combo] = (codes, cles)
    return tables


def feature_agregation_indexee(poi_ids, positions, tables, nb_feature, nb_question, seed):
    """Version de `feature_agregation` qui lit les groupes précalculés.

    Args:
        poi_ids (ndarray): `poi_id` du corpus, dans l'ordre des lignes.
        positions (ndarray): Positions, triées, des POIs retenus par la question.
        tables (dict): Sortie de `prepare_feature_groups`.
        nb_feature (int): Nombre d'attributs combinés.
        nb_question (int): Nombre maximal de groupes retournés.
        seed (int): Graine de l'échantillonnage final.

    Returns:
        DataFrame: Mêmes colonnes que `feature_agregation`.
    """
    rows = []
    for combo, (codes, cles) in tables.items():
        if len(combo) != nb_feature:
            continue
        sous_codes = codes[positions]
        garde = sous_codes >= 0
        sous_codes, sous_positions = sous_codes[garde], positions[garde]

        # Trier par code regroupe les lignes d'un même groupe en tranches contiguës ;
        # `kind="stable"` garde l'ordre du corpus à l'intérieur de chaque tranche.
        ordre = np.argsort(sous_codes, kind="stable")
        sous_codes, sous_positions = sous_codes[ordre], sous_positions[ordre]
        groupes, debuts, tailles = np.unique(sous_codes, return_index=True, return_counts=True)

        for code, debut, taille in zip(groupes, debuts, tailles):
            values = cles[code]
            rows.append({
                "features": combo,
                "values": values,
                "number features": nb_feature,
                # Tranche de positions : la liste d'ids n'est construite qu'au tirage.
                "results_features_pois_id": sous_positions[debut:debut + taille],
                "size": int(taille),
                **dict(zip(combo, values)),
            })

    if not rows:
        return pd.DataFrame(rows)

    out = pd.DataFrame(rows).sample(n=min(nb_question, len(rows)), random_state=seed)
    out["results_features_pois_id"] = [
        list(poi_ids[tranche]) for tranche in out["results_features_pois_id"]
    ]
    return out


def question_geo_semantic(
        df_question,
        df_osm,
        nb_q, 
        features_cols, 
        ratio, 
        seed,
        dropna,
    ):
    """Croise chaque question spatiale avec les groupes d'attributs de sa réponse.

    Pour une question donnée, seuls les POIs de sa vérité terrain sont éligibles :
    c'est sur eux qu'on cherche des combinaisons d'attributs. Croiser avec le
    corpus entier produirait des contraintes sans aucune réponse.

    `ratio` stratifie sur le **nombre** d'attributs combinés, pas sur leur nom :
    `{1: 1, 2: 1, 3: 1}` demande autant de questions à un, deux et trois
    attributs. C'est ce qui règle la difficulté du jeu produit — plus il y a
    d'attributs, plus la contrainte est fine et la réponse courte.

    Args:
        df_question (DataFrame): Questions spatiales, avec au moins `query` et
            `results_poi_id`.
        df_osm (DataFrame): Corpus complet, avec les colonnes d'attributs.
        nb_q (int): Budget de groupes sémantiques **par question spatiale**.
        features_cols (list[str]): Colonnes d'attributs éligibles.
        ratio (dict[int, float] | None): Poids par nombre d'attributs. Par
            défaut, équirépartition de 1 à `len(features_cols)`.
        seed (int): Graine transmise à `feature_agregation`.

    Returns:
        DataFrame: Une ligne par couple (question spatiale, groupe d'attributs).
            Colonnes de la question, plus celles de `feature_agregation`.

    TODO: le volume de sortie n'est pas `nb_q` mais de l'ordre de
        `len(df_question) × nb_q` : le budget est consommé une fois par question
        spatiale. Mesuré : 10 questions spatiales et nb_q=10 donnent 30 morceaux
        concaténés. Si `nb_q` doit piloter la taille finale, il faut l'allouer
        sur les questions avant la boucle.
    """
    if len(df_question)==0:
        assert(f"Le tableau {df_question} est vide.")
    balance = allocate(nb_q, ratio)
    tables = prepare_feature_groups(df_osm, features_cols, balance.keys(), dropna)
    rang_poi = pd.Index(df_osm["poi_id"])
    poi_ids = df_osm["poi_id"].to_numpy()
    df_out = []
    for _, question in df_question.iterrows():
        id_poi = question["results_poi_id"]
        positions = np.unique(rang_poi.get_indexer(id_poi))
        positions = positions[positions >= 0]
        for nb_f, nb_row in balance.items():
            answers_poi = feature_agregation_indexee(poi_ids, positions, tables, nb_f, nb_row, seed)
            res = question.to_frame().T.merge(answers_poi, how="cross")
            df_out.append(res)
    return pd.concat(df_out, ignore_index=True)


def make_question_composite(
        df_osm, 
        df_area, 
        df_streets, 
        nb_q_by_temp=100, 
        nb_q_by_feat=10, 
        list_template=list_template, 
        ratio={1: 1, 2: 1, 3: 1}, 
        feature_cols=FEATURES,
        seed=42,
        dropna=True, 
    ):
    """Produit le jeu complet de questions composites.

    Enchaîne les trois étapes : générer les questions spatiales, les croiser avec
    les groupes d'attributs, puis réécrire l'énoncé pour que la contrainte
    sémantique y apparaisse.

    Args:
        df_osm (GeoDataFrame): Corpus de POIs, avec les colonnes d'attributs.
        df_area (GeoDataFrame | None): Zones, exigées par les templates `area_*`.
        df_streets (GeoDataFrame | None): Rues, exigées par les `street_*`.
        nb_q_by_temp (int): `nb_q` transmis à chaque générateur spatial.
        nb_q_by_feat (int): Budget de groupes sémantiques par question spatiale.
        list_template (list[str]): Clés de `REGISTRY` à exécuter.
        ratio (dict[int, float] | None): Poids par nombre d'attributs combinés.
        feature_cols (list[str]): Colonnes d'attributs éligibles.
        seed (int): Graine du tirage sémantique.

    Returns:
        DataFrame: Les questions composites, avec la colonne `query_geo_semantic`
            portant l'énoncé enrichi et `query` conservant l'énoncé spatial seul.
    """
    sources = {"df_osm": df_osm, "df_area": df_area, "df_streets": df_streets}
    df_geo_q = pd.concat(
        [REGISTRY[name](nb_q=nb_q_by_temp,
                        **{k: v for k, v in sources.items()
                           if k in inspect.signature(REGISTRY[name]).parameters})
         for name in list_template],
        ignore_index=True)
    df_geo_sem = question_geo_semantic(df_geo_q, df_osm, nb_q_by_feat, feature_cols, ratio, seed, dropna)
    query_row = []
    for _, row in df_geo_sem.iterrows():
        mots = row["query"].split(" ")               
        qualif = " ".join(f"{f}={v}" for f, v in zip(row["features"], row["values"]))
        mots.insert(1, qualif)                         
        query_row.append(" ".join(mots))
    df_geo_sem["query_geo_semantic"] = pd.DataFrame(query_row)
    return df_geo_sem
