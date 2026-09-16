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

TODO: le nom `make_question_geo` date de l'époque où la fonction ne faisait que
    la partie spatiale. Elle produit maintenant des questions composites, et le
    nom devrait le dire.
TODO: `from ... geospatial import *` n'est plus utilisé — les générateurs sont
    atteints par `REGISTRY`, pas par leur nom de module. L'étoile masque
    l'origine réelle des symboles sans rien apporter.
"""

from src.template_question.geospatial import *
from collections import defaultdict
from itertools import combinations
import pandas as pd
import inspect

from src.template_question import geospatial
from src.template_question.registry import REGISTRY
from src.template_question.ratio import allocate


#: Les douze générateurs spatiaux, désignés par leur clé dans `REGISTRY` — donc
#: par leur nom de fonction, et non par le nom court du registre de `schema.py`.
#: Les deux registres cohabitent : celui-ci est peuplé par le décorateur
#: `@template` à l'import de `geospatial`.
list_template = ['make_question_area_border', 'make_question_area_direction',
                 'make_question_area_inside', 'make_question_area_outside',
                 'make_question_point_between', 'make_question_point_near_cardinal',
                 'make_question_point_near_metric', 'make_question_point_near',
                 'make_question_point_towards', 'make_question_street_along',
                 'make_question_street_cross', 'make_question_street_opposite_side']

def feature_agregation(df, nb_feature, nb_question, features_cols, dropna=True, seed=42):
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
    ids = df["poi_id"]
    rows = []
    for combo in combinations(features_cols, nb_feature):
        df_full = df.dropna(subset=list(combo))
        g = df_full.groupby(list(combo), dropna=dropna, observed=True)
        for values, idx in g.groups.items():
            # `groups` rend une valeur nue pour une clé simple, un tuple sinon :
            # on normalise pour que `zip(combo, values)` marche dans les deux cas.
            values = values if isinstance(values, tuple) else (values,)
            rows.append({
                "features": combo,
                "values": values,
                "number features": nb_feature,
                "results_features_pois_id": list(ids.loc[idx]),
                "size": len(idx),
                **dict(zip(combo, values)),
            })

    out = pd.DataFrame(rows)
    return (out.sample(n=min(nb_question, len(out)), random_state=seed))

def question_geo_semantic(df_question, df_osm, nb_q, features_cols, ratio=None, seed=42):
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
    if ratio is None:
        ratio = {i: 1 for i in range(1, len(features_cols) + 1)}
    balance = allocate(nb_q, ratio)
    df_out = []
    for _, question in df_question.iterrows():
        # Restreindre au corpus de la réponse : la contrainte sémantique doit
        # porter sur des POIs que la question spatiale retient déjà.
        id_poi = question["results_poi_id"]
        sub_df = df_osm[df_osm['poi_id'].isin(set(id_poi))]
        for nb_f, nb_row in balance.items():
            answers_poi = feature_agregation(sub_df, nb_f, nb_row, features_cols, seed=seed)
            # `how="cross"` duplique la ligne de question autant de fois qu'il y
            # a de groupes : une question composite par couple.
            res = question.to_frame().T.merge(answers_poi, how="cross")
            df_out.append(res)
    return pd.concat(df_out, ignore_index=True)


def make_question_geo(
        df_osm, df_area=None, df_streets=None, nb_q_by_temp=100, nb_q_by_feat=10, 
        list_template=list_template, ratio=None, 
        feature_cols=["outdoor_seating", "indoor_seating", "cuisine"],
        seed=42
    ):
    """Produit le jeu complet de questions composites.

    Enchaîne les trois étapes : générer les questions spatiales, les croiser avec
    les groupes d'attributs, puis réécrire l'énoncé pour que la contrainte
    sémantique y apparaisse.

    Les générateurs n'ont pas tous la même signature — quatre veulent `df_area`,
    trois `df_streets`, les autres `df_osm` seul. `inspect.signature` filtre donc
    `sources` pour ne passer à chacun que ce qu'il déclare, ce qui permet de les
    appeler en boucle sans les connaître.

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

    TODO: un générateur qui rend autre chose qu'un DataFrame casse le `pd.concat`
        pour le lot entier, avec un message qui ne le nomme pas
        (`cannot concatenate object of type '<class 'str'>'`). C'est le cas de
        `make_question_point_near_cardinal` sous son seuil, qui rend un message
        d'erreur au lieu de le lever. Vérifier le type en sortie de boucle, ou
        faire lever les générateurs.
    TODO: `df_area=None` et `df_streets=None` par défaut, alors que sept des
        douze templates les exigent : l'appel `make_question_geo(df_osm)` lève un
        `AttributeError: 'NoneType' object has no attribute 'geometry'` depuis
        l'intérieur d'un template. Écarter les templates dont les sources
        manquent, ou exiger les arguments.
    TODO: `seed=42` est écrit en dur dans l'appel à `question_geo_semantic`, ce
        qui ignore le paramètre `seed` de cette fonction.
    TODO: `pd.DataFrame(query_row)` fabrique une frame à une colonne que pandas
        doit réaligner sur l'index. `df_geo_sem["query_geo_semantic"] = query_row`
        écrit la liste directement, sans alignement — équivalent ici parce que
        `ignore_index=True` a donné un `RangeIndex`, mais robuste dans le cas
        contraire.
    """
    sources = {"df_osm": df_osm, "df_area": df_area, "df_streets": df_streets}
    # Chaque générateur ne reçoit que les sources que sa signature déclare.
    df_geo_q = pd.concat(
        [REGISTRY[name](nb_q=nb_q_by_temp,
                        **{k: v for k, v in sources.items()
                           if k in inspect.signature(REGISTRY[name]).parameters})
         for name in list_template],
        ignore_index=True)
    df_geo_sem = question_geo_semantic(df_geo_q, df_osm, nb_q_by_feat, feature_cols, ratio, seed=42)
    # Réécriture de l'énoncé : le qualificatif sémantique se glisse juste après
    # la catégorie cherchée, qui est toujours le premier mot.
    #   « cafe near poi5 »  ->  « cafe cuisine=italian near poi5 »
    query_row = []
    for _, row in df_geo_sem.iterrows():
        mots = row["query"].split(" ")               
        # `features` porte les noms, `values` les valeurs : il faut les deux,
        # « italian » seul ne dirait pas de quelle colonne il est la valeur.
        qualif = " ".join(f"{f}={v}" for f, v in zip(row["features"], row["values"]))
        # `insert` mute la liste et rend None : on insère, puis on relit `mots`.
        mots.insert(1, qualif)                         
        query_row.append(" ".join(mots))
    df_geo_sem["query_geo_semantic"] = pd.DataFrame(query_row)
    return df_geo_sem
