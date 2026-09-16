"""Questions sémantiques : décrire un POI par ses attributs, sans géographie.

Là où `geospatial` interroge une relation spatiale — « le café le plus proche de
X » — ce module interroge des **attributs** : « un restaurant italien avec
terrasse ». Aucune coordonnée n'intervient ; le corpus est lu comme une table
d'attributs, et la difficulté vient du nombre de contraintes combinées.

Les attributs ne sont pas des colonnes brutes mais des objets `Feature`
(cf. `src/features.py`), qui savent deux choses : lire la valeur d'un POI
(`check`) et la traduire en fragment de phrase (`to_text`). Cette indirection
permet aux features dérivées — « à moins de 200 m de la Seine » — d'exister à
côté des tags OSM bruts, et laisse `to_text` rendre `None` quand l'attribut
n'est pas renseigné, ce qui écarte le POI au lieu de produire un énoncé bancal.
"""
from pandas import DataFrame
import numpy as np
from itertools import combinations

from collections import defaultdict
from src.config import *
from src.template_question.ratio import allocate


def generate_question(poi, n_features, features):
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

    TODO: la fonction rend **toutes** les combinaisons, mais son seul appelant
        n'en garde que la première (`question[0]`). Le reste du travail est
        jeté. Mesuré sur un restaurant à trois attributs renseignés :
        3 combinaisons calculées pour `n_features=1`, 3 pour `n_features=2`,
        une seule retenue à chaque fois.
    TODO: `available` est construit dans l'ordre de `features`, donc la première
        combinaison est toujours la même pour un POI donné. Retirer deux fois le
        même POI produit deux fois la question identique — d'où les doublons
        mesurés (4 sur 60). Si une seule combinaison doit être retenue, autant
        la tirer au sort ici, avec le `rng` de l'appelant.
    """
    # Calcule tous les fragments disponibles pour ce POI
    available = []
    for feature in features:
        value = feature.check(poi)
        text = feature.to_text(value)
        if text is not None and text != "nan":
            if isinstance(text, list):
                available.extend(text)  # aplatit les features multi-valeurs
            else:
                available.append(text)

    # Pas assez de features disponibles pour ce POI
    if len(available) < n_features:
        return None
    return [[poi["category"]] + sorted(combo) for combo in combinations(available, n_features)]


def make_questions_semantic(df_osm, features, nb_question=100, ratio=None, rng=None, seed=42):
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
        nb_question (int): Nombre total de questions visé.
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

    TODO: le paramètre `features` n'est pas utilisé pour calculer les strates —
        la boucle lit le **global** `FEATURES` importé par `from src.config
        import *`. Passer une liste de features différente de `FEATURES` donne
        donc des strates incohérentes, et **boucle indéfiniment** dès qu'une
        strate est inatteignable : avec une seule feature passée, la strate
        `nb_f=2` n'est jamais satisfaite et `while n < nb_q` ne termine pas.
        Vérifié en exécutant.
    TODO: `while n < nb_q` n'a aucun plafond de tentatives. Même avec
        `features=FEATURES`, un corpus dont aucun POI ne porte assez d'attributs
        renseignés fige l'appel. Les templates `geospatial` récents portent un
        `max_try` pour cette raison.
    TODO: `ratio` porte mal son nom : quand il est fourni, il est consommé tel
        quel comme un nombre de questions par strate, sans passer par
        `allocate` ; quand il est absent, il est calculé par `allocate`. Les
        deux chemins ne reçoivent donc pas la même chose — un poids dans un cas,
        un compte dans l'autre. Ailleurs dans le projet, `ratio` désigne
        toujours des poids donnés à `allocate`.
    TODO: `r[k] = 1/len(FEATURES)` — la division est inutile, `allocate`
        normalise déjà les poids bruts. `r[k] = 1` suffit, et évite le piège de
        la division entière qui a vidé `point_between` en son temps.
    TODO: rien n'empêche de retirer le même POI dans la même strate. Mesuré :
        4 doublons (même `poi`, même `nb_feature`) sur 60 questions, et comme
        `generate_question` rend toujours la même combinaison, ces doublons
        portent un énoncé identique.
    """
    if not ratio:
        # Strates : de 1 attribut combiné jusqu'à len(FEATURES).
        r = defaultdict(int)
        for k in range(1, len(FEATURES)+1):
            r[k] = 1/len(FEATURES)
        ratio = allocate(nb_question, r)
    if not rng:
        rng = np.random.default_rng(seed)
    category = df_osm["category"].unique()
    dic_question = defaultdict(list)
    for nb_f, nb_q in ratio.items():
        n=0
        # Tirage par rejet : `n` n'avance que sur un POI assez documenté.
        while n<nb_q:
            cat_q = rng.choice(category)
            pois = df_osm[df_osm["category"]==cat_q]
            
            id_poi = rng.choice(pois["poi_id"])
            question = generate_question(pois[pois["poi_id"]==id_poi].iloc[0], nb_f, features)
            if question:
                # `question[0]` : seule la première des combinaisons est retenue.
                dic_question["question"].append(question[0])
                dic_question["poi"].append(id_poi)
                dic_question["nb_feature"].append(nb_f)
                n += 1
    return DataFrame(dic_question)
