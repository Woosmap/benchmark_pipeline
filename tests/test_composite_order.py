"""Conservation de l'ordre de pertinence spatiale dans les questions composites.

`question_geo_semantic` filtre la réponse spatiale d'une question par une
contrainte d'attributs. Le filtre doit **préserver l'ordre** :
`results_features_pois_id` est une *sous-suite* de `results_poi_id`, pas
seulement un sous-ensemble. C'est ce que garantissent le `reindex` et le
`groupby(sort=False)` de `generator_qcomposite.py` — deux choix silencieux, que
rien ne signale s'ils sont perdus au cours d'un refactor.

L'ordre doit survivre jusqu'au dataset écrit : les derniers tests couvrent donc
aussi les `make_rows` des deux registres, où un `list(set(...))` le détruisait.

Ces tests n'appellent aucun template : ils donnent directement aux fonctions un
`df_question` fabriqué ici. Pas de DuckDB, pas de réseau, et une vérité terrain
recalculable à la main.
"""

import numpy as np
import pandas as pd
import pytest

from benchmark_pipeline.generator.template_question.composite.format_samples import (
    make_rows as make_rows_composite,
)
from benchmark_pipeline.generator.template_question.composite.generator_qcomposite import (
    question_geo_semantic,
)
from benchmark_pipeline.generator.template_question.semantic.batcher import (
    make_rows as make_rows_semantique,
)

SEED = 42

FEATURE_COLS = ["category", "cooking_type"]

#: Ordre de pertinence spatiale d'une réponse, délibérément décroisé de l'ordre
#: des `poi_id`. Sans ce décroisement, un tri accidentel — `groupby` repassé en
#: `sort=True`, un `set()`, un `sorted()` — rendrait la même suite et les tests
#: passeraient sur un générateur cassé.
ORDRE_SPATIAL = [7, 3, 9, 1, 5, 2]

#: Attributs de chaque POI, dans l'ordre de `FEATURE_COLS`. Les catégories
#: alternent le long de `ORDRE_SPATIAL` : chaque groupe est donc entrelacé avec
#: les autres, et un filtre qui réordonne se voit.
ATTRIBUTS = {
    7: ("cafe", "italien"),
    3: ("restaurant", "italien"),
    9: ("cafe", "japonais"),
    1: ("restaurant", "italien"),
    5: ("cafe", "italien"),
    2: ("restaurant", "japonais"),
}


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def df_osm_attributs():
    """Corpus minimal : `poi_id` et les colonnes d'attributs.

    Returns:
        DataFrame: un POI par ligne, trié par `poi_id`.
    """
    # Trié par `poi_id`, donc dans un ordre différent d'ORDRE_SPATIAL : c'est au
    # `reindex` du générateur de réimposer l'ordre spatial. Un corpus déjà dans
    # le bon ordre masquerait sa disparition.
    return pd.DataFrame(
        [{"poi_id": poi_id, "category": category, "cooking_type": cuisine}
         for poi_id, (category, cuisine) in sorted(ATTRIBUTS.items())]
    )


@pytest.fixture
def df_question():
    """Une question spatiale dont la réponse suit `ORDRE_SPATIAL`.

    Returns:
        DataFrame: une ligne, colonnes `query` et `results_poi_id`.
    """
    return pd.DataFrame([{
        "query": "pois near poi0",
        "results_poi_id": list(ORDRE_SPATIAL),
    }])


# --------------------------------------------------------------------------- #
# helpers d'oracle
# --------------------------------------------------------------------------- #

def est_sous_suite(candidate, reference):
    """Dit si `candidate` apparaît dans `reference`, dans le même ordre.

    Args:
        candidate (list): Suite à chercher.
        reference (list): Suite de référence.

    Returns:
        bool: True si `candidate` est une sous-suite de `reference`.
    """
    # `in` sur un itérateur le consomme au fur et à mesure : deux occurrences du
    # même POI exigent donc deux occurrences dans la référence, là où une
    # comparaison d'ensembles les confondrait.
    reste = iter(reference)
    return all(poi in reste for poi in candidate)


def reponse_attendue(features, values):
    """Réponse exacte que doit produire une contrainte d'attributs.

    Args:
        features (tuple[str]): Colonnes contraintes.
        values (tuple): Valeurs imposées, parallèles à `features`.

    Returns:
        list[int]: POIs de `ORDRE_SPATIAL` satisfaisant la contrainte, dans
            l'ordre spatial.
    """
    return [
        poi for poi in ORDRE_SPATIAL
        if all(ATTRIBUTS[poi][FEATURE_COLS.index(colonne)] == valeur
               for colonne, valeur in zip(features, values))
    ]


def generer(df_question, df_osm, part_avec_features=1.0):
    """Lance le générateur composite sur le monde de ce module.

    `nb_q_by_feat=20` dépasse le nombre de contraintes disponibles — 4 à un
    attribut, 4 à deux — donc le tirage les retient **toutes** : le test couvre
    chaque groupe plutôt qu'un échantillon dépendant de la graine.

    Args:
        df_question (DataFrame): Questions spatiales.
        df_osm (DataFrame): Corpus avec ses attributs.
        part_avec_features (float): Part des questions à enrichir.

    Returns:
        DataFrame: Les questions composites.
    """
    return question_geo_semantic(
        df_question, df_osm, FEATURE_COLS,
        ratio_features={1: 1, 2: 1},
        nb_q_by_feat=20,
        part_avec_features=part_avec_features,
        seed=SEED,
    )


# --------------------------------------------------------------------------- #
# contrôle de l'oracle
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("candidate, attendu", [
    ([7, 9, 5], True),                  # groupe « cafe », dans l'ordre spatial
    ([5, 9, 7], False),                 # mêmes POIs, ordre inversé
    ([7, 5, 9], False),                 # une seule transposition
    ([], True),
    (list(ORDRE_SPATIAL), True),
    ([7, 3, 4], False),                 # un POI hors de la réponse spatiale
    ([7, 7], False),                    # doublon sans doublon en face
])
def test_est_sous_suite_detecte_le_desordre(candidate, attendu):
    """Sans ce contrôle, un `est_sous_suite` trop permissif validerait tout."""
    assert est_sous_suite(candidate, ORDRE_SPATIAL) is attendu


# --------------------------------------------------------------------------- #
# invariant d'ordre
# --------------------------------------------------------------------------- #

def test_reponse_filtree_est_une_sous_suite(df_question, df_osm_attributs):
    """Toute réponse composite suit l'ordre de la réponse spatiale dont elle sort."""
    df_composite = generer(df_question, df_osm_attributs)

    assert not df_composite.empty, "aucune contrainte tirée : le test ne prouve rien"
    for _, ligne in df_composite.iterrows():
        pois = list(ligne["results_features_pois_id"])
        assert est_sous_suite(pois, ligne["results_poi_id"]), (
            f"{dict(zip(ligne['features'], ligne['values']))} : "
            f"{pois} n'est pas une sous-suite de {ligne['results_poi_id']}"
        )


def test_reponse_filtree_est_le_groupe_complet(df_question, df_osm_attributs):
    """La réponse est *exactement* le groupe, pas une sous-suite quelconque.

    Vérification complémentaire de la précédente : une liste vide est une
    sous-suite valide de n'importe quoi, donc l'invariant d'ordre seul ne dit
    rien sur les POIs oubliés.
    """
    df_composite = generer(df_question, df_osm_attributs)

    for _, ligne in df_composite.iterrows():
        attendu = reponse_attendue(ligne["features"], ligne["values"])
        assert list(ligne["results_features_pois_id"]) == attendu, (
            f"{dict(zip(ligne['features'], ligne['values']))} : "
            f"attendu {attendu}"
        )


def test_toutes_les_contraintes_sont_couvertes(df_question, df_osm_attributs):
    """Les deux tests d'ordre voient bien les 8 groupes du monde de test.

    Si le générateur n'en produisait que les singletons, l'ordre ne serait
    contraint nulle part et les assertions passeraient à vide.
    """
    df_composite = generer(df_question, df_osm_attributs)

    groupes = {tuple(ligne["results_features_pois_id"])
               for _, ligne in df_composite.iterrows()}
    assert sum(len(groupe) > 1 for groupe in groupes) >= 4


def test_sans_contrainte_la_reponse_spatiale_est_intacte(df_question, df_osm_attributs):
    """Une question non enrichie recopie sa réponse spatiale, ordre compris."""
    df_composite = generer(df_question, df_osm_attributs, part_avec_features=0.0)

    assert len(df_composite) == 1
    ligne = df_composite.iloc[0]
    assert list(ligne["results_features_pois_id"]) == ORDRE_SPATIAL
    assert ligne["features"] == () and ligne["values"] == ()


# --------------------------------------------------------------------------- #
# écriture du dataset
# --------------------------------------------------------------------------- #

#: Réponse à doublons, telle qu'en produit `point_near_metric`. Dédoublonnée en
#: conservant la première occurrence, elle vaut `[7, 3, 9]` ; un `set` rendait
#: `[9, 3, 7]`, qui n'est ni l'ordre spatial ni un tri.
REPONSE_AVEC_DOUBLONS = [7, 3, 7, 9, 3]


def test_make_rows_composite_conserve_l_ordre(df_question, df_osm_attributs):
    """La mise en forme du dataset recopie les réponses sans les réordonner."""
    df_composite = generer(df_question, df_osm_attributs)

    lignes = make_rows_composite(
        df_composite, df_osm_attributs, np.arange(len(df_composite)),
        np.random.default_rng(SEED), n_hard=2, nb_unmatches=1,
    )

    for (_, composite), (_, ligne) in zip(df_composite.iterrows(), lignes.iterrows()):
        assert list(ligne["pois"]) == list(composite["results_features_pois_id"]), (
            f"{dict(zip(composite['features'], composite['values']))} : "
            f"ordre perdu à l'écriture"
        )


def test_make_rows_composite_dedoublonne_sans_reordonner(df_osm_attributs):
    """Le dédoublonnage reste assuré, sans l'effet de bord du `set`."""
    df_composite = pd.DataFrame([{
        "query_geo_semantic": "pois cooking_type=italien near poi0",
        "results_poi_id": list(ORDRE_SPATIAL),
        "results_features_pois_id": list(REPONSE_AVEC_DOUBLONS),
        "features": ("cooking_type",),
        "values": ("italien",),
    }])

    lignes = make_rows_composite(
        df_composite, df_osm_attributs, np.array([0]),
        np.random.default_rng(SEED), n_hard=2, nb_unmatches=1,
    )

    assert list(lignes.loc[0, "pois"]) == [7, 3, 9]


def test_make_rows_semantique_dedoublonne_sans_reordonner(df_osm_attributs):
    """Le registre sémantique portait le même défaut, à la même ligne."""
    df_question_semantique = pd.DataFrame([{
        "question": "pois cooking_type=italien near poi0",
        "pois": list(REPONSE_AVEC_DOUBLONS),
        "features": {"cooking_type": "italien"},
    }])

    lignes = make_rows_semantique(
        df_question_semantique, df_osm_attributs, np.array([0]),
        np.random.default_rng(SEED), n_hard=2, nb_unmatches=1,
    )

    assert list(lignes.loc[0, "pois"]) == [7, 3, 9]
