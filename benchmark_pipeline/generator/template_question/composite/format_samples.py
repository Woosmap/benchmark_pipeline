from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypedDict

from numpy.random import Generator
from numpy.typing import NDArray
from pandas import DataFrame, Series

PoiId = int | tuple

class Batch(TypedDict):
    """Un batch d'entraînement contrastif."""

    questions: list[list[str]]
    poi_ids: list[PoiId] 


def count_matches(df_osm: DataFrame, constraints: dict[str, Any]) -> Series:
    """Pour chaque POI, le nombre de contraintes qu'il satisfait.

    Args:
        df_osm: corpus de POI.
        constraints: colonne -> valeur attendue.

    Returns:
        Série d'entiers, indexée comme `df_osm`.
    """
    matches = [df_osm[column] == value for column, value in constraints.items()]
    return sum(matches)

def hard_negatives(
        pos: int, 
        df_question: DataFrame, 
        df_osm: DataFrame,
        nb_unmatches: int = 1
    ) -> NDArray:
    """POI qui satisfont une partie des contraintes de la question, mais pas toutes.

    Args:
        pos: position de la question dans `df_question`.
        df_question: questions, colonne `features`.
        df_osm: corpus de POI.
        nb_unmatches: nombre de contraintes non satisfaites.

    Returns:
        Les `poi_id` candidats.
    """
    constraints = df_question.iloc[pos]["features"]
    nb_matched = count_matches(df_osm, constraints)
    nb_asked = len(constraints)
    return df_osm["poi_id"][(nb_matched >= (nb_asked - nb_unmatches)) & (nb_matched < nb_asked)].to_numpy()

def tirer(rng: Generator, candidats: Sequence[PoiId], k: int) -> list[PoiId]:
    """Tire au plus k candidats distincts, sans planter si le pool est petit.

    Args:
        rng: générateur numpy.
        candidats: pool, indexable par position.
        k: nombre voulu.

    Returns:
        Les valeurs tirées, `[]` si le pool est vide.
    """
    if len(candidats) == 0:
        return []
    positions = rng.choice(len(candidats), size=min(k, len(candidats)), replace=False)
    return [candidats[i] for i in positions]

def split_train_test(
        n_questions: int, 
        ratio_test: float,
        rng: Generator
    ) -> tuple[NDArray, NDArray]:
    """Mélange les positions 0..n-1, puis coupe en test et train.

    Args:
        n_questions: nombre de questions.
        ratio_test: part réservée au test.
        rng: générateur numpy.

    Returns:
        (train, test), des positions.
    """
    positions = rng.permutation(n_questions)
    n_test = int(n_questions * ratio_test)
    return positions[n_test:], positions[:n_test]

def make_rows(
        df_question: DataFrame,
        df_osm: DataFrame,
        positions: NDArray,
        rng: Generator,
        n_hard: int,
        nb_unmatches: int,
        is_test: bool = False,
        colonnes_cibles: Sequence[str] = (),
        colonne_question: str = "query_geo_semantic",
        colonne_pois: str = "results_features_pois_id",
        colonne_pois_spatiaux: str = "results_poi_id",
    ) -> DataFrame:
    """Une ligne par question : texte, POI pertinents, cibles, négatifs difficiles.

    Args:
        df_question: questions composites, colonnes `features`, `values`, plus
            celles désignées par `colonne_question`, `colonne_pois` et
            `colonnes_cibles`.
        df_osm: corpus de POI.
        positions: positions des questions à traiter.
        rng: générateur numpy.
        n_hard: négatifs difficiles tirés par question.
        nb_unmatches: nombre de contraintes qu'un négatif difficile peut rater.
        is_test: sans négatifs difficiles.
        colonnes_cibles: colonnes de `df_question` recopiées telles quelles dans
            chaque ligne, par exemple `("ellypse",)`.
        colonne_question: colonne portant l'énoncé. `query_geo_semantic` est
            l'énoncé enrichi de la contrainte d'attribut, `query` le spatial seul.
        colonne_pois: colonne portant les réponses. `results_features_pois_id` est
            l'intersection relation spatiale ∩ attributs, `results_poi_id` la
            réponse spatiale seule.
        colonne_pois_spatiaux: colonne où puiser les négatifs difficiles — les POIs
            qui satisfont la relation spatiale mais pas tous les attributs.

    Returns:
        DataFrame avec `index`, `question`, `pois`, les `colonnes_cibles`, et
            `hard_pois` hors test.
    """
    rows = []
    for pos in positions:
        ligne = df_question.iloc[pos]
        # `dict.fromkeys` et non `set` : les deux dédoublonnent, mais un `set`
        # rend ses éléments dans l'ordre de ses cases de hachage, ce qui détruit
        # l'ordre de pertinence spatiale que le générateur a pris soin de poser.
        pois = list(dict.fromkeys(ligne[colonne_pois]))
        row = {
            "index": pos,
            "question": ligne[colonne_question],
            "pois": pois,
            **{colonne: ligne[colonne] for colonne in colonnes_cibles},
        }
        if not is_test:
            contraintes = dict(zip(ligne["features"], ligne["values"]))
            spatiaux = df_osm[df_osm["poi_id"].isin(set(ligne[colonne_pois_spatiaux]))]
            nb_matched = count_matches(spatiaux, contraintes)
            nb_asked = len(contraintes)
            candidats = spatiaux["poi_id"][(nb_matched >= nb_asked - nb_unmatches)
                                           & (nb_matched < nb_asked)].to_numpy()
            row["hard_pois"] = [p for p in tirer(rng, candidats, n_hard) if p not in pois]
        rows.append(row)
    return DataFrame(rows)


def format_composit_samples(
        df_question: DataFrame, 
        df_osm: DataFrame, 
        rng: Generator,
        ratio_test: float = 0.1,
        ratio_features: dict[int, float] | None = None,
        ratio_category: dict[str, float] | None = None,
        nb_hard: int = 20,
        nb_unmatches: int = 1,
    ) -> tuple[list[Batch], NDArray]:

    train, test = split_train_test(len(df_question), ratio_test, rng)
    train_set = make_rows(df_question, df_osm, train, rng, nb_hard, nb_unmatches)
    test_set = make_rows(df_question, df_osm, test, rng, nb_hard, nb_unmatches, is_test=True)
    return train_set, test_set
