from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypedDict

import numpy as np
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
    ) -> DataFrame:
    """Une ligne par question : texte, POI pertinents, négatifs difficiles.

    Args:
        df_question: questions, colonnes `question`, `pois`, `features`.
        df_osm: corpus de POI.
        positions: positions des questions à traiter.
        rng: générateur numpy.
        n_hard: négatifs difficiles tirés par question.
        nb_unmatches: transmis à `hard_negatives`.

    Returns:
        DataFrame avec les colonnes `question`, `pois`, `hard_pois`.
    """
    rows = []
    for pos in positions:
        # `dict.fromkeys` et non `set` : les deux dédoublonnent, mais un `set`
        # rend ses éléments dans l'ordre de ses cases de hachage, ce qui détruit
        # l'ordre de pertinence des POIs. Cf. `composite/format_samples.py`.
        pois = list(dict.fromkeys(df_question["pois"].iloc[pos]))
        if is_test:
            rows.append({
                "index": pos,
                "question": df_question["question"].iloc[pos],
                "pois": pois,
            })
        else:
            candidats = hard_negatives(pos, df_question, df_osm, nb_unmatches)
            hard_pois = [p for p in tirer(rng, candidats, n_hard) if p not in pois]
            rows.append({
                "index": pos,
                "question": df_question["question"].iloc[pos],
                "pois": pois,
                "hard_pois": hard_pois,
            })
    return DataFrame(rows)


def make_benchmark_question(
        df_question: DataFrame, 
        df_osm: DataFrame, 
        rng: Generator,
        ratio_test: float = 0.1,
        ratio_features: dict[int, float] | None = None,
        ratio_category: dict[str, float] | None = None,
        nb_hard: int = 20,
        nb_unmatches: int = 1,
    ) -> tuple[list[Batch], NDArray]:
    """Split des questions, stratification du train, puis construction des batchs.

    Args:
        df_question: questions.
        df_osm: corpus de POI.
        rng: générateur numpy.
        ratio_test: part réservée au test.
        ratio_features: proportions par nombre de contraintes, par exemple
            {1: 0.4, 2: 0.4, 3: 0.2}. `None` garde toutes les questions.
        ratio_category: proportions par catégorie, par exemple
            {"restaurant": 0.25, "cafe": 0.25, "bar": 0.25, "hotel": 0.25} pour
            corriger un corpus déséquilibré. `None` garde toutes les questions.
            Dans les deux cas, une valeur absente du dict est écartée.
        batch_size: questions par batch.
        n_times: variantes par batch, passé à `augment_benchmark`.
        nb_answers: réponses conservées par question à l'augmentation.
        nb_hard: négatifs difficiles par question.
        nb_unmatches: transmis à `hard_negatives`.

    Returns:
        Les batchs d'entraînement, et les positions de test.
    """
    train, test = split_train_test(len(df_question), ratio_test, rng)
    train_set = make_rows(df_question, df_osm, train, rng, nb_hard, nb_unmatches)
    test_set = make_rows(df_question, df_osm, test, rng, nb_hard, nb_unmatches, is_test=True)
    return train_set, test_set
