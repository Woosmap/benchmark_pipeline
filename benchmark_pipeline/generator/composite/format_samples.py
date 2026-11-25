from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypedDict

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from pandas import DataFrame, Series, factorize
from benchmark_pipeline.config import (
    ANSWERS_COLUMN, NB_HARD_QUESTION, NB_HARD_UNMATCHED, QUESTION_COLUMN, ROWS_TARGET, SPATIAL_ANSWERS_COLUMN,
)

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

def split_spatial_questions(
        df_geo: DataFrame,
        ratio_test: float,
        rng: Generator
    ) -> tuple[DataFrame, DataFrame]:
    """Coupe les questions spatiales en train et test, avant l'ajout des attributs.

    Les questions de même `query` (même énoncé spatial) vont du même côté : les énoncés sont mélangés,
    puis mis en test jusqu'à atteindre `ratio_test` des questions.

    Args:
        df_geo: questions spatiales, colonne `query`.
        ratio_test: part réservée au test.
        rng: générateur numpy.

    Returns:
        (train, test), deux DataFrames réindexés à partir de 0.
    """
    groupes, _ = factorize(df_geo["query"])
    ordre = rng.permutation(groupes.max() + 1)
    n_test = int(len(df_geo) * ratio_test)
    nb_groupes_test = np.searchsorted(np.cumsum(np.bincount(groupes)[ordre]), n_test) + 1 if n_test else 0
    est_test = np.isin(groupes, ordre[:nb_groupes_test])
    return df_geo[~est_test].reset_index(drop=True), df_geo[est_test].reset_index(drop=True)

def make_rows(
        df_question: DataFrame,
        df_osm: DataFrame,
        positions: NDArray,
        rng: Generator,
        n_hard: int,
        nb_unmatches: int,
        is_test: bool = False,
        rows_target: Sequence[str] = (),
        colonne_question: str = "query_texte",
        colonne_pois: str = "results_features_pois_id",
        colonne_pois_spatiaux: str = "results_poi_id",
    ) -> DataFrame:
    """Une ligne par question : texte, POI pertinents, cibles, négatifs difficiles.

    Args:
        df_question: questions composites, colonnes `features`, `values`, plus
            celles désignées par `colonne_question`, `colonne_pois` et
            `rows_target`.
        df_osm: corpus de POI.
        positions: positions des questions à traiter.
        rng: générateur numpy.
        n_hard: négatifs difficiles tirés par question.
        nb_unmatches: nombre de contraintes qu'un négatif difficile peut rater.
        is_test: sans négatifs difficiles.
        rows_target: colonnes de `df_question` recopiées telles quelles dans
            chaque ligne, par exemple `("ellypse",)`.
        colonne_question: colonne portant l'énoncé. `query_geo_semantic` est
            l'énoncé enrichi de la contrainte d'attribut, `query` le spatial seul.
        colonne_pois: colonne portant les réponses. `results_features_pois_id` est
            l'intersection relation spatiale ∩ attributs, `results_poi_id` la
            réponse spatiale seule.
        colonne_pois_spatiaux: colonne où puiser les négatifs difficiles — les POIs
            qui satisfont la relation spatiale mais pas tous les attributs.

    Returns:
        DataFrame avec `index`, `query_texte`, `pois`, les `rows_target`, et
            `hard_pois` hors test.
    """
    rows = []
    for pos in positions:
        ligne = df_question.iloc[pos]
        pois = list(dict.fromkeys(ligne[colonne_pois]))
        row = {
            "index": pos,
            "query": ligne[colonne_question],
            "answers": pois,
            **{colonne: ligne[colonne] for colonne in rows_target},
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
        is_test: bool = False,
        rows_target: list[str] = ROWS_TARGET,
        nb_hard: int = NB_HARD_QUESTION,
        nb_unmatches: int = NB_HARD_UNMATCHED,
        colonne_question: str = QUESTION_COLUMN,
        colonne_pois: str = ANSWERS_COLUMN,
        colonne_pois_spatiaux: str = SPATIAL_ANSWERS_COLUMN,
    ) -> DataFrame:
    """Met en forme les questions d'une branche (train ou test), dans un ordre tiré au hasard.

    `index` de chaque ligne est la position de la question dans `df_question` ; les lignes du test
    n'ont pas de négatifs difficiles.

    colonne_question, colonne_pois, colonne_pois_spatiaux : passed to `make_rows`.
    """
    return make_rows(df_question, df_osm, rng.permutation(len(df_question)), rng, nb_hard, nb_unmatches,
                     is_test=is_test, rows_target=rows_target, colonne_question=colonne_question,
                     colonne_pois=colonne_pois, colonne_pois_spatiaux=colonne_pois_spatiaux)
