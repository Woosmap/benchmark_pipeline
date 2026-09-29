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

def stratify(
        df_question: DataFrame, 
        positions: NDArray,
        ratio_features: dict[int, float], 
        rng: Generator
    ) -> NDArray:
    """Sous-ensemble de questions respectant les proportions par nombre de contraintes.

    Args:
        df_question: questions, colonne `nb_feature`.
        positions: positions candidates.
        ratio_features: nb_feature -> part du total, par exemple {1: 0.4, 2: 0.4, 3: 0.2}.
        rng: générateur numpy.

    Returns:
        Les positions retenues, mélangées.
    """
    nb_feature = df_question["nb_feature"].to_numpy()
    pools = {nb: positions[nb_feature[positions] == nb] for nb in ratio_features}

    # Combien de questions au total peut-on tirer sans casser les proportions ?
    # C'est la strate la plus pauvre qui décide.
    n_total = min([len(pools[nb]) / ratio
                   for nb, ratio in ratio_features.items() if ratio > 0], default=0)

    selected = []
    for nb, ratio in ratio_features.items():
        taille = int(n_total * ratio)
        selected += list(rng.choice(pools[nb], size=taille, replace=False))

    selected = np.array(selected, dtype=int)
    rng.shuffle(selected)   # sinon les batchs sortent triés par difficulté
    return selected

def make_batches(
        df_question: DataFrame, 
        df_osm: DataFrame, 
        positions: NDArray,
        rng: Generator, 
        batch_size: int = 32, 
        n_hard: int = 2,
        nb_unmatches: int = 1
    ) -> list[Batch]:
    """Découpe les positions en batchs : questions, pool de POI, masque de pertinence.

    Args:
        df_question: questions, colonnes `question`, `pois`, `features`.
        df_osm: corpus de POI.
        positions: positions des questions à batcher.
        rng: générateur numpy.
        batch_size: questions par batch.
        n_hard: négatifs difficiles ajoutés par question.
        nb_unmatches: transmis à `hard_negatives`.

    Returns:
        Un batch par tranche de `batch_size` positions.
    """
    batches = []

    for start in range(0, len(positions), batch_size):
        chunk = positions[start:start + batch_size]

        # Le pool est commun : les POI pertinents d'une question servent de
        # négatifs à toutes les autres du batch, sans avoir à les chercher.
        pois_par_question = [set(df_question["pois"].iloc[pos]) for pos in chunk]
        poi_ids = []
        for pois in pois_par_question:
            for poi in pois:
                if poi not in poi_ids:
                    poi_ids.append(poi)

        for i, pos in enumerate(chunk):
            candidats = hard_negatives(pos, df_question, df_osm, nb_unmatches)
            for poi in tirer(rng, candidats, n_hard):
                if poi not in poi_ids and poi not in pois_par_question[i]:
                    poi_ids.append(poi)

        batches.append({
            "questions": [df_question["question"].iloc[pos] for pos in chunk],
            "poi_ids": poi_ids,
        })

    return batches

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
        pois = list(set(df_question["pois"].iloc[pos]))
        if is_test:
            rows.append({
                "question": df_question["question"].iloc[pos],
                "pois": pois,
            })
        else:
            candidats = hard_negatives(pos, df_question, df_osm, nb_unmatches)
            hard_pois = [p for p in tirer(rng, candidats, n_hard) if p not in pois]
            rows.append({
                "question": df_question["question"].iloc[pos],
                "pois": pois,
                "hard_pois": hard_pois,
            })
    return DataFrame(rows)

def index_ground_truth(df_questions: DataFrame) -> dict[tuple[str, ...], list[PoiId]]:
    """Question (en tuple) -> ses POI pertinents.

    Args:
        df_questions: colonnes `question` et `pois`.

    Returns:
        Dédoublonné, ordre d'origine conservé.
    """
    return {tuple(q): list(dict.fromkeys(pois))
            for q, pois in zip(df_questions["question"], df_questions["pois"])}

def change_answer_batch(
        batch: Batch, 
        df_questions: DataFrame, 
        nb_answers: int,
        rng: Generator,
    ) -> Batch:
    """Rejoue le même batch avec d'autres réponses.

    Args:
        batch: batch produit par `make_batches`.
        df_questions: vérité terrain, colonnes `question` et `pois`.
        n_answers: nombre de réponses conservées par question.
        rng: à passer pour que l'augmentation soit reproductible.

    Returns:
        Un nouveau batch ; `batch` n'est pas modifié.
    """
    ground = index_ground_truth(df_questions)

    questions, poi_ids = batch["questions"], batch["poi_ids"]
    verites = [set(ground[tuple(q)]) for q in questions]
    reponses = []

    for question in questions:
        candidats = ground[tuple(question)]
        reponses.extend(tirer(rng, candidats, min(nb_answers, len(candidats))))
    poi_ids_out = list(dict.fromkeys(reponses))
    pertinents = set().union(*verites) if verites else set()

    deja = set(poi_ids_out)
    poi_ids_out += [poi for poi in poi_ids
                    if poi not in deja and poi not in pertinents]
    return {"questions": list(questions), "poi_ids": poi_ids_out}

def augment_benchmark(
        batches: list[Batch], 
        df_questions: DataFrame,
        n_times: int, 
        nb_answers: int,
        rng: Generator,
    ) -> list[Batch]:
    """Ajoute n_times variantes de chaque batch, avec d'autres réponses.

    Args:
        batches: batchs produits par `make_batches` ; allongée sur place.
        df_questions: vérité terrain.
        n_times: variantes par batch.
        rng: générateur numpy

    Returns:
        `batches`, de taille (1 + n_times) x len(batches).

    TODO: à n_times=100, les masques pèsent 65 Mo pour 1 000 questions et 3,2 Go
        pour 50 000. Tirer les réponses dans le collator coûterait une mémoire
        constante.
    """
    for k in range(len(batches)):
        for _ in range(n_times):
            batches.append(change_answer_batch(batches[k], df_questions, nb_answers, rng=rng))
    return batches

def make_benchmark_question(
        df_question: DataFrame, 
        df_osm: DataFrame, 
        rng: Generator,
        ratio_test: float = 0.1,
        ratio_features: dict[int, float] = {1: 0.4, 2: 0.4, 3: 0.2}, 
        batch_size: int = 40, 
        n_times: int = 100,
        nb_answers: int = 5,
        nb_hard: int = 20,
        nb_unmatches: int = 1,
    ) -> tuple[list[Batch], NDArray]:
    """Split des questions, stratification du train, puis construction des batchs.

    Args:
        df_question: questions.
        df_osm: corpus de POI.
        ratio_test: part réservée au test.
        ratio_features: proportions passées à `stratify`.
        rng: générateur numpy.
        batch_size: questions par batch.
        n_hard: négatifs difficiles par question.
        nb_unmatches: transmis à `hard_negatives`.

    Returns:
        Les batchs d'entraînement, et les positions de test.
    """
    train, test = split_train_test(len(df_question), ratio_test, rng)
    train = stratify(df_question, train, ratio_features, rng)
    train_set = make_rows(df_question, df_osm, train, rng, nb_hard, nb_unmatches)
    test_set = make_rows(df_question, df_osm, test, rng, nb_hard, nb_unmatches, is_test=True)
    #train_batches = augment_benchmark(train_batches, df_question, n_times, nb_answers=nb_answers, rng=rng)
    return train_set, test_set
