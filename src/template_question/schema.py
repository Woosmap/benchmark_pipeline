"""Schéma d'un benchmark de questions et vérification de sa cohérence.

Un benchmark est un `DataFrame` « une ligne = une question ». Chaque ligne porte
deux choses :

* un **contexte**, qui décrit la question posée — la catégorie cherchée
  (`category_query`), l'énoncé en langue naturelle (`query`), et selon le
  template une ancre ponctuelle (`anchor_*`), une zone (`area_*`) ou une rue
  (`street_*`) ;
* une **vérité terrain**, sous forme de listes parallèles `results_poi_id`,
  `results_poi_name`, `results_poi_dist`, `results_poi_rank`, ordonnées par
  pertinence décroissante.

Ce module est la source de vérité unique sur ce schéma : il alimente à la fois la
paramétrisation des tests (`tests/`) et l'écriture sur disque
(`src.utils.dataset_io`), pour qu'il n'y ait pas deux listes de colonnes à
maintenir en parallèle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# colonnes
# --------------------------------------------------------------------------- #

#: Colonnes présentes dans tout benchmark, quel que soit le template.
BENCHMARK_CORE_COLUMNS = ("query", "category_query", "function")

#: Les listes parallèles de la vérité terrain. `results_poi_id` et
#: `results_poi_name` sont exigées partout ; `dist` et `rank` sont vérifiées
#: quand elles existent, parce que `point_between` ne les produit pas encore.
RESULT_LIST_COLUMNS = (
    "results_poi_id",
    "results_poi_name",
    "results_poi_dist",
    "results_poi_rank",
)

REQUIRED_RESULT_COLUMNS = ("results_poi_id", "results_poi_name")

#: Colonnes contenant des géométries shapely. Elles ne survivent pas à un
#: aller-retour parquet sans passer par du WKB, cf. `src.utils.dataset_io`.
GEOMETRY_COLUMNS = (
    "area_geometry",
    "street_geometry",
    "street_a_geometry",
    "street_b_geometry",
    "poi_y_geometry",
)

#: Colonnes qui désignent un POI servant d'ancre à la question. Une ancre ne
#: doit jamais figurer dans sa propre réponse.
ANCHOR_ID_COLUMNS = ("anchor_index", "point_b_index", "poi_y_id")


# --------------------------------------------------------------------------- #
# registre des templates
# --------------------------------------------------------------------------- #

class Template:
    """Description d'un générateur de questions.

    Args:
        name: Nom court du template, utilisé comme identifiant de test et comme
            nom de fichier à l'enregistrement.
        module: Chemin d'import du module qui contient le générateur.
        generator: Nom de la fonction `make_question_*`.
        fixtures: Noms des fixtures pytest à passer en positionnel au
            générateur (`df_osm` seul, ou `df_osm` + `df_area` / `df_streets`).
        context_columns: Colonnes de contexte propres au template, en plus de
            `BENCHMARK_CORE_COLUMNS`.
        label_column: Colonne portant le libellé que l'énoncé `query` doit
            mentionner (nom de l'ancre, de la zone ou de la rue).
        dist_is_ranking_key: Vrai si `results_poi_dist` est bien la grandeur qui
            a servi à classer. Faux pour `street_opposite_side`, qui classe par
            écart le long de la rue mais publie la distance à la rue — l'ordre
            croissant de `dist` n'y est donc pas un invariant.
    """

    def __init__(self, name, module, generator, fixtures, context_columns,
                 label_column=None, dist_is_ranking_key=True):
        self.name = name
        self.module = module
        self.generator = generator
        self.fixtures = fixtures
        self.context_columns = context_columns
        self.label_column = label_column
        self.dist_is_ranking_key = dist_is_ranking_key

    def __repr__(self):
        return f"Template({self.name})"


_PKG = "src.template_question.type_A"

_ANCHOR_COLS = ("anchor_index", "anchor_name", "anchor_category",
                "anchor_x", "anchor_y", "same_cat")
_POINT_B_COLS = ("point_b_index", "point_b_name", "point_b_category",
                 "point_b_x", "point_b_y")
_AREA_COLS = ("area_index", "area_name", "area_geometry")

#: Les 12 templates de type A. L'ordre est celui de la progression logique :
#: proximité, puis direction, puis zone, puis rue.
TEMPLATE_REGISTRY = (
    Template(
        "point_near", f"{_PKG}.point_near", "make_question_point_near",
        ("df_osm",), _ANCHOR_COLS, label_column="anchor_name",
    ),
    Template(
        "point_near_metric", f"{_PKG}.point_near_metric",
        "make_question_point_near_metric",
        ("df_osm",), _ANCHOR_COLS + ("distance",), label_column="anchor_name",
    ),
    Template(
        "point_near_cardinal", f"{_PKG}.point_near_cardinal",
        "make_question_point_near_cardinal",
        ("df_osm",), _ANCHOR_COLS + ("direction",), label_column="anchor_name",
    ),
    Template(
        "point_towards", f"{_PKG}.point_towards", "make_question_point_towards",
        ("df_osm",), _ANCHOR_COLS + _POINT_B_COLS, label_column="anchor_name",
    ),
    Template(
        "point_between", f"{_PKG}.point_between", "make_question_point_between",
        ("df_osm",), _ANCHOR_COLS + _POINT_B_COLS, label_column="anchor_name",
    ),
    Template(
        "area_inside", f"{_PKG}.area_inside", "make_question_area_inside",
        ("df_osm", "df_area"), _AREA_COLS, label_column="area_name",
    ),
    Template(
        "area_outside", f"{_PKG}.area_outside", "make_question_area_outside",
        ("df_osm", "df_area"), _AREA_COLS, label_column="area_name",
    ),
    Template(
        "area_border", f"{_PKG}.area_border", "make_question_area_border",
        ("df_osm", "df_area"), _AREA_COLS, label_column="area_name",
    ),
    Template(
        "area_direction", f"{_PKG}.area_direction", "make_question_area_direction",
        ("df_osm", "df_area"), _AREA_COLS + ("direction",), label_column="area_name",
    ),
    Template(
        "street_along", f"{_PKG}.street_along", "make_question_street_along",
        ("df_osm", "df_streets"),
        ("street_index", "street_name", "street_geometry"),
        label_column="street_name",
    ),
    Template(
        "street_cross", f"{_PKG}.street_cross", "make_question_street_cross",
        ("df_osm", "df_streets"),
        ("street_a_index", "street_a_name", "street_a_geometry",
         "street_b_index", "street_b_name", "street_b_geometry"),
        label_column="street_a_name",
    ),
    Template(
        "street_opposite_side", f"{_PKG}.street_opposite_side",
        "make_question_street_opposite_side",
        ("df_osm", "df_streets"),
        ("street_index", "street_name", "street_geometry",
         "poi_y_id", "poi_y_name", "poi_y_geometry"),
        label_column="street_name",
        # classe par `score` (écart le long de la rue), publie `cross`
        # (distance à la rue) : cf. street_opposite_side.py:137
        dist_is_ranking_key=False,
    ),
)

TEMPLATES_BY_NAME = {t.name: t for t in TEMPLATE_REGISTRY}


# --------------------------------------------------------------------------- #
# vérification de cohérence
# --------------------------------------------------------------------------- #

def _as_list(value):
    """Normalise une cellule de colonne-liste.

    Parquet restitue les listes en `np.ndarray` ; les générateurs produisent des
    `list`. Les deux doivent se comparer, d'où cette normalisation.
    """
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple, pd.Series)):
        return list(value)
    return [value]


def validate_benchmark(bench, df_osm=None, dist_is_ranking_key=True,
                       max_rows=None, allow_empty=False):
    """Contrôle la cohérence question ↔ réponse d'un benchmark.

    Ne lève pas : renvoie la liste des incohérences trouvées, pour que l'appelant
    décide quoi en faire — échouer un test, ou seulement avertir à l'écriture.

    Les contrôles qui ont besoin du corpus (existence des `poi_id`, appariement
    id↔nom, catégorie des résultats) sont sautés si `df_osm` est absent.

    Args:
        bench (DataFrame): Le benchmark à vérifier.
        df_osm (GeoDataFrame | None): Corpus de POIs, avec `poi_id`, `poi_name`
            et `category`. Sans lui, seuls les contrôles internes tournent.
        dist_is_ranking_key (bool): Si faux, ne vérifie pas que
            `results_poi_dist` croît avec le rang. Cf. `Template`.
        max_rows (int | None): Ne vérifie que les `max_rows` premières lignes.
        allow_empty (bool): Si vrai, une question sans réponse est ignorée au
            lieu d'être signalée. Une réponse vide n'est pas une *incohérence* —
            rien n'y contredit l'énoncé — mais une question **inévaluable** : un
            modèle de recherche ne peut pas être noté dessus. Les deux défauts se
            corrigent à des endroits différents, d'où la séparation.

    Returns:
        list[str]: Messages d'incohérence, vide si le benchmark est cohérent.
    """
    problems = []

    if not isinstance(bench, pd.DataFrame):
        return [f"attendu un DataFrame, reçu {type(bench).__name__}"]
    if bench.empty:
        return ["benchmark vide : aucune question générée"]

    missing = [c for c in BENCHMARK_CORE_COLUMNS + REQUIRED_RESULT_COLUMNS
               if c not in bench.columns]
    if missing:
        problems.append(f"colonnes manquantes : {missing}")
        return problems

    present_lists = [c for c in RESULT_LIST_COLUMNS if c in bench.columns]
    for col in RESULT_LIST_COLUMNS:
        if col not in bench.columns:
            problems.append(f"colonne de résultats absente : {col}")

    if df_osm is not None:
        name_by_id = dict(zip(df_osm["poi_id"], df_osm["poi_name"]))
        cat_by_id = dict(zip(df_osm["poi_id"], df_osm["category"]))
    else:
        name_by_id = cat_by_id = None

    frame = bench if max_rows is None else bench.head(max_rows)

    for idx, row in frame.iterrows():
        ids = _as_list(row["results_poi_id"])
        where = f"ligne {idx} ({row.get('query', '?')!r})"

        # 2. une question sans réponse n'est pas exploitable
        if not ids:
            if not allow_empty:
                problems.append(f"{where} : réponse vide")
            continue

        # 4. listes parallèles
        lengths = {c: len(_as_list(row[c])) for c in present_lists}
        if len(set(lengths.values())) > 1:
            problems.append(f"{where} : listes non parallèles {lengths}")
            continue

        # 5. rang contigu, 1-based, trié
        if "results_poi_rank" in present_lists:
            ranks = _as_list(row["results_poi_rank"])
            expected = list(range(1, len(ids) + 1))
            if [int(r) for r in ranks] != expected:
                problems.append(
                    f"{where} : results_poi_rank attendu 1..{len(ids)}, "
                    f"reçu {ranks[:5]}{'…' if len(ranks) > 5 else ''}"
                )

        # 6. pas de doublon dans une réponse
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            problems.append(f"{where} : poi_id dupliqués {dupes[:5]}")

        # 10. distances positives, finies, croissantes avec le rang
        if "results_poi_dist" in present_lists:
            dists = [float(d) for d in _as_list(row["results_poi_dist"])]
            if any(not np.isfinite(d) for d in dists):
                problems.append(f"{where} : results_poi_dist contient NaN/inf")
            elif any(d < 0 for d in dists):
                problems.append(f"{where} : results_poi_dist négative")
            elif dist_is_ranking_key and dists != sorted(dists):
                first = next(i for i in range(1, len(dists))
                             if dists[i] < dists[i - 1])
                problems.append(
                    f"{where} : results_poi_dist décroît au rang {first + 1} "
                    f"({dists[first - 1]:.1f} → {dists[first]:.1f})"
                )

        # 11. l'ancre est absente de sa propre réponse
        for col in ANCHOR_ID_COLUMNS:
            if col in bench.columns and row[col] in ids:
                problems.append(f"{where} : {col}={row[col]} figure dans sa propre réponse")

        # 12. l'énoncé mentionne la catégorie cherchée
        query = row["query"]
        if not isinstance(query, str) or not query.strip():
            problems.append(f"{where} : query vide")
        elif str(row["category_query"]) not in query:
            problems.append(
                f"{where} : query ne mentionne pas category_query="
                f"{row['category_query']!r}"
            )

        if df_osm is None:
            continue

        # 7. les POIs de la réponse existent
        unknown = [i for i in ids if i not in name_by_id]
        if unknown:
            problems.append(
                f"{where} : {len(unknown)} poi_id absents du corpus, "
                f"ex. {unknown[:3]}"
            )
            continue

        # 8. l'appariement id ↔ nom n'a pas été mélangé
        names = _as_list(row["results_poi_name"])
        mismatched = [(i, n, name_by_id[i])
                      for i, n in zip(ids, names) if name_by_id[i] != n]
        if mismatched:
            i, got, exp = mismatched[0]
            problems.append(
                f"{where} : appariement id↔nom cassé pour poi_id={i} "
                f"(benchmark {got!r}, corpus {exp!r}) — {len(mismatched)} cas"
            )

        # 9. tout résultat est de la catégorie demandée
        wrong_cat = [(i, cat_by_id[i]) for i in ids
                     if cat_by_id[i] != row["category_query"]]
        if wrong_cat:
            problems.append(
                f"{where} : {len(wrong_cat)} résultats hors catégorie "
                f"{row['category_query']!r}, ex. {wrong_cat[:3]}"
            )

    return problems
