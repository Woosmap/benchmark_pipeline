"""Invariants que tout benchmark doit respecter, quel que soit le template.

Ces tests ne regardent pas la *sémantique* spatiale (« ce POI est-il vraiment au
nord ? », c'est l'affaire de `test_template_semantics.py`) mais la cohérence
structurelle entre l'énoncé et la réponse : les listes sont-elles parallèles, le
rang est-il un classement, les POIs annoncés existent-ils, sont-ils de la
catégorie demandée, l'ancre s'est-elle retrouvée dans sa propre réponse.

Chaque template est un cas paramétré. Ceux dont le défaut est connu portent un
`xfail(strict=True)` posé depuis `known_defects.py` : ils sont comptés `xfailed`
tant qu'ils sont cassés, et basculent en échec `XPASS` le jour où ils sont
réparés — ce qui signale qu'il faut retirer le marqueur.
"""

import importlib

import pandas as pd
import pytest

from src.template_question.schema import (
    BENCHMARK_CORE_COLUMNS,
    REQUIRED_RESULT_COLUMNS,
    RESULT_LIST_COLUMNS,
    TEMPLATE_REGISTRY,
    _as_list,
    validate_benchmark,
)
from tests.known_defects import (
    BROKEN_TEMPLATES,
    IMPORT_ERRORS,
    INCOHERENT_TEMPLATES,
    UNANSWERABLE_TEMPLATES,
    UNPLOTTABLE_TEMPLATES,
    merge_reasons,
)


def for_each_template(*defect_tables):
    """Paramètre un test sur les 12 templates, xfail pour ceux qui échoueront.

    Les marqueurs sont posés **par test**, pas une fois pour le module : un
    template peut très bien s'importer et ne casser qu'à l'appel du générateur
    (c'est le cas des quatre `area_*`). Un `xfail` global serait alors un XPASS
    sur `test_module_imports` et ferait échouer la suite pour de mauvaises
    raisons. Chaque test déclare donc les familles de défauts qui le concernent.
    """
    expected_failures = merge_reasons(*defect_tables)
    cases = [
        pytest.param(
            template, id=template.name,
            marks=(
                [pytest.mark.xfail(strict=True,
                                   reason=expected_failures[template.name])]
                if template.name in expected_failures else []
            ),
        )
        for template in TEMPLATE_REGISTRY
    ]
    return pytest.mark.parametrize("template", cases)


# --------------------------------------------------------------------------- #
# le générateur tourne et publie ce qu'il annonce
# --------------------------------------------------------------------------- #

@for_each_template(IMPORT_ERRORS)
def test_module_imports(template):
    """Le module du template s'importe sans effet de bord.

    Un module qui exécute du code au chargement (appel de générateur resté en
    bas de fichier) est inutilisable ailleurs que dans le notebook d'origine.
    """
    importlib.import_module(template.module)


@for_each_template(BROKEN_TEMPLATES)
def test_generator_produces_questions(template, run_template):
    """Le générateur renvoie un DataFrame non vide."""
    bench = run_template(template)
    assert isinstance(bench, pd.DataFrame), (
        f"{template.generator} doit renvoyer un DataFrame, "
        f"pas un {type(bench).__name__}"
    )
    assert not bench.empty, (
        f"{template.generator} n'a produit aucune question : le tirage ne "
        f"trouve pas de candidat valide dans le corpus de test"
    )


@for_each_template(BROKEN_TEMPLATES)
def test_declared_columns_present(template, run_template):
    """Toutes les colonnes annoncées par le template sont là.

    Le registre est la source de vérité : s'il promet `area_geometry`, la
    visualisation et l'enregistrement comptent dessus.
    """
    bench = run_template(template)
    expected = (BENCHMARK_CORE_COLUMNS + REQUIRED_RESULT_COLUMNS
                + tuple(template.context_columns))
    missing = [c for c in expected if c not in bench.columns]
    assert not missing, (
        f"colonnes annoncées mais absentes de {template.name} : {missing}\n"
        f"colonnes produites : {sorted(bench.columns)}"
    )


@for_each_template(BROKEN_TEMPLATES, INCOHERENT_TEMPLATES)
def test_result_lists_complete(template, run_template):
    """Les quatre listes de vérité terrain sont publiées.

    `results_poi_dist` et `results_poi_rank` ne sont pas facultatives : sans
    rang, une réponse n'est plus un classement et ne peut pas servir à calculer
    un nDCG ou un MRR.
    """
    bench = run_template(template)
    missing = [c for c in RESULT_LIST_COLUMNS if c not in bench.columns]
    assert not missing, (
        f"{template.name} ne publie pas {missing} : la réponse n'est pas un "
        f"classement exploitable"
    )


@for_each_template(BROKEN_TEMPLATES)
def test_function_column_names_a_real_callable(template, run_template):
    """La colonne `function` désigne une fonction qui existe vraiment.

    C'est la seule trace, dans le benchmark enregistré, de la manière dont la
    vérité terrain a été calculée. Un nom périmé rend le jeu inauditable.
    """
    bench = run_template(template)
    module = importlib.import_module(template.module)
    for name in bench["function"].unique():
        assert callable(getattr(module, name, None)), (
            f"function={name!r} annoncée par {template.name} n'est pas un "
            f"callable de {template.module}"
        )


# --------------------------------------------------------------------------- #
# cohérence question ↔ réponse
# --------------------------------------------------------------------------- #

@for_each_template(BROKEN_TEMPLATES, INCOHERENT_TEMPLATES)
def test_benchmark_is_coherent(template, run_template, df_osm):
    """Contrôle groupé de tous les invariants de `validate_benchmark`.

    Listes parallèles, rang contigu 1-based, absence de doublon, existence des
    `poi_id`, appariement id↔nom, catégorie des résultats, distances finies et
    croissantes, exclusion de l'ancre, mention de la catégorie dans l'énoncé.

    Les questions sans réponse sont tolérées ici : elles sont inévaluables, pas
    incohérentes, et `test_no_unanswerable_questions` s'en charge séparément.
    """
    bench = run_template(template)
    problems = validate_benchmark(
        bench, df_osm,
        dist_is_ranking_key=template.dist_is_ranking_key,
        allow_empty=True,
    )
    assert not problems, (
        f"{len(problems)} incohérence(s) dans {template.name} :\n  - "
        + "\n  - ".join(problems[:15])
        + (f"\n  … et {len(problems) - 15} autres" if len(problems) > 15 else "")
    )


@for_each_template(BROKEN_TEMPLATES)
def test_query_mentions_its_context(template, run_template):
    """L'énoncé nomme bien l'objet sur lequel il porte.

    Une question « cafe near X » dont la colonne `anchor_name` dit Y est
    incohérente : l'évaluation et l'affichage désigneraient deux lieux
    différents. C'est exactement le symptôme qu'a produit la réécriture partielle
    des f-strings lors de la sortie du notebook.
    """
    bench = run_template(template)
    if template.label_column is None:
        pytest.skip("template sans libellé de contexte")

    mismatched = [
        (row["query"], row[template.label_column])
        for _, row in bench.iterrows()
        if str(row[template.label_column]) not in str(row["query"])
    ]
    assert not mismatched, (
        f"{len(mismatched)} énoncé(s) de {template.name} ne mentionnent pas "
        f"leur {template.label_column} ; ex. query={mismatched[0][0]!r} "
        f"mais {template.label_column}={mismatched[0][1]!r}"
    )


@for_each_template(BROKEN_TEMPLATES, UNANSWERABLE_TEMPLATES)
def test_no_unanswerable_questions(template, run_template):
    """Aucune question n'est livrée sans réponse.

    Une question dont la vérité terrain est vide ne peut pas noter un modèle :
    tout classement obtient le même score. Ces questions doivent être écartées
    au tirage, pas laissées dans le jeu.
    """
    bench = run_template(template)
    empty = [
        (i, row["query"]) for i, row in bench.iterrows()
        if not _as_list(row["results_poi_id"])
    ]
    assert not empty, (
        f"{len(empty)}/{len(bench)} questions de {template.name} sont sans "
        f"réponse ; ex. ligne {empty[0][0]} : {empty[0][1]!r}"
    )


# --------------------------------------------------------------------------- #
# compatibilité avec la visualisation
# --------------------------------------------------------------------------- #

@for_each_template(BROKEN_TEMPLATES, UNPLOTTABLE_TEMPLATES)
def test_satisfies_plot_question_preconditions(template, run_template, df_osm):
    """Les préconditions vérifiées par `plot_question` sont remplies.

    `questionA_viz.plot_question` lève explicitement sur listes non parallèles,
    rangs dupliqués et `poi_id` introuvables (questionA_viz.py:116-131). Un
    benchmark qui ne passe pas ces contrôles n'est pas affichable, donc pas
    relisible à l'œil.

    Marqué depuis `UNPLOTTABLE_TEMPLATES` et non `INCOHERENT_TEMPLATES` : il
    manque `results_poi_dist` à `area_direction`, dont `plot_question` n'a pas
    besoin. Le marquer ici le ferait XPASS et échouer la suite pour un défaut
    qu'il n'a pas.
    """
    bench = run_template(template)
    geo_index = df_osm.set_index("poi_id").index

    for i, row in bench.iterrows():
        ids = _as_list(row["results_poi_id"])
        names = _as_list(row["results_poi_name"])
        ranks = _as_list(row["results_poi_rank"])
        if not ids:
            continue
        assert len(ids) == len(ranks) == len(names), (
            f"{template.name} ligne {i} : listes non parallèles "
            f"({len(ids)} ids, {len(ranks)} rangs, {len(names)} noms) — "
            f"plot_question lève dessus"
        )
        assert len(set(ranks)) == len(ranks), (
            f"{template.name} ligne {i} : rangs dupliqués"
        )
        unknown = [p for p in ids if p not in geo_index]
        assert not unknown, (
            f"{template.name} ligne {i} : {len(unknown)} poi_id absents du "
            f"corpus, ex. {unknown[:3]}"
        )
