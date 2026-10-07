"""Les cibles d'apprentissage dérivées d'un benchmark.

`generator/targets/` est arrivé avec f1be9c1 et n'avait aucun test. Deux
modules, deux cibles :

* `ellypses.get_ellypse` — l'ellipse qui couvre la zone de réponse d'une
  question, pour que le modèle apprenne une *région* et non une liste de POIs.
  Elle s'aiguille sur la colonne `function`, c'est-à-dire sur les mêmes chaînes
  que publie chaque template : les deux fichiers doivent bouger ensemble, et
  c'est ce couplage que la première série de tests tient ;
* `mask_anchors.make_mask` — le masque qui distingue, dans les tokens de
  l'énoncé, ceux qui nomment une ancre de ceux qui portent la relation
  spatiale.

Les deux modules sont aujourd'hui **inimportables** dans l'environnement que
`pyproject.toml` décrit, cf. `known_defects.UNIMPORTABLE_MODULES`. Le premier
test le constate en `xfail(strict=True)` ; les suivants se contentent d'un skip,
et se mettront à vérifier dès que l'import passera — que ce soit parce que les
imports morts auront été retirés ou parce que scikit-learn aura été déclaré.
"""

import ast
import importlib.util
import math
import pathlib

import pytest

from benchmark_pipeline.generator.template_question.schema import TEMPLATE_REGISTRY
from tests.known_defects import DEAD_IMPORTS, SEMANTIC_DEFECTS

ELLYPSES = "benchmark_pipeline.generator.targets.ellypses"
MASK_ANCHORS = "benchmark_pipeline.generator.targets.mask_anchors"

#: Motif commun aux skips : dire *pourquoi* le module manque, pour qu'un skip ne
#: se lise pas comme une dépendance optionnelle qu'on aurait choisi d'ignorer.
_SKIP = ("module inimportable sans l'extra `nlp`, cf. "
         "known_defects.DEAD_IMPORTS et test_target_modules_import_only_what_they_use")


def _imported_but_unused(module_name):
    """Noms importés par un module et jamais référencés ensuite.

    Lit la source plutôt que le module chargé : le contrôle doit donner le même
    résultat avec ou sans l'extra `nlp` installé, sinon il dirait tantôt oui
    tantôt non selon l'environnement — et un `xfail(strict=True)` qui dépend de
    l'environnement casse la CI au lieu de décrire le code.

    Args:
        module_name (str): Chemin d'import du module.

    Returns:
        list[str]: Les noms liés par un `import`, triés, qui n'apparaissent
            nulle part ailleurs dans le module.
    """
    source = pathlib.Path(importlib.util.find_spec(module_name).origin).read_text(
        encoding="utf-8")
    arbre = ast.parse(source)

    lies = {}
    for noeud in ast.walk(arbre):
        if isinstance(noeud, (ast.Import, ast.ImportFrom)):
            for alias in noeud.names:
                lies[alias.asname or alias.name.split(".")[0]] = noeud.lineno

    utilises = {n.id for n in ast.walk(arbre) if isinstance(n, ast.Name)}
    utilises |= {n.attr for n in ast.walk(arbre) if isinstance(n, ast.Attribute)}
    return sorted(nom for nom in lies if nom not in utilises)


# Un `xfail` par cas, et non un marqueur unique sur le test : les deux modules
# ont chacun leur ligne fautive, et un motif partagé en désignerait une seule.
@pytest.mark.parametrize("module", [
    pytest.param(name, id=name.rsplit(".", 1)[-1],
                 marks=pytest.mark.xfail(strict=True, reason=reason))
    for name, reason in sorted(DEAD_IMPORTS.items())
])
def test_target_modules_import_only_what_they_use(module):
    """Les modules de `targets/` n'importent rien qu'ils n'utilisent.

    C'est la précondition de tout le reste du fichier. Les deux importent
    scikit-learn sans s'en servir, et scikit-learn n'est pas une dépendance
    déclarée : il n'arrive que par l'extra `nlp`. Après un `uv sync` sans
    extras, les deux modules lèvent donc ModuleNotFoundError et les tests qui
    suivent sont sautés.

    Contrôle statique et non par import : il doit tomber pareil avec ou sans
    l'extra, sinon ce `xfail(strict=True)` passerait en XPASS dans la CI — qui
    installe `--all-extras` — et ferait échouer le build pour un défaut qui est
    toujours là.
    """
    morts = _imported_but_unused(module)
    assert not morts, (
        f"{module} importe sans jamais les utiliser : {morts}"
    )


@pytest.fixture(scope="module")
def ellypses():
    """Le module `targets.ellypses`, ou skip s'il reste inimportable."""
    return pytest.importorskip(ELLYPSES, reason=_SKIP)


@pytest.fixture(scope="module")
def mask_anchors():
    """Le module `targets.mask_anchors`, ou skip s'il reste inimportable."""
    return pytest.importorskip(MASK_ANCHORS, reason=_SKIP)


# --------------------------------------------------------------------------- #
# ellipse de réponse
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("template", [pytest.param(t, id=t.name)
                                      for t in TEMPLATE_REGISTRY])
def test_every_function_value_has_an_ellipse_branch(ellypses, template,
                                                    run_template):
    """Chaque valeur de `function` publiée a sa branche dans `get_ellypse`.

    `get_ellypse` lève `ValueError` sur une valeur inconnue : c'est le seul
    endroit qui relie les chaînes publiées par les douze templates à un calcul.
    Renommer un helper sans toucher à la chaîne — ou l'inverse — casse la cible
    de tous les benchmarks du template concerné, sans qu'aucun test de contrat
    ne s'en aperçoive.
    """
    bench = run_template(template)
    for i, row in bench.iterrows():
        try:
            ellypses.get_ellypse(row)
        except ValueError as exc:
            pytest.fail(
                f"{template.name} ligne {i} : function={row['function']!r} "
                f"n'a pas de branche dans get_ellypse ({exc})"
            )
        except KeyError as exc:
            pytest.fail(
                f"{template.name} ligne {i} : get_ellypse lit la colonne "
                f"{exc} que le template ne publie pas"
            )


@pytest.mark.parametrize("template", [pytest.param(t, id=t.name)
                                      for t in TEMPLATE_REGISTRY])
def test_get_ellypse_returns_five_finite_numbers(ellypses, template,
                                                 run_template):
    """L'ellipse est (dx, dy, demi-axe u, demi-axe v, inclinaison), tous finis.

    Les deux demi-axes sont des longueurs : nulles ou négatives, elles donnent
    une région vide ou retournée, et la cible devient inapprenable. Un NaN
    viendrait d'une géométrie dégénérée — deux ancres confondues, une rue de
    longueur nulle — qu'il vaut mieux voir ici que dans la fonction de perte.
    """
    bench = run_template(template)
    for i, row in bench.iterrows():
        ellipse = ellypses.get_ellypse(row)
        assert len(ellipse) == 5, (
            f"{template.name} ligne {i} : get_ellypse rend {len(ellipse)} "
            f"valeurs au lieu de 5"
        )
        assert all(math.isfinite(v) for v in ellipse), (
            f"{template.name} ligne {i} : ellipse non finie {ellipse}"
        )
        _, _, demi_u, demi_v, _ = ellipse
        assert demi_u > 0 and demi_v > 0, (
            f"{template.name} ligne {i} : demi-axes {demi_u}, {demi_v} — "
            f"la région de réponse est vide"
        )


def test_near_metric_ellipse_is_exactly_the_announced_disc(ellypses,
                                                           run_template):
    """« à moins de d mètres » : l'ellipse est le disque de rayon d, centré.

    C'est le seul template dont la bonne réponse géométrique soit connue
    exactement — le rayon est dans l'énoncé. Les autres branches reposent sur
    des constantes choisies à la main ; celle-ci est vérifiable.
    """
    from benchmark_pipeline.generator.template_question.schema import TEMPLATES_BY_NAME

    bench = run_template(TEMPLATES_BY_NAME["point_near_metric"])
    for i, row in bench.iterrows():
        dx, dy, demi_u, demi_v, inclinaison = ellypses.get_ellypse(row)
        rayon = float(row["distance"])
        assert (dx, dy) == (0.0, 0.0), (
            f"ligne {i} : le disque doit être centré sur l'ancre, décalé de "
            f"({dx}, {dy})"
        )
        assert demi_u == demi_v == rayon, (
            f"ligne {i} : rayon annoncé {rayon} m, ellipse {demi_u}×{demi_v}"
        )
        assert inclinaison == 0.0, "un disque n'a pas d'inclinaison"


@pytest.mark.parametrize("direction, signe_dx, signe_dy", [
    ("north", 0, 1), ("south", 0, -1), ("east", 1, 0), ("west", -1, 0),
])
def test_cardinal_ellipse_is_offset_towards_its_direction(ellypses, direction,
                                                          signe_dx, signe_dy):
    """L'ellipse d'une question cardinale est décalée du bon côté de l'ancre.

    Une erreur de signe ici place la région de réponse à l'opposé de ce que
    l'énoncé demande, et le modèle apprend le contraire de la relation — sans
    qu'aucun contrôle de forme ne puisse le voir.
    """
    import pandas as pd

    row = pd.Series({"function": "cardinal_azimuth_sql", "direction": direction})
    dx, dy, _, _, _ = ellypses.get_ellypse(row)

    assert math.copysign(1, dx) * (dx != 0) == signe_dx, (
        f"{direction} : décalage en x de {dx}, attendu de signe {signe_dx}"
    )
    assert math.copysign(1, dy) * (dy != 0) == signe_dy, (
        f"{direction} : décalage en y de {dy}, attendu de signe {signe_dy}"
    )


def test_get_ellypse_rejects_an_unknown_function(ellypses):
    """Une `function` inconnue doit lever, pas rendre une ellipse par défaut.

    C'est ce qui fait de `get_ellypse` le garde-fou du couplage avec les
    templates : un nom qui ne correspond à rien doit s'entendre, pas produire
    une région arbitraire que l'entraînement avalerait sans rien dire.
    """
    import pandas as pd

    with pytest.raises(ValueError, match="inconnu"):
        ellypses.get_ellypse(pd.Series({"function": "pas_un_template"}))


def test_add_ellypse_adds_one_ellipse_per_question(ellypses, run_template):
    """`add_ellypse` ajoute une colonne, une ellipse par ligne."""
    from benchmark_pipeline.generator.template_question.schema import TEMPLATES_BY_NAME

    bench = run_template(TEMPLATES_BY_NAME["point_near"]).copy()
    out = ellypses.add_ellypse(bench)

    assert "ellypse" in out.columns
    assert len(out) == len(bench)
    assert all(len(e) == 5 for e in out["ellypse"])


# --------------------------------------------------------------------------- #
# masque des ancres
# --------------------------------------------------------------------------- #

def test_make_mask_marks_the_first_token_then_the_rest(mask_anchors):
    """1 sur le premier token d'une ancre, 2 sur les suivants, 0 ailleurs.

    La distinction 1/2 est ce qui permet de recoller les tokens en entités : un
    masque binaire ne saurait pas séparer deux ancres adjacentes.
    """
    mask = mask_anchors.make_mask([5, 10, 11, 12, 7], [[10, 11, 12]])
    assert mask == [0, 1, 2, 2, 0]


def test_make_mask_handles_several_anchors(mask_anchors):
    """Deux ancres dans le même énoncé sont marquées chacune de son côté."""
    mask = mask_anchors.make_mask([1, 20, 21, 2, 30, 3], [[20, 21], [30]])
    assert mask == [0, 1, 2, 0, 1, 0]


def test_make_mask_marks_every_occurrence_of_an_anchor(mask_anchors):
    """Une ancre citée deux fois est marquée deux fois.

    L'énoncé d'`opposite_side` nomme la rue une seule fois, mais rien
    n'interdit qu'un template répète un libellé — et un modèle entraîné sur un
    masque partiel apprendrait que la seconde occurrence est du vocabulaire.
    """
    mask = mask_anchors.make_mask([9, 9, 4, 9, 9], [[9, 9]])
    assert mask == [1, 2, 0, 1, 2]


def test_make_mask_does_not_let_two_anchors_overlap(mask_anchors):
    """Un token déjà pris par une ancre n'est pas réécrit par une autre.

    Deux ancres dont les tokens se chevauchent — un nom de rue contenu dans un
    nom de place — produiraient sinon un masque où un 2 suit un 2 sans 1,
    c'est-à-dire une entité sans début.
    """
    mask = mask_anchors.make_mask([7, 8, 9], [[7, 8], [8, 9]])
    assert mask == [1, 2, 0]
    assert 2 not in mask[:1], "un 2 ne doit jamais ouvrir une entité"


def test_make_mask_ignores_an_empty_anchor(mask_anchors):
    """Une ancre sans token ne doit pas tout marquer.

    `make_mask` est appelé avec le résultat d'un tokenizer : une ancre dont le
    nom est vide ou absent rend une liste vide, et une boucle naïve la
    trouverait à chaque position.
    """
    assert mask_anchors.make_mask([1, 2, 3], [[]]) == [0, 0, 0]
    assert mask_anchors.make_mask([1, 2, 3], []) == [0, 0, 0]


def test_make_mask_returns_the_length_of_the_query(mask_anchors):
    """Le masque a toujours la longueur des tokens de l'énoncé.

    Il est consommé en parallèle des tokens ; un décalage d'un cran décalerait
    toutes les étiquettes sans lever nulle part.
    """
    tokens = [4, 5, 6, 7, 8]
    for anchors in ([[5, 6]], [[99]], [], [[4, 5, 6, 7, 8]]):
        assert len(mask_anchors.make_mask(tokens, anchors)) == len(tokens)


def test_get_anchors_collects_the_context_labels(mask_anchors, run_template):
    """`get_anchors` récupère les libellés que l'énoncé nomme.

    Lit la ligne réelle d'un template à deux ancres : les deux noms doivent
    sortir, et rien d'autre — une colonne de catégorie ou de coordonnée
    marquerait des tokens qui ne sont pas des entités.
    """
    from benchmark_pipeline.generator.template_question.schema import TEMPLATES_BY_NAME

    bench = run_template(TEMPLATES_BY_NAME["point_between"])
    row = bench.iloc[0]
    anchors = mask_anchors.get_anchors(row)

    assert set(anchors) == {row["anchor_a_name"], row["anchor_b_name"]}


@pytest.mark.xfail(strict=True,
                   reason=SEMANTIC_DEFECTS["mask_anchors_ignore_le_poi_de_reference"])
def test_get_anchors_collects_the_reference_poi_of_opposite_side(mask_anchors,
                                                                 run_template):
    """Le POI de référence d'`opposite_side` est une ancre comme les autres.

    Son énoncé le nomme — « cafe across rue X from poi7 » — donc ses tokens
    doivent être masqués. `ANCHOR_COL` ne reconnaît que `anchor|area|street`,
    jamais `poi_y_name` : c'est le seul libellé cité par un énoncé qui échappe
    au masque.
    """
    from benchmark_pipeline.generator.template_question.schema import TEMPLATES_BY_NAME

    bench = run_template(TEMPLATES_BY_NAME["street_opposite_side"])
    row = bench.iloc[0]
    anchors = mask_anchors.get_anchors(row)

    assert row["poi_y_name"] in anchors, (
        f"ancres relevées {anchors} — le POI de référence "
        f"{row['poi_y_name']!r}, pourtant nommé par l'énoncé "
        f"{row['query']!r}, n'en fait pas partie"
    )
