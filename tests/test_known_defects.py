"""Démonstration ciblée des défauts qui ne se voient pas au niveau du contrat.

`test_template_contract.py` et `test_template_semantics.py` couvrent déjà ce qui
casse un benchmark entier. Restent des défauts plus fins, qui n'empêchent pas la
sortie d'être bien formée mais faussent le volume produit, ou acceptent en
silence une entrée absurde. Chacun a ici son test dédié, au plus près de la
fonction fautive, pour que le jour de la correction on sache exactement quoi
relancer.

Les tests marqués `@defect` ou `@incoherent` sont `xfail(strict=True)` : ils
passent au vert le jour où le défaut est corrigé, ce que pytest signale alors
comme un XPASS à traiter. Les autres sont des tests de non-régression
ordinaires — plusieurs d'entre eux démontraient un défaut jusqu'à sa
correction, et sont restés pour garder le comportement acquis.
"""

import numpy as np
import pytest

from tests.known_defects import BROKEN_TEMPLATES, INCOHERENT_TEMPLATES, SEMANTIC_DEFECTS

gpd = pytest.importorskip("geopandas")
from shapely.geometry import LineString, MultiLineString

from tests.conftest import OX, OY, STREET_H_Y


def defect(key, blocked_by=None):
    """`xfail(strict=True)` portant le motif du défaut `key`.

    Args:
        key: Clé dans `SEMANTIC_DEFECTS`, le défaut que le test démontre.
        blocked_by: Nom d'un template de `BROKEN_TEMPLATES` dont la panne
            empêche, en plus, d'atteindre ce défaut. Corriger `key` seul ne
            suffira donc pas à faire passer le test — le motif le dit
            explicitement, pour qu'un XPASS manquant ne soit pas pris pour un
            oubli. Inutilisé tant que `BROKEN_TEMPLATES` est vide ; conservé
            parce que la situation s'est déjà produite deux fois.
    """
    reason = SEMANTIC_DEFECTS[key]
    if blocked_by is not None:
        reason = f"{reason}\n  — inatteignable tant que : {BROKEN_TEMPLATES[blocked_by]}"
    return pytest.mark.xfail(strict=True, reason=reason)


def incoherent(template_name):
    """`xfail(strict=True)` portant le motif d'`INCOHERENT_TEMPLATES`.

    Pour les défauts que `test_benchmark_is_coherent` constate déjà en bloc,
    mais dont il vaut la peine de montrer la cause isolément : son message
    énumère les symptômes, pas la ligne fautive.
    """
    return pytest.mark.xfail(strict=True, reason=INCOHERENT_TEMPLATES[template_name])


# --------------------------------------------------------------------------- #
# volumétrie : `nb_q` est un **plafond**
# --------------------------------------------------------------------------- #
#
# Un générateur a le droit de rendre moins que `nb_q` — le corpus ne permet pas
# toujours d'en produire autant. Il n'a pas le droit d'en rendre plus. Les tests
# de cette section vérifient donc `<= nb_q`, et les deux qui constataient une
# sous-production ont été retirés : ils affirmaient un contrat qui n'existe pas.

@pytest.mark.parametrize("nb_q", [10, 14])
def test_point_near_metric_honours_nb_q(df_osm, nb_q):
    """`nb_q` pilote le nombre de questions, pas le nombre d'ancres.

    La boucle tirait `nb_q` ancres puis déclinait chacune sur les quatre rayons
    de `list_distance` sans diviser le quota : 40 questions pour nb_q=10, 56
    pour nb_q=14, soit exactement 4×. La faute existait avant f1be9c1 sous une
    autre forme — le quota était divisé par la *somme* au lieu du produit — et
    le refactor l'avait réintroduite en supprimant la division.
    """
    from benchmark_pipeline.generator.template_question.geospatial.point_near_metric import (
        make_question_point_near_metric,
    )

    bench = make_question_point_near_metric(df_osm, nb_q=nb_q, seed=42)
    assert len(bench) <= nb_q, (
        f"nb_q={nb_q} est un plafond, {len(bench)} questions produites "
        f"(facteur {len(bench) / nb_q:.2f})"
    )


@pytest.mark.parametrize("nb_q", [10, 14])
def test_point_near_cardinal_honours_nb_q(df_osm, nb_q):
    """Même faute de quota, avec les quatre secteurs cardinaux.

    40 questions pour nb_q=10, 56 pour nb_q=14. Le quota se divise en amont
    plutôt que de tronquer après coup : les quatre secteurs d'une même ancre
    sont consécutifs, et couper dans le tas amputerait les dernières ancres de
    leurs directions.
    """
    from benchmark_pipeline.generator.template_question.geospatial.point_near_cardinal import (
        make_question_point_near_cardinal,
    )

    bench = make_question_point_near_cardinal(df_osm, nb_q=nb_q, seed=42)
    assert len(bench) <= nb_q, (
        f"nb_q={nb_q} est un plafond, {len(bench)} questions produites "
        f"(facteur {len(bench) / nb_q:.2f})"
    )


@pytest.mark.parametrize("nb_q", [10, 14])
def test_area_direction_honours_nb_q(df_osm, df_area, nb_q):
    """Idem pour `area_direction` : `nb_q` zones × 4 directions.

    40 pour nb_q=10, 56 pour nb_q=14. Avant f1be9c1 la condition d'arrêt
    `while nq != n_queries_per_stratum` était inatteignable et le volume partait
    dans tous les sens (802 questions pour nb_q=10, 135 pour nb_q=40) ; il est
    passé par un stade faux mais prévisible avant d'être plafonné,
    cf. `test_area_direction_volume_grows_with_nb_q`.
    """
    from benchmark_pipeline.generator.template_question.geospatial.area_direction import (
        make_question_area_direction,
    )

    bench = make_question_area_direction(df_osm, df_area, nb_q=nb_q, seed=42)
    assert len(bench) <= nb_q, (
        f"nb_q={nb_q} est un plafond, {len(bench)} questions produites "
        f"(facteur {len(bench) / nb_q:.2f})"
    )


def test_area_direction_volume_grows_with_nb_q(df_osm, df_area):
    """À défaut du quota exact, le volume doit au moins être monotone.

    Démontrait un défaut jusqu'à f1be9c1 : la condition d'arrêt dépendait du
    plafond de tentatives et non du quota, donc demander plus de questions
    pouvait en produire moins (802 pour nb_q=10, 135 pour nb_q=40). Le tirage
    est maintenant une simple boucle sur `nb_q` zones — mesuré 40, 56 et 72
    pour nb_q 10, 14 et 18. Le test reste comme garde-fou.
    """
    from benchmark_pipeline.generator.template_question.geospatial.area_direction import (
        make_question_area_direction,
    )

    volumes = [
        len(make_question_area_direction(df_osm, df_area, nb_q=n, seed=42))
        for n in (10, 14, 18)
    ]
    assert volumes == sorted(volumes), (
        f"volumes {volumes} pour nb_q 10/14/18 : le volume décroît quand la "
        f"commande augmente"
    )


def test_point_between_honours_nb_q(df_osm):
    """`point_between` rend exactement le nombre de questions demandé.

    Il a longtemps dépassé le quota d'une boucle de catégories d'ancre (15
    questions pour nb_q=10). La stratification par `ratio.allocate` sur les
    couples (catégorie de A, catégorie de B) réellement présents dans les
    voisinages a réglé ça : mesuré 10, 14 et 110 pour ces mêmes `nb_q`. C'est
    aujourd'hui le seul template dont le volume est exact — d'où ce test de
    non-régression, qui fixe la référence pour les autres.
    """
    from benchmark_pipeline.generator.template_question.geospatial.point_between import (
        make_question_point_between,
    )

    for nb_q in (10, 14):
        bench = make_question_point_between(df_osm, nb_q=nb_q, seed=42)
        assert len(bench) <= nb_q, (
            f"nb_q={nb_q} est un plafond, {len(bench)} questions produites"
        )


def test_point_between_has_no_empty_answers_at_full_size(df_osm):
    """Aucune question d'entre-deux ne doit être livrée sans réponse.

    Le tirage écarte les couples dont le corridor est vide (`if results.empty:
    continue`, point_between.py:151) au lieu de publier la question quand même.
    Le test reste à `nb_q=110`, taille à laquelle le défaut se reproduisait
    (3 questions vides sur 130) — c'est là qu'une régression se verrait, pas à
    `nb_q=10`.
    """
    from benchmark_pipeline.generator.template_question.geospatial.point_between import (
        make_question_point_between,
    )
    from benchmark_pipeline.generator.template_question.schema import _as_list

    bench = make_question_point_between(df_osm, nb_q=110, seed=42)
    empty = [i for i, row in bench.iterrows()
             if not _as_list(row["results_poi_id"])]
    assert not empty, (
        f"{len(empty)}/{len(bench)} questions sans réponse, ex. lignes "
        f"{empty[:3]} : le corridor A→B était vide et la question a tout de "
        f"même été ajoutée"
    )


# --------------------------------------------------------------------------- #
# tirage des zones : borné par nb_q au lieu du corpus
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("template", ["area_inside", "area_direction"])
def test_area_templates_survive_a_small_area_corpus(df_osm, df_area, template):
    """Moins de zones que de questions demandées ne doit pas faire lever.

    `rng.choice(n, size=nb_q, replace=False)` exige `n >= nb_q`. Un corpus de
    trois zones — ce que produit n'importe quel filtrage un peu strict, et ce
    que contenaient les fixtures avant ce commit — fait donc lever les deux
    templates au lieu de leur faire rendre les trois questions possibles.
    """
    import importlib

    module = importlib.import_module(
        f"benchmark_pipeline.generator.template_question.geospatial.{template}"
    )
    generateur = getattr(module, f"make_question_{template}")

    bench = generateur(df_osm, df_area.head(3), nb_q=10, seed=42)
    assert not bench.empty


def test_area_inside_can_draw_every_area(df_osm, df_area):
    """Toute zone du corpus doit être tirable, pas seulement les `nb_q` premières.

    `rng.choice(min(len(areas), nb_q), size=nb_q, …)` tire dans `range(nb_q)` et
    non dans les zones : passé la `nb_q`-ième, aucune zone n'est atteignable,
    quelle que soit la graine. Mesuré : 10 zones distinctes sur 18, identiques
    d'une graine à l'autre sur 30 tirages.

    Conséquence pour le benchmark : la couverture spatiale du jeu est décidée
    par l'ordre des lignes de `df_area`, pas par le tirage.
    """
    from benchmark_pipeline.generator.template_question.geospatial.area_inside import (
        make_question_area_inside,
    )

    tirees = set()
    for seed in range(30):
        tirees |= set(make_question_area_inside(df_osm, df_area, nb_q=10,
                                                seed=seed)["area_name"])

    eligibles = set(df_area[df_area.geometry.area > 1000]["area_name"])
    jamais = sorted(eligibles - tirees)
    assert not jamais, (
        f"{len(jamais)}/{len(eligibles)} zones ne sortent sur aucune des 30 "
        f"graines, ex. {jamais[:3]}"
    )


# --------------------------------------------------------------------------- #
# vérité terrain : la bonne relation, mais pas sur les bons POIs
# --------------------------------------------------------------------------- #

def test_point_near_metric_publishes_only_its_own_radius(df_osm):
    """Chaque ligne ne porte que le lot de son propre rayon.

    La boucle calculait `result = results[results["distance"] == d]` puis publiait
    `results` : les quatre lots concaténés, identiques sur les quatre lignes d'une
    même ancre. Corrigé — ce test garde la propriété, et isole la cause (quatre
    lignes rigoureusement identiques) là où `test_benchmark_is_coherent` n'en
    voyait que les symptômes, doublons et distances décroissantes.
    """
    from benchmark_pipeline.generator.template_question.geospatial.point_near_metric import (
        make_question_point_near_metric,
    )

    bench = make_question_point_near_metric(df_osm, nb_q=10, seed=42)
    identiques = [
        (ancre, sorted(group["distance"]))
        for ancre, group in bench.groupby("anchor_index")
        if group["results_poi_id"].apply(tuple).nunique() == 1 and len(group) > 1
    ]
    assert not identiques, (
        f"{len(identiques)}/{bench['anchor_index'].nunique()} ancres ont la "
        f"même réponse pour tous leurs rayons ; ex. ancre {identiques[0][0]}, "
        f"rayons {identiques[0][1]}"
    )


def test_border_area_filters_by_the_requested_category(df_osm, df_area):
    """`pois_near_border` doit rendre la catégorie que l'énoncé demande.

    Le helper a perdu son paramètre de catégorie au passage de `border_area` à
    `pois_near_border` : il rend tous les POIs de la bande, quelle que soit leur
    catégorie, alors que l'énoncé et `category_query` en annoncent une. Mesuré :
    les 10 questions d'un tirage à nb_q=10 contiennent des POIs hors catégorie.
    """
    from benchmark_pipeline.generator.template_question.geospatial.area_border import (
        make_question_area_border,
    )

    bench = make_question_area_border(df_osm, df_area, nb_q=10, seed=42)
    categorie = df_osm.set_index("poi_id")["category"]

    fautives = [
        (i, row["category_query"],
         sorted({categorie[p] for p in row["results_poi_id"]}))
        for i, row in bench.iterrows()
        if any(categorie[p] != row["category_query"] for p in row["results_poi_id"])
    ]
    assert not fautives, (
        f"{len(fautives)}/{len(bench)} questions mélangent les catégories ; "
        f"ex. ligne {fautives[0][0]} demande {fautives[0][1]!r} et répond "
        f"{fautives[0][2]}"
    )


# --------------------------------------------------------------------------- #
# classement dégénéré : le rang n'ordonne rien
# --------------------------------------------------------------------------- #

def test_area_direction_rank_is_a_ranking(df_osm, df_area):
    """Le rang publié doit refléter un classement, pas l'ordre du corpus.

    `segregate_pois` masque sans trier : `results_poi_rank` numérote donc
    l'ordre d'apparition dans `df_osm`. Mesuré : les 33 réponses de plus d'un
    POI d'un tirage à nb_q=10 sont toutes dans l'ordre du corpus.

    Le test compare à cet ordre plutôt qu'à un classement attendu : on ne sait
    pas quel critère le template *devrait* employer (distance au centroïde,
    avancement le long de l'axe), mais on sait qu'il n'en emploie aucun.
    """
    from benchmark_pipeline.generator.template_question.geospatial.area_direction import (
        make_question_area_direction,
    )

    bench = make_question_area_direction(df_osm, df_area, nb_q=10, seed=42)
    position = {poi_id: i for i, poi_id in enumerate(df_osm["poi_id"])}

    non_triviales = [r for _, r in bench.iterrows() if len(r["results_poi_id"]) > 1]
    ordre_corpus = [r for r in non_triviales
                    if [position[p] for p in r["results_poi_id"]]
                    == sorted(position[p] for p in r["results_poi_id"])]
    assert len(ordre_corpus) < len(non_triviales), (
        f"les {len(non_triviales)} réponses de plus d'un POI sont toutes dans "
        f"l'ordre de df_osm : aucun classement n'a été appliqué"
    )


def test_border_area_separates_interior_from_boundary(df_osm, df_area):
    """`pois_near_border` distingue un POI du bord d'un POI du centre.

    Mesurée contre le polygone **plein**, la distance vaut 0 partout à
    l'intérieur : un POI au centre exact de la zone était donné aussi « en
    bordure » que celui collé au pourtour. `pois_near_border` mesure contre
    `area.boundary` depuis f1be9c1 — mesuré sur le Parc Carre : 24 POIs
    intérieurs, 20 distances distinctes entre 43,2 et 151,6 m. Ce test garde la
    correction.

    Une catégorie doit être passée : `pois_near_border` l'exige depuis qu'on lui
    a confié le filtrage, l'oubli étant précisément ce qui faisait répondre cinq
    catégories à une question qui en demandait une. Le `skip` ci-dessous couvre
    le cas où la strate retenue serait trop petite pour comparer.
    """
    from benchmark_pipeline.generator.template_question.geospatial.area_border import (
        pois_near_border,
    )

    area = df_area.geometry.iloc[0]
    results = pois_near_border(df_osm, area, band=200, category="cafe")
    inside = results[results.geometry.within(area)]
    if len(inside) < 2:
        pytest.skip("pas assez de POIs intérieurs pour comparer")

    assert inside["dist"].nunique() > 1, (
        f"les {len(inside)} POIs intérieurs sont tous à distance "
        f"{inside['dist'].iloc[0]} : la bordure n'est pas mesurée"
    )


def test_interior_point_is_at_zero_distance_from_its_polygon(df_area):
    """Rappel exécutable de la cause des classements dégénérés.

    C'est cette propriété de shapely — et non un bug — qui rend `area_inside`
    incapable d'ordonner quoi que ce soit tant qu'il mesure contre le polygone
    plein. `area_border` s'en est sorti en mesurant contre `.boundary`.
    """
    area = df_area.geometry.iloc[0]
    centre = area.centroid

    assert centre.distance(area) == 0.0
    assert centre.distance(area.boundary) > 0.0
    assert np.isclose(centre.distance(area.boundary), 200.0), (
        "le centre du Parc Carre (400x400 m) est à 200 m de son pourtour"
    )


# --------------------------------------------------------------------------- #
# entrées absurdes
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bogus", ["north s", "h so", "outh wes", ""])
def test_segregate_pois_rejects_unknown_directions(df_osm, df_area, bogus):
    """Une direction inconnue doit lever, pas être interprétée au hasard.

    `direction_area` testait `direction in "north south"`, c'est-à-dire une
    **sous-chaîne** : il répondait vrai pour `"north s"` comme pour `""`, et
    triait alors sur un axe choisi par accident. `segregate_pois` indexe un
    dictionnaire de masques, donc lève KeyError — ce test garde la correction,
    parce que la faute d'origine est invisible à la relecture.
    """
    from benchmark_pipeline.generator.template_question.geospatial.area_direction import (
        segregate_pois,
    )

    with pytest.raises(KeyError):
        segregate_pois(df_osm, df_area.geometry.iloc[0], [bogus])


@pytest.mark.parametrize("nb_q", [1, 3, 20])
def test_point_near_cardinal_accepts_any_nb_q(df_osm, nb_q):
    """Aucun plancher arbitraire sur `nb_q` : le générateur ne doit jamais lever.

    Le module refusait `nb_q <= 20` en annonçant « nb_q doit être ≥ 20 » : il
    rejetait la valeur que son propre message donnait pour valide, et la suite
    devait le lancer à 21 via une table d'exceptions dans `conftest`.

    Ce test exigeait `len(bench) == 4 * nb_q`, c'est-à-dire exactement le défaut
    de quota : chaque ancre étant déclinée sur quatre directions, le template
    rendait quatre fois ce qu'on lui demandait. `nb_q` étant un plafond, il tire
    maintenant `nb_q // 4` ancres — donc **zéro** sous `nb_q=4`, faute de pouvoir
    construire une ancre complète dans le budget — et moins encore depuis que
    les secteurs vides sont sautés, une question sans réponse étant
    inévaluable. Rendre moins est permis ; lever ne l'est pas, et c'est cela
    qu'on vérifie ici.
    """
    from benchmark_pipeline.generator.template_question.geospatial.point_near_cardinal import (
        make_question_point_near_cardinal,
    )

    bench = make_question_point_near_cardinal(df_osm, nb_q=nb_q, seed=42)
    assert len(bench) <= nb_q, (
        f"nb_q={nb_q} est un plafond, {len(bench)} questions produites"
    )


# --------------------------------------------------------------------------- #
# géométries écartées par erreur
# --------------------------------------------------------------------------- #

def test_opposite_side_handles_multilinestring_streets(df_osm):
    """Une rue en plusieurs tronçons doit rester tirable.

    Le filtre écrit `"MultiString"` (street_opposite_side.py:127), type de
    géométrie qui n'existe pas dans shapely : aucune MultiLineString ne le
    franchit. Sur un corpus qui n'en contient que, le tirage devient vide. Or
    `load_streets` produit bien des MultiLineString — les rues coupées par une
    place ou un carrefour — et le reste du module sait les traiter, puisque
    `side_of_street` et `opposite_side` prennent explicitement `geoms[0]`.
    """
    from benchmark_pipeline.generator.template_question.geospatial.street_opposite_side import (
        make_question_street_opposite_side,
    )

    # la rue horizontale des fixtures, coupée en deux tronçons : les POIs « à
    # cheval » restent donc de part et d'autre.
    multi_only = gpd.GeoDataFrame(
        [{
            "id_street": 0,
            "street_name": "rue Horizontale Coupee",
            "geometry": MultiLineString([
                [(OX - 50, STREET_H_Y), (OX + 250, STREET_H_Y)],
                [(OX + 260, STREET_H_Y), (OX + 650, STREET_H_Y)],
            ]),
        }],
        crs=2154,
    )

    bench = make_question_street_opposite_side(df_osm, multi_only, nb_q=5, seed=42)
    assert not bench.empty, (
        "aucune question produite sur un corpus de rues MultiLineString : "
        "le filtre geom_type les a toutes écartées"
    )


# --------------------------------------------------------------------------- #
# croisements qui n'en sont pas
# --------------------------------------------------------------------------- #

def test_touching_streets_only_returns_real_crossings():
    """`touching_streets` ne doit pas retenir deux rues qui ne se croisent pas.

    La tolérance de 1 m admet deux rues seulement *voisines*. Leur
    `intersection()` est alors vide, et `distance()` contre une géométrie vide
    vaut NaN : toute la vérité terrain de la question perd son ordre.
    """
    from benchmark_pipeline.generator.template_question.geospatial.street_cross import (
        touching_streets,
    )

    # deux segments parallèles distants de 0,5 m : proches, mais disjoints
    streets = gpd.GeoDataFrame(
        [
            {"id_street": 0.0,
             "geometry": LineString([(OX, OY), (OX + 100, OY)])},
            {"id_street": 1.0,
             "geometry": LineString([(OX, OY + 0.5), (OX + 100, OY + 0.5)])},
        ],
        crs=2154,
    )

    neighbours = touching_streets(streets, streets.iloc[0])
    for _, other in neighbours.iterrows():
        junction = streets.geometry.iloc[0].intersection(other.geometry)
        assert not junction.is_empty, (
            f"id_street={other['id_street']} est retenue comme croisant la "
            f"rue 0, mais leur intersection est vide — "
            f"distance() y renverra NaN"
        )


def test_street_modules_agree_on_street_identity(df_osm, df_streets):
    """Les deux modules « rue » doivent identifier une rue de la même façon.

    `street_along.py:58` tire dans `df_streets.index` ; `street_cross.py:71-72`
    tire une *valeur* de `id_street` et la passe à `.loc`, ce qui n'est correct
    que si l'index et la colonne coïncident. Sur un `df_streets` réindexé — ce
    que fait n'importe quel filtrage en amont — le second se trompe de rue en
    silence, ou lève.
    """
    from benchmark_pipeline.generator.template_question.geospatial.street_cross import (
        make_question_street_cross,
    )

    # index volontairement décorrélé de id_street, comme après un filtrage
    shuffled = df_streets.copy()
    shuffled.index = [10, 11, 12]

    bench = make_question_street_cross(df_osm, shuffled, nb_q=5, seed=42)
    valid = set(shuffled["id_street"])
    wrong = [i for i in bench["street_a_index"] if i not in valid]
    assert not wrong, (
        f"street_a_index contient des valeurs qui ne sont pas des id_street : "
        f"{wrong[:3]}"
    )


# --------------------------------------------------------------------------- #
# stratification : le helper que les templates devraient tous adopter
# --------------------------------------------------------------------------- #
#
# `ratio.allocate` n'a pas de test ailleurs, alors que `point_between` — le seul
# template dont le volume soit exact — s'appuie dessus. Les cinq tests qui
# suivent fixent son contrat pour les templates qui restent à y passer.

def test_allocate_distributes_exactly_n():
    """La somme des parts vaut exactement `n`, sans perte à l'arrondi.

    C'est toute la raison d'être de la méthode des plus forts restes : un
    `round()` naïf par catégorie perdrait ou inventerait des questions, et le
    benchmark ne ferait plus la taille commandée. C'est exactement ce que fait
    le `nb_q // len(list_cat)` d'`area_outside` et consorts.
    """
    from benchmark_pipeline.generator.template_question.ratio import allocate

    for n in (0, 1, 7, 10, 110, 1000):
        counts = allocate(n, {c: 1 for c in "abcde"})
        assert sum(counts.values()) == n, (
            f"allocate({n}, 5 catégories égales) totalise "
            f"{sum(counts.values())} au lieu de {n}"
        )


def test_allocate_respects_proportions():
    """Les parts suivent les poids demandés."""
    from benchmark_pipeline.generator.template_question.ratio import allocate

    assert allocate(10, {"a": 0.1, "b": 0.3, "c": 0.6}) == {"a": 1, "b": 3, "c": 6}


def test_allocate_accepts_raw_weights():
    """Des poids bruts valent des proportions : ils sont normalisés.

    L'appelant ne doit donc pas avoir à normaliser lui-même : passer
    `{cat: 1 for cat in list_cat}` suffit à demander l'équirépartition, sans
    division préalable.
    """
    from benchmark_pipeline.generator.template_question.ratio import allocate

    assert allocate(10, {"a": 1, "b": 3, "c": 6}) == allocate(
        10, {"a": 0.1, "b": 0.3, "c": 0.6}
    )


def test_allocate_spreads_the_remainder_over_the_largest_fractions():
    """Le reste va aux plus fortes parties fractionnaires, pas au hasard.

    10 questions sur 3 catégories égales : 3,33 chacune, donc 3 partout et une
    unité à répartir. La méthode la donne à la première par ordre de reste
    décroissant — déterministe, donc rejouable.
    """
    from benchmark_pipeline.generator.template_question.ratio import allocate

    counts = allocate(10, {c: 1 for c in "abc"})
    assert sum(counts.values()) == 10
    assert sorted(counts.values()) == [3, 3, 4], (
        f"répartition attendue 4/3/3, obtenue {sorted(counts.values())}"
    )


def test_allocate_rejects_weights_that_sum_to_zero():
    """Des poids tous nuls doivent lever, et non produire un jeu vide.

    C'est le piège dans lequel `point_between` est tombé pendant son refactor
    de stratification, en écrivant `{cat: 1//len(list_cat)}` : une division
    *entière*, donc 0 pour toute catégorie dès qu'il y en a plus d'une. Corrigé
    depuis en `1/len(list_cat)`, mais la faute est invisible à la lecture — une
    barre de plus et le jeu se vide.

    `allocate` lève bien ici. Ce test fixe ce comportement, pour que la faute
    reste bruyante au lieu de produire un benchmark vide en silence.
    """
    from benchmark_pipeline.generator.template_question.ratio import allocate

    with pytest.raises(ZeroDivisionError):
        allocate(10, {c: 1 // 5 for c in "abcde"})


# --------------------------------------------------------------------------- #
# non-régression : ce qui a été réparé doit le rester
# --------------------------------------------------------------------------- #

def test_every_template_module_imports_cleanly():
    """Les douze modules s'importent sans effet de bord.

    Le bas de `street_cross.py` avait gardé l'appel du notebook
    (`bench4=make_question_crossstreet(df_osm, df_streets)`) : importer le
    module tentait de générer un benchmark avec des variables inexistantes, et
    aucun autre module ne pouvait s'en servir. C'est le paquet entier qui doit
    être importable, puisque `geospatial/__init__.py` les importe tous pour
    peupler le registre par décorateur.
    """
    import importlib

    from benchmark_pipeline.generator.template_question.schema import TEMPLATE_REGISTRY

    for template in TEMPLATE_REGISTRY:
        module = importlib.import_module(template.module)
        assert callable(getattr(module, template.generator, None)), (
            f"{template.generator} absent de {template.module}"
        )


def test_area_name_is_the_only_spelling_of_the_area_label():
    """Les modules `area_*` lisent `area_name`, jamais `name_area`.

    Les quatre générateurs de zone lisaient `area['name_area']`, orthographe
    qu'aucun loader ne produit : tous levaient KeyError. Corrigé — ce test
    relit la source pour que l'orthographe fautive ne revienne pas par
    copier-coller d'un module à l'autre.
    """
    from benchmark_pipeline.generator.template_question.schema import TEMPLATES_BY_NAME
    from tests.conftest import template_source

    for name in ("area_inside", "area_outside", "area_border", "area_direction"):
        source = template_source(TEMPLATES_BY_NAME[name]).read_text(encoding="utf-8")
        assert "name_area" not in source, (
            f"{name}.py contient encore `name_area` ; le contrat des loaders "
            f"porte `area_name`"
        )


def test_cardinal_azimuth_convention_is_clockwise_from_north():
    """L'azimut des templates est horaire depuis le nord, pas trigonométrique.

    Ce test n'est pas un défaut mais un garde-fou : `atan2(dx, dy)` (et non
    `atan2(dy, dx)`) est le détail dont dépendent `point_near_cardinal` et
    `point_towards`. L'inverser ferait pointer « nord » vers l'est sans casser
    aucun autre test.
    """
    from tests.conftest import azimuth_deg

    assert azimuth_deg(0, 0, 0, 1) == pytest.approx(0.0)      # nord
    assert azimuth_deg(0, 0, 1, 0) == pytest.approx(90.0)     # est
    assert azimuth_deg(0, 0, 0, -1) == pytest.approx(180.0)   # sud
    assert azimuth_deg(0, 0, -1, 0) == pytest.approx(270.0)   # ouest
