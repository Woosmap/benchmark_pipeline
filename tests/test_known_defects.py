"""Démonstration ciblée des défauts qui ne se voient pas au niveau du contrat.

`test_template_contract.py` et `test_template_semantics.py` couvrent déjà ce qui
casse un benchmark entier. Restent des défauts plus fins, qui n'empêchent pas la
sortie d'être bien formée mais faussent le volume produit, ou acceptent en
silence une entrée absurde. Chacun a ici son test dédié, au plus près de la
fonction fautive, pour que le jour de la correction on sache exactement quoi
relancer.

Les tests marqués `@defect` sont `xfail(strict=True)` : ils passent au vert le
jour où le défaut est corrigé, ce que pytest signale alors comme un XPASS à
traiter. Les autres sont des tests de non-régression ordinaires — plusieurs
d'entre eux démontraient un défaut jusqu'à sa correction, et sont restés pour
garder le comportement acquis.
"""

import numpy as np
import pytest

from tests.known_defects import BROKEN_TEMPLATES, SEMANTIC_DEFECTS

gpd = pytest.importorskip("geopandas")
from shapely.geometry import LineString, MultiLineString  # noqa: E402

from tests.conftest import OX, OY, STREET_H_Y  # noqa: E402


def defect(key, blocked_by=None):
    """`xfail(strict=True)` portant le motif du défaut `key`.

    Args:
        key: Clé dans `SEMANTIC_DEFECTS`, le défaut que le test démontre.
        blocked_by: Nom d'un template de `BROKEN_TEMPLATES` dont la panne
            empêche, en plus, d'atteindre ce défaut. Corriger `key` seul ne
            suffira donc pas à faire passer le test — le motif le dit
            explicitement, pour qu'un XPASS manquant ne soit pas pris pour un
            oubli. Plus utilisé depuis que les huit templates cassés ont été
            réparés ; conservé parce que `point_between` est de nouveau dans ce
            cas et que la situation se reproduira.
    """
    reason = SEMANTIC_DEFECTS[key]
    if blocked_by is not None:
        reason = f"{reason}\n  — inatteignable tant que : {BROKEN_TEMPLATES[blocked_by]}"
    return pytest.mark.xfail(strict=True, reason=reason)


# --------------------------------------------------------------------------- #
# volumétrie : le nombre de questions produites ne correspond pas à nb_q
# --------------------------------------------------------------------------- #

@defect("point_between_quota")
@pytest.mark.parametrize("nb_q", [10])
def test_point_between_honours_nb_q(df_osm, nb_q):
    """Le quota par catégorie est tenu, mais pas le total.

    Le tirage est désormais stratifié sur `category_query` et parfaitement
    équilibré — c'est la partie réparée. Reste que la boucle interne sur les
    catégories d'ancre dépasse le compte avant de le revérifier : mesuré
    30 questions pour nb_q=10 (×3,0), 60 pour nb_q=40 (×1,5), 130 pour
    nb_q=110 (×1,18).

    Le débordement s'amortit quand `nb_q` grandit, parce que le dépassement est
    d'au plus une boucle de catégories d'ancre — constant en valeur absolue,
    donc de moins en moins visible en relatif. À nb_q=110 il repasse sous la
    tolérance de 25 %, d'où le paramétrage sur les deux tailles où il se
    manifeste réellement.
    """
    from src.template_question.geospatial.point_between import (
        make_question_point_between,
    )

    bench = make_question_point_between(df_osm, nb_q=nb_q, seed=42)
    assert len(bench) == pytest.approx(nb_q, rel=0.25), (
        f"nb_q={nb_q} demandé, {len(bench)} questions produites "
        f"(facteur {len(bench) / nb_q:.2f})"
    )


def test_point_between_has_no_empty_answers_at_full_size(df_osm):
    """Aucune question d'entre-deux ne doit être livrée sans réponse.

    Le tirage part désormais de la cible : un POI de la catégorie demandée est
    choisi d'abord, puis A et B sont construits autour de lui, ce qui met la
    réponse à l'abri du vide par construction. Le test reste à `nb_q=110`,
    taille à laquelle le défaut se reproduisait (3 questions vides sur 130) —
    c'est là qu'une régression se verrait, pas à `nb_q=10`.
    """
    from src.template_question.schema import _as_list
    from src.template_question.geospatial.point_between import (
        make_question_point_between,
    )

    bench = make_question_point_between(df_osm, nb_q=110, seed=42)
    empty = [i for i, row in bench.iterrows()
             if not _as_list(row["results_poi_id"])]
    assert not empty, (
        f"{len(empty)}/{len(bench)} questions sans réponse, ex. lignes "
        f"{empty[:3]} : le corridor A→B était vide et la question a tout de "
        f"même été ajoutée"
    )


@pytest.mark.parametrize("nb_q", [40, 110])
def test_point_near_metric_honours_nb_q(df_osm, nb_q):
    """`nb_q` pilote bien le nombre de questions produites.

    Le générateur divisait `nb_q` par la *somme* `len(list_cat) + len(
    list_distance)` alors que sa boucle parcourt leur *produit* : il rendait
    240 questions pour nb_q=110. Corrigé par l'allocation sur les cellules
    (catégorie d'ancre × catégorie interrogée), qui alloue directement le
    budget au lieu de le diviser à l'aveugle — mesuré 8 / 40 / 108 pour
    nb_q 10 / 40 / 110. Ce test garde la propriété.
    """
    from src.template_question.geospatial.point_near_metric import (
        make_question_point_near_metric,
    )

    bench = make_question_point_near_metric(df_osm, nb_q=nb_q, seed=42)
    assert len(bench) == pytest.approx(nb_q, rel=0.25), (
        f"nb_q={nb_q} demandé, {len(bench)} questions produites "
        f"(facteur {len(bench) / nb_q:.2f})"
    )


@defect("point_near_cardinal_quota")
@pytest.mark.parametrize("nb_q", [10, 40])
def test_point_near_cardinal_honours_nb_q(df_osm, nb_q):
    """Même faute de quota que `point_near_metric`, en pire.

    La boucle parcourt catégories × 4 directions et divise par leur somme : le
    jeu produit est environ 4× la commande. Le facteur n'est pas constant — il
    sature quand le corpus s'épuise — donc on ne peut même pas le corriger
    après coup en tronquant.
    """
    from src.template_question.geospatial.point_near_cardinal import (
        make_question_point_near_cardinal,
    )

    bench = make_question_point_near_cardinal(df_osm, nb_q=nb_q, seed=42)
    assert len(bench) == pytest.approx(nb_q, rel=0.25), (
        f"nb_q={nb_q} demandé, {len(bench)} questions produites "
        f"(facteur {len(bench) / nb_q:.2f})"
    )


@defect("area_direction_quota")
def test_area_direction_honours_nb_q(df_osm, df_area):
    """Idem pour `area_direction`, dont la condition d'arrêt est inatteignable.

    `while nq != n_queries_per_stratum` avec une boucle interne de 4 directions :
    `nq` avance de 4 en 4 et enjambe le quota sans jamais l'égaler. Seul le
    plafond de 200 tentatives finit par arrêter la boucle.
    """
    from src.template_question.geospatial.area_direction import (
        make_question_area_direction,
    )

    nb_q = 40
    bench = make_question_area_direction(df_osm, df_area, nb_q=nb_q, seed=42)
    assert len(bench) == pytest.approx(nb_q, rel=0.25), (
        f"nb_q={nb_q} demandé, {len(bench)} questions produites"
    )


@defect("area_direction_quota")
def test_area_direction_volume_grows_with_nb_q(df_osm, df_area):
    """À défaut du quota exact, le volume devrait au moins être monotone.

    Demander plus de questions doit en produire au moins autant. Ici la
    condition d'arrêt dépend du plafond de tentatives et non du quota, donc le
    volume part dans tous les sens — mesuré : 802 questions pour nb_q=10 mais
    135 pour nb_q=40. C'est le symptôme le plus lisible du défaut : même en
    renonçant à la valeur exacte, le générateur n'est pas pilotable.
    """
    from src.template_question.geospatial.area_direction import (
        make_question_area_direction,
    )

    volumes = [
        len(make_question_area_direction(df_osm, df_area, nb_q=n, seed=42))
        for n in (10, 40)
    ]
    assert volumes[0] <= volumes[1], (
        f"nb_q=10 produit {volumes[0]} questions et nb_q=40 seulement "
        f"{volumes[1]} : le volume décroît quand la commande augmente"
    )


@defect("street_cross_sous_production")
@pytest.mark.parametrize("nb_q", [10, 40])
def test_street_cross_honours_nb_q(df_osm, df_streets, nb_q):
    """`street_cross` doit atteindre son quota comme les autres templates.

    Le défaut est ici l'inverse des précédents : le tirage exige deux rues
    sécantes *et* des POIs près du croisement, et abandonne la strate au lieu de
    retirer une autre paire. Mesuré : 7 questions pour nb_q=10, 25 pour 40.

    Les fixtures n'offrent qu'un seul vrai croisement, donc une part du déficit
    leur est imputable — d'où la tolérance large. Mais un template qui rend la
    main sous le quota sans le signaler déséquilibre le jeu final entre
    templates, en silence.
    """
    from src.template_question.geospatial.street_cross import (
        make_question_street_cross,
    )

    bench = make_question_street_cross(df_osm, df_streets, nb_q=nb_q, seed=42)
    assert len(bench) >= 0.75 * nb_q, (
        f"nb_q={nb_q} demandé, {len(bench)} questions produites "
        f"({len(bench) / nb_q:.0%} du quota)"
    )


# --------------------------------------------------------------------------- #
# entrées absurdes acceptées en silence
# --------------------------------------------------------------------------- #

@defect("area_direction_test_de_sous_chaine")
@pytest.mark.parametrize("bogus", ["north s", "h so", "outh wes", ""])
def test_direction_area_rejects_unknown_directions(df_osm, df_area, bogus):
    """Une direction inconnue doit lever, pas être interprétée au hasard.

    `direction in "north south"` teste une **sous-chaîne** : il répond vrai pour
    `"north s"` comme pour `""`. La fonction trie alors sur un axe choisi par
    accident au lieu de signaler l'erreur. Même faute ligne suivante, où
    `("south west")` est une chaîne entre parenthèses et non un tuple.
    """
    from src.template_question.geospatial.area_direction import direction_area

    area = df_area.geometry.iloc[0]
    with pytest.raises((KeyError, ValueError)):
        direction_area(df_osm, area, bogus)


# --------------------------------------------------------------------------- #
# classement dégénéré : la distance ne discrimine rien
# --------------------------------------------------------------------------- #

@defect("area_border_mesure_le_polygone_plein")
def test_border_area_separates_interior_from_boundary(df_osm, df_area):
    """`border_area` doit distinguer un POI du bord d'un POI du centre.

    Mesurée contre le polygone **plein**, la distance vaut 0 partout à
    l'intérieur : un POI au centre exact de la zone est donné aussi « en
    bordure » que celui collé au pourtour. Contre `area.boundary`, les deux se
    séparent — c'est le sens même de la question.
    """
    from src.template_question.geospatial.area_border import border_area

    area = df_area.geometry.iloc[0]
    results = border_area(area, "cafe", df_osm)
    inside = results[results["poi_id"].isin(
        df_osm["poi_id"][df_osm.geometry.within(area)]
    )]
    if len(inside) < 2:
        pytest.skip("pas assez de POIs intérieurs pour comparer")

    assert inside["dist"].nunique() > 1, (
        f"les {len(inside)} POIs intérieurs sont tous à distance "
        f"{inside['dist'].iloc[0]} : la bordure n'est pas mesurée"
    )


# --------------------------------------------------------------------------- #
# géométries écartées par erreur
# --------------------------------------------------------------------------- #

@defect("street_opposite_side_multilinestring_ignoree")
def test_opposite_side_handles_multilinestring_streets(df_osm):
    """Une rue en plusieurs tronçons doit rester tirable.

    Le filtre écrit `"MultiString"` (street_opposite_side.py:109), type de
    géométrie qui n'existe pas dans shapely : aucune MultiLineString ne le
    franchit. Sur un corpus qui n'en contient que, le tirage devient vide. Or
    `load_streets` produit bien des MultiLineString — les rues coupées par une
    place ou un carrefour — et le reste du module sait les traiter, puisque
    `side_of_street` et `cross_along` prennent explicitement `geoms[0]`.
    """
    from src.template_question.geospatial.street_opposite_side import (
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

@defect("street_cross_intersection_vide")
def test_touching_streets_only_returns_real_crossings():
    """`touching_streets` ne doit pas retenir deux rues qui ne se croisent pas.

    La tolérance de 1 m admet deux rues seulement *voisines*. Leur
    `intersection()` est alors vide, et `distance()` contre une géométrie vide
    vaut NaN : toute la vérité terrain de la question perd son ordre.
    """
    from src.template_question.geospatial.street_cross import touching_streets

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


@defect("street_identity_incoherente")
def test_street_modules_agree_on_street_identity(df_osm, df_streets):
    """Les deux modules « rue » doivent identifier une rue de la même façon.

    `street_along.py:73` tire dans `df_streets.index` ; `street_cross.py:83-84`
    tire une *valeur* de `id_street` et la passe à `.loc`, ce qui n'est correct
    que si l'index et la colonne coïncident. Sur un `df_streets` réindexé — ce
    que fait n'importe quel filtrage en amont — le second se trompe de rue en
    silence, ou lève.
    """
    from src.template_question.geospatial.street_cross import make_question_street_cross

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
# stratification : le helper que `point_between` est en train d'adopter
# --------------------------------------------------------------------------- #
#
# `src/template_question/ratio.py` n'a pas de test, alors que `point_between`
# s'appuie dessus pour répartir les questions entre catégories. Les cinq tests
# qui suivent fixent son contrat avant que les autres templates l'adoptent.

def test_allocate_distributes_exactly_n():
    """La somme des parts vaut exactement `n`, sans perte à l'arrondi.

    C'est toute la raison d'être de la méthode des plus forts restes : un
    `round()` naïf par catégorie perdrait ou inventerait des questions, et le
    benchmark ne ferait plus la taille commandée.
    """
    from src.template_question.ratio import allocate

    for n in (0, 1, 7, 10, 110, 1000):
        counts = allocate(n, {c: 1 for c in "abcde"})
        assert sum(counts.values()) == n, (
            f"allocate({n}, 5 catégories égales) totalise "
            f"{sum(counts.values())} au lieu de {n}"
        )


def test_allocate_respects_proportions():
    """Les parts suivent les poids demandés."""
    from src.template_question.ratio import allocate

    assert allocate(10, {"a": 0.1, "b": 0.3, "c": 0.6}) == {"a": 1, "b": 3, "c": 6}


def test_allocate_accepts_raw_weights():
    """Des poids bruts valent des proportions : ils sont normalisés.

    L'appelant ne doit donc pas avoir à normaliser lui-même : passer
    `{cat: 1 for cat in list_cat}` suffit à demander l'équirépartition, sans
    division préalable.
    """
    from src.template_question.ratio import allocate

    assert allocate(10, {"a": 1, "b": 3, "c": 6}) == allocate(
        10, {"a": 0.1, "b": 0.3, "c": 0.6}
    )


def test_allocate_spreads_the_remainder_over_the_largest_fractions():
    """Le reste va aux plus fortes parties fractionnaires, pas au hasard.

    10 questions sur 3 catégories égales : 3,33 chacune, donc 3 partout et une
    unité à répartir. La méthode la donne à la première par ordre de reste
    décroissant — déterministe, donc rejouable.
    """
    from src.template_question.ratio import allocate

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
    from src.template_question.ratio import allocate

    with pytest.raises(ZeroDivisionError):
        allocate(10, {c: 1 // 5 for c in "abcde"})


# --------------------------------------------------------------------------- #
# non-régression : ce qui a été réparé doit le rester
# --------------------------------------------------------------------------- #

def test_street_cross_module_has_no_import_side_effect():
    """Importer un module de template ne doit rien exécuter.

    Le bas de `street_cross.py` avait gardé l'appel du notebook
    (`bench4=make_question_crossstreet(df_osm, df_streets)`) : importer le
    module tentait de générer un benchmark avec des variables inexistantes, et
    aucun autre module ne pouvait s'en servir. Corrigé — ce test garde la
    propriété, parce que la faute revient à chaque template sorti d'un notebook.
    """
    import importlib

    importlib.import_module("src.template_question.geospatial.street_cross")


def test_every_template_module_imports_cleanly():
    """Les douze modules s'importent sans effet de bord, pas seulement un.

    Généralise le test précédent : c'est le paquet entier qui doit être
    importable, puisque `geospatial/__init__.py` les importe tous pour peupler le
    registre par décorateur.
    """
    import importlib

    from src.template_question.schema import TEMPLATE_REGISTRY

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
    import pathlib

    for name in ("area_inside", "area_outside", "area_border", "area_direction"):
        source = pathlib.Path(
            f"src/template_question/geospatial/{name}.py").read_text(encoding="utf-8")
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


def test_interior_point_is_at_zero_distance_from_its_polygon(df_area):
    """Rappel exécutable de la cause des classements dégénérés.

    C'est cette propriété de shapely — et non un bug — qui rend
    `inside_area`/`border_area` incapables d'ordonner quoi que ce soit tant
    qu'ils mesurent contre le polygone plein.
    """
    area = df_area.geometry.iloc[0]
    centre = area.centroid

    assert centre.distance(area) == 0.0
    assert centre.distance(area.boundary) > 0.0
    assert np.isclose(centre.distance(area.boundary), 200.0), (
        "le centre du Parc Carre (400x400 m) est à 200 m de son pourtour"
    )
