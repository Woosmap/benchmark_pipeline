"""Démonstration ciblée des défauts qui ne se voient pas au niveau du contrat.

`test_template_contract.py` et `test_template_semantics.py` couvrent déjà ce qui
casse un benchmark entier. Restent des défauts plus fins, qui n'empêchent pas la
sortie d'être bien formée mais faussent le volume produit, ou acceptent en
silence une entrée absurde. Chacun a ici son test dédié, au plus près de la
fonction fautive, pour que le jour de la correction on sache exactement quoi
relancer.

Tous sont `xfail(strict=True)` : ils passent au vert le jour où le défaut est
corrigé, ce que pytest signale alors comme un XPASS à traiter.
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
        blocked_by: Nom du template dont la panne empêche, en plus, d'atteindre
            ce défaut. Corriger `key` seul ne suffira donc pas à faire passer le
            test — le motif le dit explicitement, pour qu'un XPASS manquant ne
            soit pas pris pour un oubli.
    """
    reason = SEMANTIC_DEFECTS[key]
    if blocked_by is not None:
        reason = f"{reason}\n  — inatteignable tant que : {BROKEN_TEMPLATES[blocked_by]}"
    return pytest.mark.xfail(strict=True, reason=reason)


# --------------------------------------------------------------------------- #
# volumétrie : le nombre de questions produites ne correspond pas à nb_q
# --------------------------------------------------------------------------- #

@defect("point_near_metric_quota")
@pytest.mark.parametrize("nb_q", [40, 110])
def test_point_near_metric_honours_nb_q(df_osm, nb_q):
    """`nb_q` doit piloter le nombre de questions produites.

    Le générateur boucle sur le *produit* catégories × distances, mais divise
    par leur *somme* : le quota par strate est donc trop grand et le jeu final
    dépasse largement la commande. Un benchmark dont on ne maîtrise pas la
    taille se compare mal d'une exécution à l'autre.
    """
    from src.template_question.type_A.point_near_metric import (
        make_question_point_near_metric,
    )

    bench = make_question_point_near_metric(df_osm, nb_q=nb_q, seed=42)
    assert len(bench) == pytest.approx(nb_q, rel=0.25), (
        f"nb_q={nb_q} demandé, {len(bench)} questions produites "
        f"(facteur {len(bench) / nb_q:.2f})"
    )


@defect("area_direction_quota", blocked_by="area_direction")
def test_area_direction_honours_nb_q(df_osm, df_area):
    """Idem pour `area_direction`, dont la condition d'arrêt est inatteignable.

    `while nq != n_queries_per_stratum` avec une boucle interne de 4 directions :
    `nq` avance de 4 en 4 et enjambe le quota sans jamais l'égaler. Seul le
    plafond de 200 tentatives finit par arrêter la boucle.
    """
    from src.template_question.type_A.area_direction import (
        make_question_area_direction,
    )

    nb_q = 40
    bench = make_question_area_direction(df_osm, df_area, nb_q=nb_q, seed=42)
    assert len(bench) == pytest.approx(nb_q, rel=0.25), (
        f"nb_q={nb_q} demandé, {len(bench)} questions produites"
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
    from src.template_question.type_A.area_direction import direction_area

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
    from src.template_question.type_A.area_border import border_area

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

@defect("street_opposite_side_multilinestring_ignoree", blocked_by="street_opposite_side")
def test_opposite_side_handles_multilinestring_streets(df_osm):
    """Une rue en plusieurs tronçons doit rester tirable.

    Le filtre écrit `"MultiString"`, type de géométrie qui n'existe pas dans
    shapely : aucune MultiLineString ne le franchit. Sur un corpus qui n'en
    contient que, le tirage devient vide. Or `load_streets` produit bien des
    MultiLineString — les rues coupées par une place ou un carrefour.
    """
    from src.template_question.type_A.street_opposite_side import (
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

@defect("street_cross_intersection_vide", blocked_by="street_cross")
def test_touching_streets_only_returns_real_crossings():
    """`touching_streets` ne doit pas retenir deux rues qui ne se croisent pas.

    La tolérance de 1 m admet deux rues seulement *voisines*. Leur
    `intersection()` est alors vide, et `distance()` contre une géométrie vide
    vaut NaN : toute la vérité terrain de la question perd son ordre.
    """
    from src.template_question.type_A.street_cross import touching_streets

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


@defect("street_identity_incoherente", blocked_by="street_cross")
def test_street_modules_agree_on_street_identity(df_osm, df_streets):
    """Les deux modules « rue » doivent identifier une rue de la même façon.

    `street_along.py:71` tire dans `df_streets.index` ; `street_cross.py:81-82`
    tire une *valeur* de `id_street` et la passe à `.loc`, ce qui n'est correct
    que si l'index et la colonne coïncident. Sur un `df_streets` réindexé — ce
    que fait n'importe quel filtrage en amont — le second se trompe de rue en
    silence, ou lève.
    """
    from src.template_question.type_A.street_cross import make_question_street_cross

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
# le défaut qui empêche l'import
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(strict=True, reason=BROKEN_TEMPLATES["street_cross"])
def test_street_cross_module_has_no_import_side_effect():
    """Importer un module de template ne doit rien exécuter.

    Le bas de `street_cross.py` a gardé l'appel du notebook. Importer le module
    tente donc de générer un benchmark avec des variables qui n'existent pas, et
    aucun autre module ne peut s'en servir.
    """
    import importlib

    importlib.import_module("src.template_question.type_A.street_cross")


# --------------------------------------------------------------------------- #
# non-régression : ce qui marche doit continuer de marcher
# --------------------------------------------------------------------------- #

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
