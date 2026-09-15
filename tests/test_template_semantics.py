"""La réponse dit-elle vraiment ce que la question demande ?

`test_template_contract.py` vérifie la *forme* de la réponse. Ici on vérifie son
*contenu* : chaque vérité terrain est recalculée indépendamment, en numpy et
shapely purs, sans repasser par DuckDB ni par les helpers du template. Comparer
les deux attrape ce qu'aucun contrôle de forme ne voit — une requête SQL bien
formée qui ne calcule pas la relation annoncée par l'énoncé.

Les templates aujourd'hui cassés portent un `xfail(strict=True)` : l'oracle est
écrit quand même, et se mettra à vérifier pour de bon dès que le générateur
tournera.
"""

import numpy as np
import pytest

from src.template_question.schema import TEMPLATES_BY_NAME, _as_list
from tests.conftest import angular_gap_deg, azimuth_deg
from tests.known_defects import BROKEN_TEMPLATES, SEMANTIC_DEFECTS, merge_reasons

#: Le `k` par défaut des générateurs. Au-delà, la réponse est tronquée et un
#: contrôle de complétude n'a plus de sens.
DEFAULT_K = 100

#: Les distances comparées sont des mètres Lambert-93 ; le millimètre suffit
#: largement à distinguer un vrai écart d'un arrondi de représentation.
TOL_M = 1e-6


def oracle_for(template_name, *defect_tables):
    """Décore un test d'oracle, xfail si le template est répertorié cassé."""
    reasons = merge_reasons(*defect_tables)

    def decorate(func):
        if template_name in reasons:
            return pytest.mark.xfail(
                strict=True, reason=reasons[template_name]
            )(func)
        return func

    return decorate


def bench_of(name, run_template):
    """Raccourci : le benchmark d'un template désigné par son nom."""
    return run_template(TEMPLATES_BY_NAME[name])


def category_subset(df_osm, category, exclude=()):
    """POIs d'une catégorie, moins les `poi_id` exclus."""
    sub = df_osm[df_osm["category"] == category]
    if len(exclude):
        sub = sub[~sub["poi_id"].isin(exclude)]
    return sub


# --------------------------------------------------------------------------- #
# proximité
# --------------------------------------------------------------------------- #

@oracle_for("point_near", BROKEN_TEMPLATES)
def test_point_near_matches_bruteforce_knn(run_template, df_osm):
    """« X près de Y » classe bien les X par distance croissante à Y.

    Oracle : tri par `np.hypot` sur toute la catégorie. Le jitter des fixtures
    garantit qu'aucune paire de distances n'est ex æquo, donc l'ordre attendu
    est unique et la comparaison peut porter sur les identifiants eux-mêmes.
    """
    bench = bench_of("point_near", run_template)

    for i, row in bench.iterrows():
        exclude = [row["anchor_index"]] if row["same_cat"] else []
        sub = category_subset(df_osm, row["category_query"], exclude)
        dist = np.hypot(sub["x"] - row["anchor_x"], sub["y"] - row["anchor_y"])
        order = np.argsort(dist.to_numpy(), kind="stable")

        expected_ids = sub["poi_id"].to_numpy()[order][:DEFAULT_K].tolist()
        expected_dist = np.sort(dist.to_numpy())[:DEFAULT_K]

        assert _as_list(row["results_poi_id"]) == expected_ids, (
            f"ligne {i} ({row['query']!r}) : classement différent de l'oracle\n"
            f"  attendu {expected_ids[:6]}\n"
            f"  obtenu  {_as_list(row['results_poi_id'])[:6]}"
        )
        assert np.allclose(_as_list(row["results_poi_dist"]), expected_dist,
                           atol=TOL_M), (
            f"ligne {i} ({row['query']!r}) : distances différentes de l'oracle"
        )


@oracle_for("point_near_metric", BROKEN_TEMPLATES)
def test_point_near_metric_respects_its_radius(run_template, df_osm):
    """« X à moins de d mètres de Y » : rayon respecté, et exhaustif.

    Deux erreurs symétriques sont possibles et toutes deux graves pour un
    benchmark : livrer un POI hors rayon (faux positif dans la vérité terrain)
    ou en oublier un qui y est (faux négatif, qui pénalise un modèle correct).
    L'oracle contrôle les deux sens.
    """
    bench = bench_of("point_near_metric", run_template)

    for i, row in bench.iterrows():
        radius = row["distance"]
        ids = _as_list(row["results_poi_id"])
        dists = _as_list(row["results_poi_dist"])

        over = [(p, d) for p, d in zip(ids, dists) if d >= radius]
        assert not over, (
            f"ligne {i} ({row['query']!r}) : {len(over)} POIs au-delà du rayon "
            f"de {radius} m, ex. {over[:3]}"
        )

        exclude = [row["anchor_index"]] if row["same_cat"] else []
        sub = category_subset(df_osm, row["category_query"], exclude)
        dist = np.hypot(sub["x"] - row["anchor_x"], sub["y"] - row["anchor_y"])
        expected = set(sub["poi_id"][dist < radius])

        if len(expected) <= DEFAULT_K:
            forgotten = expected - set(ids)
            assert not forgotten, (
                f"ligne {i} ({row['query']!r}) : {len(forgotten)} POIs dans le "
                f"rayon absents de la réponse, ex. {sorted(forgotten)[:3]}"
            )


@oracle_for("point_near_cardinal", BROKEN_TEMPLATES)
def test_point_near_cardinal_stays_in_its_sector(run_template, df_osm):
    """« X au nord de Y » ne retient que le secteur angulaire annoncé.

    L'azimut suit la convention des templates — horaire depuis le nord, d'où
    `atan2(dx, dy)`. Une inversion des arguments passerait tous les contrôles de
    forme tout en plaçant « nord » à l'est.
    """
    bench = bench_of("point_near_cardinal", run_template)
    from src.template_question.type_A.point_near_cardinal import CARDINAL_AZ

    half_width = 70.0            # défaut du générateur
    geo = df_osm.set_index("poi_id")

    for i, row in bench.iterrows():
        ids = _as_list(row["results_poi_id"])
        if not ids:
            continue
        center = CARDINAL_AZ[row["direction"]]
        pts = geo.loc[ids]
        az = azimuth_deg(row["anchor_x"], row["anchor_y"],
                         pts["x"].to_numpy(), pts["y"].to_numpy())
        gap = angular_gap_deg(az, center)

        outside = [(p, round(float(a), 1)) for p, a, g in zip(ids, az, gap)
                   if g > half_width + 1e-9]
        assert not outside, (
            f"ligne {i} ({row['query']!r}) : {len(outside)} POIs hors du "
            f"secteur {row['direction']} (centre {center}°, ±{half_width}°), "
            f"ex. (poi_id, azimut) {outside[:3]}"
        )


@oracle_for("point_towards", BROKEN_TEMPLATES)
def test_point_towards_follows_the_ab_bearing(run_template, df_osm):
    """« X près de A vers B » retient le cône orienté de A vers B."""
    bench = bench_of("point_towards", run_template)

    half_width = 70.0
    geo = df_osm.set_index("poi_id")

    for i, row in bench.iterrows():
        ids = _as_list(row["results_poi_id"])
        if not ids:
            continue
        center = azimuth_deg(row["anchor_x"], row["anchor_y"],
                             row["point_b_x"], row["point_b_y"])
        pts = geo.loc[ids]
        az = azimuth_deg(row["anchor_x"], row["anchor_y"],
                         pts["x"].to_numpy(), pts["y"].to_numpy())

        outside = [p for p, g in zip(ids, angular_gap_deg(az, center))
                   if g > half_width + 1e-9]
        assert not outside, (
            f"ligne {i} ({row['query']!r}) : {len(outside)} POIs hors du cône "
            f"A→B (cap {center:.1f}°, ±{half_width}°), ex. {outside[:3]}"
        )


@oracle_for("point_between", BROKEN_TEMPLATES)
def test_point_between_stays_in_the_ab_corridor(run_template, df_osm):
    """« X entre A et B » : projection dans le segment, écart sous le corridor.

    Le SQL projette sur AB (`along`) et mesure l'écart latéral (`cross_m`). Un
    POI derrière A ou au-delà de B a un `along` hors de `[0, |AB|]` et n'est pas
    « entre » les deux, quelle que soit sa distance.
    """
    bench = bench_of("point_between", run_template)

    corridor = 200.0             # défaut du générateur
    geo = df_osm.set_index("poi_id")

    for i, row in bench.iterrows():
        ids = _as_list(row["results_poi_id"])
        if not ids:
            continue
        ax, ay = row["anchor_x"], row["anchor_y"]
        abx, aby = row["point_b_x"] - ax, row["point_b_y"] - ay
        ab_len = np.hypot(abx, aby)

        pts = geo.loc[ids]
        dx = pts["x"].to_numpy() - ax
        dy = pts["y"].to_numpy() - ay
        along = (dx * abx + dy * aby) / ab_len
        cross = np.abs(dx * aby - dy * abx) / ab_len

        bad_along = [(p, round(float(a), 1)) for p, a in zip(ids, along)
                     if a < -1e-6 or a > ab_len + 1e-6]
        assert not bad_along, (
            f"ligne {i} ({row['query']!r}) : {len(bad_along)} POIs hors du "
            f"segment A→B (|AB|={ab_len:.1f} m), ex. (poi_id, along) "
            f"{bad_along[:3]}"
        )
        bad_cross = [(p, round(float(c), 1)) for p, c in zip(ids, cross)
                     if c > corridor + 1e-6]
        assert not bad_cross, (
            f"ligne {i} ({row['query']!r}) : {len(bad_cross)} POIs au-delà du "
            f"corridor de {corridor} m, ex. (poi_id, écart) {bad_cross[:3]}"
        )


# --------------------------------------------------------------------------- #
# zones
# --------------------------------------------------------------------------- #

@oracle_for("area_inside", BROKEN_TEMPLATES)
def test_area_inside_returns_exactly_the_pois_within(run_template, df_osm):
    """« X dans Z » : tous dedans, et aucun oublié."""
    bench = bench_of("area_inside", run_template)
    geo = df_osm.set_index("poi_id")

    for i, row in bench.iterrows():
        area = row["area_geometry"]
        ids = _as_list(row["results_poi_id"])

        outside = [p for p in ids if not geo.loc[p, "geometry"].within(area)]
        assert not outside, (
            f"ligne {i} ({row['query']!r}) : {len(outside)} POIs hors de la "
            f"zone, ex. {outside[:3]}"
        )

        sub = category_subset(df_osm, row["category_query"])
        expected = set(sub["poi_id"][sub.geometry.within(area)])
        if len(expected) <= DEFAULT_K:
            assert expected == set(ids), (
                f"ligne {i} ({row['query']!r}) : oubliés "
                f"{sorted(expected - set(ids))[:3]}"
            )


@oracle_for("area_inside", BROKEN_TEMPLATES,
            {"area_inside": SEMANTIC_DEFECTS["area_inside_dist_toujours_nulle"]})
def test_area_inside_ranking_is_meaningful(run_template):
    """Le classement « dans la zone » doit discriminer les POIs.

    `distance(point, polygone)` vaut 0 pour tout point intérieur : la colonne de
    tri est constante et `sort_values` laisse l'ordre d'insertion. Le classement
    livré comme vérité terrain est alors arbitraire — deux modèles qui ordonnent
    différemment les mêmes bons POIs sont notés différemment sans raison.
    """
    bench = bench_of("area_inside", run_template)

    degenerate = [
        (i, row["query"], len(_as_list(row["results_poi_dist"])))
        for i, row in bench.iterrows()
        if len(_as_list(row["results_poi_dist"])) > 1
        and len(set(_as_list(row["results_poi_dist"]))) == 1
    ]
    assert not degenerate, (
        f"{len(degenerate)}/{len(bench)} questions ont une distance constante "
        f"sur toute la réponse : le classement n'ordonne rien. "
        f"Ex. ligne {degenerate[0][0]} ({degenerate[0][1]!r}), "
        f"{degenerate[0][2]} POIs à distance identique"
    )


@oracle_for("area_outside", BROKEN_TEMPLATES)
def test_area_outside_excludes_the_area(run_template, df_osm):
    """« X hors de Z » ne retient aucun POI intérieur à Z."""
    bench = bench_of("area_outside", run_template)
    geo = df_osm.set_index("poi_id")

    for i, row in bench.iterrows():
        area = row["area_geometry"]
        inside = [p for p in _as_list(row["results_poi_id"])
                  if geo.loc[p, "geometry"].within(area)]
        assert not inside, (
            f"ligne {i} ({row['query']!r}) : {len(inside)} POIs *dans* la zone "
            f"alors que la question demande l'extérieur, ex. {inside[:3]}"
        )


@oracle_for("area_border", BROKEN_TEMPLATES,
            {"area_border": SEMANTIC_DEFECTS["area_border_mesure_le_polygone_plein"]})
def test_area_border_measures_distance_to_the_boundary(run_template, df_osm):
    """« X en bordure de Z » se mesure au *pourtour*, pas au polygone plein.

    Contre le polygone plein, tout POI intérieur est à distance 0 et occupe la
    tête du classement, alors que la question porte précisément sur le bord.
    Seul `area.boundary` a une distance nulle exactement sur le pourtour.
    """
    bench = bench_of("area_border", run_template)
    geo = df_osm.set_index("poi_id")

    for i, row in bench.iterrows():
        boundary = row["area_geometry"].boundary
        ids = _as_list(row["results_poi_id"])
        published = _as_list(row["results_poi_dist"])
        expected = [geo.loc[p, "geometry"].distance(boundary) for p in ids]

        assert np.allclose(published, expected, atol=1e-6), (
            f"ligne {i} ({row['query']!r}) : results_poi_dist n'est pas la "
            f"distance au bord.\n  publié  {[round(d, 1) for d in published[:5]]}"
            f"\n  attendu {[round(d, 1) for d in expected[:5]]}"
        )


@oracle_for("area_direction", BROKEN_TEMPLATES)
def test_area_direction_orders_along_the_right_axis(run_template, df_osm):
    """« X au nord de Z » : dans Z, et ordonné sur le bon axe et le bon sens.

    Nord et sud trient sur l'ordonnée, est et ouest sur l'abscisse ; nord et est
    décroissants, sud et ouest croissants. Une erreur d'axe ou de signe produit
    un classement plausible mais faux, que seule cette monotonie détecte.
    """
    bench = bench_of("area_direction", run_template)
    geo = df_osm.set_index("poi_id")
    axis_of = {"north": "y", "south": "y", "east": "x", "west": "x"}
    descending = {"north", "east"}

    for i, row in bench.iterrows():
        ids = _as_list(row["results_poi_id"])
        if len(ids) < 2:
            continue
        area = row["area_geometry"]

        outside = [p for p in ids if not geo.loc[p, "geometry"].within(area)]
        assert not outside, (
            f"ligne {i} ({row['query']!r}) : {len(outside)} POIs hors de la "
            f"zone, ex. {outside[:3]}"
        )

        axis = axis_of[row["direction"]]
        coords = [geo.loc[p, axis] for p in ids]
        ordered = (sorted(coords, reverse=True) if row["direction"] in descending
                   else sorted(coords))
        assert coords == pytest.approx(ordered), (
            f"ligne {i} ({row['query']!r}) : classement non monotone sur "
            f"l'axe {axis} pour la direction {row['direction']}\n"
            f"  obtenu {[round(c, 1) for c in coords[:6]]}"
        )


# --------------------------------------------------------------------------- #
# rues
# --------------------------------------------------------------------------- #

@oracle_for("street_along", BROKEN_TEMPLATES)
def test_street_along_measures_distance_to_the_street(run_template, df_osm):
    """« X le long de la rue R » classe par distance à la géométrie de R."""
    bench = bench_of("street_along", run_template)
    geo = df_osm.set_index("poi_id")

    for i, row in bench.iterrows():
        street = row["street_geometry"]
        ids = _as_list(row["results_poi_id"])
        expected = [geo.loc[p, "geometry"].distance(street) for p in ids]

        assert np.allclose(_as_list(row["results_poi_dist"]), expected,
                           atol=TOL_M), (
            f"ligne {i} ({row['query']!r}) : results_poi_dist n'est pas la "
            f"distance à la rue"
        )


@oracle_for("street_cross", BROKEN_TEMPLATES,
            {"street_cross": SEMANTIC_DEFECTS["street_cross_intersection_vide"]})
def test_street_cross_measures_distance_to_the_junction(run_template, df_osm):
    """« X au croisement de A et B » : les rues se croisent, et on mesure là.

    Si l'intersection est vide — ce que `touching_streets(tol=1.0)` autorise —
    `distance()` renvoie NaN et la vérité terrain n'est plus ordonnée du tout.
    """
    bench = bench_of("street_cross", run_template)
    geo = df_osm.set_index("poi_id")

    for i, row in bench.iterrows():
        junction = row["street_a_geometry"].intersection(row["street_b_geometry"])
        assert not junction.is_empty, (
            f"ligne {i} ({row['query']!r}) : les deux rues ne se croisent pas, "
            f"le « croisement » de l'énoncé n'existe pas"
        )

        ids = _as_list(row["results_poi_id"])
        expected = [geo.loc[p, "geometry"].distance(junction) for p in ids]
        assert np.allclose(_as_list(row["results_poi_dist"]), expected,
                           atol=TOL_M), (
            f"ligne {i} ({row['query']!r}) : results_poi_dist n'est pas la "
            f"distance au point de croisement"
        )


@oracle_for("street_opposite_side", BROKEN_TEMPLATES)
def test_street_opposite_side_is_really_on_the_other_side(run_template, df_osm,
                                                          df_streets):
    """« X de l'autre côté de R par rapport à Y » : côté opposé, et à portée.

    Réutilise `side_of_street` du template lui-même pour le signe : réécrire le
    produit vectoriel ici ne testerait que ma propre réimplémentation. En
    revanche les deux seuils (40 m à la rue, 80 m le long) sont recalculés.
    """
    from src.template_question.type_A.street_opposite_side import side_of_street

    bench = bench_of("street_opposite_side", run_template)
    geo = df_osm.set_index("poi_id")
    max_along, max_cross = 80.0, 40.0

    for i, row in bench.iterrows():
        street = row["street_geometry"]
        line = street.geoms[0] if street.geom_type == "MultiLineString" else street
        side_y = side_of_street(line, row["poi_y_geometry"])
        along_y = line.project(row["poi_y_geometry"])

        for poi_id in _as_list(row["results_poi_id"]):
            point = geo.loc[poi_id, "geometry"]
            assert side_of_street(line, point) != side_y, (
                f"ligne {i} ({row['query']!r}) : poi_id={poi_id} est du *même* "
                f"côté de la rue que {row['poi_y_name']!r}"
            )
            assert point.distance(line) <= max_cross + 1e-6, (
                f"ligne {i} : poi_id={poi_id} est à "
                f"{point.distance(line):.1f} m de la rue (max {max_cross})"
            )
            gap = abs(line.project(point) - along_y)
            assert gap <= max_along + 1e-6, (
                f"ligne {i} : poi_id={poi_id} est décalé de {gap:.1f} m le long "
                f"de la rue par rapport à {row['poi_y_name']!r} "
                f"(max {max_along})"
            )
