"""Fixtures d'un monde synthétique pour tester les générateurs de questions.

Les tests ne tapent ni Overpass ni le cache OSM : ils travaillent sur une grille
de POIs construite ici. Trois raisons à ce choix — le déterminisme (une vérité
terrain recalculable à la main), la vitesse, et l'absence de réseau.

Le contrat de colonnes est celui de `src/data_collection/osm_loaders.py`, qui est
la direction voulue du projet :

* `df_osm`     → `poi_id`, `poi_name`, `category`, `x`, `y`, `geometry`
* `df_area`    → `area_name`, `geometry`
* `df_streets` → `id_street`, `street_name`, `geometry`

Tout est en **EPSG:2154** (Lambert-93), donc les distances sont euclidiennes et
en mètres, comme l'attendent les requêtes DuckDB des templates.
"""

import importlib

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
from shapely.geometry import LineString, MultiLineString, Point, box  # noqa: E402

#: Graine unique de toute la suite. Les générateurs la prennent en argument ;
#: une valeur fixe rend chaque test reproductible d'une exécution à l'autre.
SEED = 42

# Origine plausible pour Paris en Lambert-93. Les valeurs exactes importent peu,
# mais des coordonnées réalistes font que les templates tournent dans le même
# ordre de grandeur qu'en production.
OX, OY = 650000.0, 6860000.0

CATEGORIES = ("cafe", "restaurant", "bar", "hotel", "park")

GRID_SIDE = 7          # 7x7 = 49 POIs
GRID_STEP = 100.0      # mètres entre deux nœuds
JITTER = 7.0           # amplitude du décalage, en mètres

#: Ordonnée de la rue horizontale, et abscisse de la verticale : elles se
#: croisent donc en (OX + 250, OY + 250).
STREET_H_Y = OY + 250.0
STREET_V_X = OX + 250.0

#: Décalage des POIs « à cheval » de part et d'autre de la rue horizontale.
STRADDLE_OFFSET = 20.0


def build_df_osm():
    """Construit le corpus de POIs synthétique.

    Deux populations :

    1. une grille 7×7 au pas de 100 m, catégorie cyclique, **avec un jitter
       déterministe de ±7 m**. Le jitter n'est pas cosmétique : sur une grille
       exacte, un même point de référence est à distance strictement égale de
       plusieurs POIs, et l'ordre de ces ex æquo diffère entre DuckDB et un
       oracle numpy. Les tests d'ordre seraient alors faux sans qu'il y ait de
       bug. Le décalage rend toutes les distances distinctes ;
    2. 12 POIs « à cheval » sur la rue horizontale, à ±20 m de part et d'autre.
       Sans eux, `opposite_side` ne renvoie rien : il exige une distance à la rue
       inférieure à 40 m, alors que la grille est au pas de 100 m.

    Returns:
        GeoDataFrame: `poi_id` (0..n-1), `poi_name`, `category`, `geometry`,
            `x`, `y`, en EPSG:2154.
    """
    rng = np.random.default_rng(0)
    rows = []
    poi_id = 0

    for iy in range(GRID_SIDE):
        for ix in range(GRID_SIDE):
            category = CATEGORIES[(ix + iy) % len(CATEGORIES)]
            dx, dy = rng.uniform(-JITTER, JITTER, size=2)
            rows.append({
                "poi_id": poi_id,
                "poi_name": f"poi{poi_id}_{category}",
                "category": category,
                "geometry": Point(OX + GRID_STEP * ix + dx,
                                  OY + GRID_STEP * iy + dy),
            })
            poi_id += 1

    for k, ix in enumerate(range(6)):
        for side, dy in (("nord", STRADDLE_OFFSET), ("sud", -STRADDLE_OFFSET)):
            category = CATEGORIES[k % len(CATEGORIES)]
            rows.append({
                "poi_id": poi_id,
                "poi_name": f"cheval{poi_id}_{side}_{category}",
                "category": category,
                "geometry": Point(OX + GRID_STEP * ix + 3.0, STREET_H_Y + dy),
            })
            poi_id += 1

    df = gpd.GeoDataFrame(rows, crs=2154)
    df["x"] = df.geometry.x
    df["y"] = df.geometry.y
    return df


def build_df_area():
    """Construit les zones synthétiques.

    Trois polygones, dont un volontairement sous le `min_size=1000` par défaut
    des générateurs, pour que le filtre de surface soit réellement éprouvé.

    Returns:
        GeoDataFrame: `area_name`, `geometry`, en EPSG:2154.
    """
    return gpd.GeoDataFrame(
        [
            # couvre le quadrant bas-gauche de la grille : 160 000 m²
            {"area_name": "Parc Carre",
             "geometry": box(OX - 50, OY - 50, OX + 350, OY + 350)},
            # quadrant haut-droit : 62 500 m²
            {"area_name": "Square Petit",
             "geometry": box(OX + 400, OY + 400, OX + 650, OY + 650)},
            # 25 m², sous min_size : doit être écartée
            {"area_name": "Micro Zone",
             "geometry": box(OX + 10, OY + 10, OX + 15, OY + 15)},
        ],
        crs=2154,
    )


def build_df_streets():
    """Construit les rues synthétiques.

    `rue Horizontale` et `rue Verticale` **se croisent** en
    (OX + 250, OY + 250) : c'est ce qui rend `street_cross` testable. Toutes
    deux mesurent 700 m, donc passent le `min_length=500` de `street_along`.
    `rue Coupee` est une MultiLineString, pour éprouver le filtre `geom_type`.

    `id_street` est délibérément **égal au label d'index**, parce que
    `street_cross.py:82` fait `df_streets.loc[<valeur de id_street>]` : sans
    cette égalité, ce template ne peut pas fonctionner du tout et l'on ne
    testerait que son plantage.

    Returns:
        GeoDataFrame: `id_street`, `street_name`, `geometry`, en EPSG:2154.
    """
    return gpd.GeoDataFrame(
        [
            {"id_street": 0, "street_name": "rue Horizontale",
             "geometry": LineString([(OX - 50, STREET_H_Y), (OX + 650, STREET_H_Y)])},
            {"id_street": 1, "street_name": "rue Verticale",
             "geometry": LineString([(STREET_V_X, OY - 50), (STREET_V_X, OY + 650)])},
            {"id_street": 2, "street_name": "rue Coupee",
             "geometry": MultiLineString([
                 [(OX, OY + 600), (OX + 200, OY + 600)],
                 [(OX + 260, OY + 600), (OX + 500, OY + 600)],
             ])},
        ],
        crs=2154,
    )


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def df_osm():
    """Corpus de POIs synthétique, partagé par toute la session."""
    return build_df_osm()


@pytest.fixture(scope="session")
def df_area():
    """Zones synthétiques, partagées par toute la session."""
    return build_df_area()


@pytest.fixture(scope="session")
def df_streets():
    """Rues synthétiques, partagées par toute la session."""
    return build_df_streets()


@pytest.fixture(scope="session")
def nb_q():
    """Nombre de questions demandé aux générateurs.

    10 pour 5 catégories, soit 2 questions par strate : assez pour que les
    invariants aient de la matière, assez peu pour que la suite reste rapide —
    les templates DuckDB ouvrent une connexion par appel.
    """
    return 10


@pytest.fixture(scope="session")
def fixture_table(df_osm, df_area, df_streets):
    """Table de résolution des fixtures déclarées dans `TEMPLATE_REGISTRY`.

    Les templates déclarent leurs entrées par nom (`("df_osm", "df_area")`) ;
    cette table les traduit en objets, pour appeler un générateur sans savoir
    lequel c'est.
    """
    return {"df_osm": df_osm, "df_area": df_area, "df_streets": df_streets}


@pytest.fixture(scope="session")
def run_template(fixture_table, nb_q):
    """Exécute un template et met son résultat en cache pour la session.

    Les générateurs ouvrent une connexion DuckDB par question ; les rejouer pour
    chacun des huit tests de contrat coûterait cher. Le cache mémorise aussi
    bien le succès que l'exception, pour qu'un template cassé lève la *même*
    erreur à chaque test sans être relancé.

    Returns:
        callable: `run_template(template) -> DataFrame`, qui relève l'exception
            d'origine si le générateur échoue.
    """
    cache = {}

    def _run(template):
        if template.name not in cache:
            try:
                module = importlib.import_module(template.module)
                generator = getattr(module, template.generator)
                args = [fixture_table[name] for name in template.fixtures]
                cache[template.name] = (None, generator(*args, nb_q=nb_q, seed=SEED))
            except Exception as exc:          # noqa: BLE001 — on veut tout capturer
                cache[template.name] = (exc, None)
        error, bench = cache[template.name]
        if error is not None:
            raise error
        return bench

    return _run


# --------------------------------------------------------------------------- #
# helpers d'oracle
# --------------------------------------------------------------------------- #

def oracle_pois(df_osm, category):
    """POIs d'une catégorie, avec leurs coordonnées en tableau.

    Sert aux recalculs indépendants : les tests sémantiques refont la vérité
    terrain en numpy/shapely pur, sans repasser par DuckDB.

    Args:
        df_osm (GeoDataFrame): Le corpus.
        category (str): Catégorie à extraire.

    Returns:
        (GeoDataFrame, ndarray): Le sous-ensemble, et ses coordonnées `(n, 2)`.
    """
    sub = df_osm[df_osm["category"] == category]
    return sub, np.column_stack([sub["x"].to_numpy(), sub["y"].to_numpy()])


def azimuth_deg(ax, ay, bx, by):
    """Azimut de A vers B, en degrés horaires depuis le nord, dans [0, 360).

    Reproduit la convention des templates (`atan2(dx, dy)`, inversé par rapport
    à la convention mathématique), pour que l'oracle soit comparable.
    """
    return (np.degrees(np.arctan2(bx - ax, by - ay)) + 360.0) % 360.0


def angular_gap_deg(azimuth, center):
    """Écart angulaire absolu entre deux azimuts, ramené dans [0, 180]."""
    return np.abs(((np.asarray(azimuth) - center + 540.0) % 360.0) - 180.0)
