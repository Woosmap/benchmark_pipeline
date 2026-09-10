"""Insère des docstrings dans question_typeA.ipynb sans toucher au code.

Repère chaque `def`, saute la signature (multi-lignes comprises), remplace la
docstring existante s'il y en a une, insère la nouvelle. Aucune autre ligne
n'est touchée.
"""
import json
import re
import sys

NB = "question_typeA.ipynb"

D = {}

# ---------------------------------------------------------------- cellule 4
D["near_sql"] = """
Retourne les `k` POIs d'une catégorie les plus proches d'un point.

Exécute une recherche par distance croissante dans DuckDB avec l'extension
spatiale. Les coordonnées et les géométries sont en Lambert-93 (EPSG:2154),
donc les distances sont euclidiennes et exprimées en mètres.

Args:
    df (GeoDataFrame): POIs candidats, avec `poi_id`, `poi_name`, `category`
        et une colonne `geometry` en EPSG:2154.
    x (float): Abscisse Lambert-93 du point de référence.
    y (float): Ordonnée Lambert-93 du point de référence.
    cat (str): Catégorie de POI recherchée.
    k (int): Nombre maximal de résultats retournés.

Returns:
    DataFrame: Colonnes `poi_id`, `poi_name`, `dist` (mètres), triées par
        distance croissante.

TODO: réutiliser une connexion DuckDB unique au lieu d'en ouvrir une et de
    rejouer `INSTALL spatial` à chaque appel.
"""

D["make_question_nearsql"] = """
Génère les questions de proximité simple « X près de Y ».

Pour chaque catégorie d'ancre, tire un quota d'ancres puis, pour chacune, une
catégorie cible au hasard. La vérité terrain est le classement des POIs de la
catégorie cible par distance à l'ancre. L'ancre est retirée de ses propres
résultats lorsque les deux catégories coïncident.

Args:
    df_osm (GeoDataFrame): POIs servant à la fois d'ancres et de cibles.
    nb_q (int): Nombre total de questions visé, stratifié par catégorie d'ancre.
    seed (int): Graine du générateur aléatoire.

Returns:
    DataFrame: Une ligne par question. Contexte `query`, `anchor_index`,
        `anchor_name`, `anchor_category`, `anchor_x`, `anchor_y`,
        `category_query`, `same_cat`, `function` ; vérité terrain
        `results_poi_id`, `results_poi_name`, `results_poi_dist`,
        `results_poi_rank`, listes parallèles ordonnées par pertinence
        décroissante.

TODO: `results_poi_rank` vaut `list(results.index)`, donc 0-based et troué
    après l'exclusion de l'ancre. Utiliser `range(1, len(results) + 1)`,
    comme le font les générateurs rue.
"""

# ---------------------------------------------------------------- cellule 5
D["near_metric_sql"] = """
Retourne les POIs d'une catégorie situés à moins d'une distance donnée d'un point.

Variante bornée de `near_sql` : le seuil métrique est appliqué dans la clause
WHERE, en mètres Lambert-93. Le résultat peut donc être vide, contrairement à
une recherche par k plus proches voisins.

Args:
    df (GeoDataFrame): POIs candidats en EPSG:2154.
    x (float): Abscisse Lambert-93 du point de référence.
    y (float): Ordonnée Lambert-93 du point de référence.
    cat (str): Catégorie de POI recherchée.
    distance (float): Rayon maximal, en mètres.
    k (int): Nombre maximal de résultats retournés.

Returns:
    DataFrame: Colonnes `poi_id`, `poi_name`, `dist`, triées par distance
        croissante ; vide si aucun POI n'entre dans le rayon.
"""

D["make_question_metricsql"] = """
Génère les questions à contrainte métrique « X à moins de D mètres de Y ».

Reprend l'échantillonnage stratifié de `make_question_nearsql` et décline
chaque couple (ancre, catégorie cible) sur tous les rayons demandés.

Args:
    df_osm (GeoDataFrame): POIs servant d'ancres et de cibles.
    list_distance (list[float]): Rayons à décliner, en mètres.
    nb_q (int): Nombre d'ancres visé, stratifié par catégorie.
    seed (int): Graine du générateur aléatoire.

Returns:
    DataFrame: Une ligne par (ancre, rayon). Mêmes colonnes que
        `make_question_nearsql`, plus `distance`.

TODO: le nombre de lignes produites vaut `nb_q * len(list_distance)` et non
    `nb_q` ; renommer le paramètre ou diviser le quota par strate.
TODO: `results_poi_rank` 0-based et troué, cf. `make_question_nearsql`.
"""

# ---------------------------------------------------------------- cellule 7
D["cardinal_azimuth_sql"] = """
Retourne les POIs d'une catégorie situés dans un secteur cardinal autour d'un point.

L'azimut de chaque POI est mesuré en degrés horaires depuis le nord — d'où
l'ordre `atan2(dx, dy)`, inversé par rapport à la convention mathématique. Un
POI est retenu si son écart angulaire à la direction visée, ramené dans
(-180°, 180°], ne dépasse pas `half_width`.

Args:
    df (GeoDataFrame): POIs candidats en EPSG:2154.
    x (float): Abscisse Lambert-93 du point de référence.
    y (float): Ordonnée Lambert-93 du point de référence.
    cat (str): Catégorie de POI recherchée.
    cardinal_dir (str): Direction visée, clé de `CARDINAL_AZ`.
    k (int): Nombre maximal de résultats retournés.
    half_width (float): Demi-ouverture du secteur, en degrés.
    con (duckdb.DuckDBPyConnection | None): Connexion à réutiliser ; une
        connexion jetable est créée si None.

Returns:
    DataFrame: Colonnes `poi_id`, `poi_name`, `dist`, `az` (degrés), triées
        par distance croissante.

TODO: la distance et l'azimut sont faux. `x`, `y` et `geom` sont en
    Lambert-93, donc en mètres, alors que `ST_Distance_Sphere` attend des
    degrés lon/lat. Et le facteur `cos(radians((y + ST_Y(geom)) / 2))` est une
    correction équirectangulaire réservée aux degrés : appliquée ici elle
    calcule `cos` d'environ 1.2e5 radians, soit un facteur arbitraire de
    [-1, 1] sur la composante est-ouest. Remplacer par
    `ST_Distance(geom, ST_Point($x, $y))` et
    `atan2(ST_X(geom) - $x, ST_Y(geom) - $y)` sans facteur, exactement comme
    `towards_b_sql` qui, lui, est correct.
TODO: documenter que le nord de la grille Lambert-93 s'écarte du nord
    géographique d'environ 0,5° à Paris — négligeable devant `half_width`.
TODO: `row_number()` fait doublon avec le rang recalculé côté Python et
    devient faux après filtrage de l'ancre.
"""

D["make_question_cardinalazi"] = """
Génère les questions de direction cardinale « X au nord de Y ».

Décline chaque ancre tirée sur les quatre directions demandées.

Args:
    df_osm (GeoDataFrame): POIs servant d'ancres et de cibles.
    list_direction (list[str]): Directions à décliner, clés de `CARDINAL_AZ`.
    half_width (float): Demi-ouverture du secteur, en degrés.
    nb_q (int): Nombre d'ancres visé, stratifié par catégorie.
    seed (int): Graine du générateur aléatoire.

Returns:
    DataFrame: Une ligne par (ancre, direction). Mêmes colonnes que
        `make_question_nearsql`, plus `direction`.

TODO: le libellé `f"{cat_q} at {d} meters from ..."` insère une direction là
    où il annonce une distance ; écrire `f"{cat_q} to the {d} of ..."`.
TODO: avec `half_width=70` les secteurs se recouvrent (nord couvre 290°→70°,
    est couvre 20°→160°), donc un POI au nord-est appartient à deux vérités
    terrain de la même ancre. `half_width=45` pave le cercle sans recouvrement.
TODO: `nb_q * len(list_direction)` lignes produites, pas `nb_q`.
TODO: `results_poi_rank` 0-based et troué, cf. `make_question_nearsql`.
"""

# ---------------------------------------------------------------- cellule 9
D["towards_b_sql"] = """
Retourne les POIs d'une catégorie situés dans la direction d'un point B depuis un point A.

Généralise `cardinal_azimuth_sql` en remplaçant la direction cardinale par
l'azimut du segment A→B. Le secteur est centré sur cet azimut ; les distances
restent mesurées depuis A, si bien qu'un POI au-delà de B est retenu tant
qu'il reste dans le cône.

Args:
    df (GeoDataFrame): POIs candidats en EPSG:2154.
    ax (float): Abscisse Lambert-93 du point A, origine du cône.
    ay (float): Ordonnée Lambert-93 du point A.
    bx (float): Abscisse Lambert-93 du point B, qui donne la direction.
    by (float): Ordonnée Lambert-93 du point B.
    cat (str): Catégorie de POI recherchée.
    k (int): Nombre maximal de résultats retournés.
    half_width (float): Demi-ouverture du cône, en degrés.
    con (duckdb.DuckDBPyConnection | None): Connexion à réutiliser.

Returns:
    DataFrame: Colonnes `poi_id`, `poi_name`, `dist` (mètres depuis A), `az`,
        triées par distance croissante.

Note:
    Distance euclidienne et azimut planaire, sans correction de latitude :
    c'est la forme correcte en CRS projeté, et le modèle à recopier dans
    `cardinal_azimuth_sql`.
"""

D["make_question_towardsb"] = """
Génère les questions directionnelles « X près de A en allant vers B ».

Pour chaque ancre A, tire un second POI B parmi ceux situés à moins de 1 000 m,
via un `cKDTree` construit sur les coordonnées projetées.

Args:
    df_osm (GeoDataFrame): POIs servant d'ancres, de points B et de cibles.
    half_width (float): Demi-ouverture du cône, en degrés.
    nb_q (int): Nombre de questions visé, stratifié par catégorie d'ancre.
    seed (int): Graine du générateur aléatoire.

Returns:
    DataFrame: Une ligne par question. Mêmes colonnes que
        `make_question_nearsql`, plus `point_b_index`, `point_b_name`,
        `point_b_category`, `point_b_x`, `point_b_y`.

TODO: `k != anchor.poi_id` compare deux espaces d'index différents. `k` est
    une position dans le `cKDTree`, construit sur `df_osm` APRÈS les filtres
    `within` et `dropna` ; `anchor.poi_id` est l'index d'origine, attribué
    AVANT ces filtres. L'ancre n'est donc pas exclue et un POI sans rapport
    l'est à sa place. Quand B tombe sur A, `atan2(0, 0)` renvoie 0 et la
    question « towards » devient silencieusement une question « nord ».
    Comparer des positions : `df_osm.index.get_loc(anchor.Index)`.
TODO: `results_poi_rank` 0-based et troué, cf. `make_question_nearsql`.
"""

# --------------------------------------------------------------- cellule 11
D["between_ab_sql"] = """
Retourne les POIs d'une catégorie situés dans le corridor reliant deux points.

Projette chaque POI dans le repère du segment A→B : `along` est l'abscisse
curviligne le long du segment, `cross_m` l'écart perpendiculaire. Un POI est
retenu si sa projection tombe entre A et B et si son écart latéral n'excède
pas `corridor_m`. Le classement privilégie la proximité à l'axe, puis
l'avancement le long du segment.

Args:
    df (GeoDataFrame): POIs candidats en EPSG:2154.
    ax (float): Abscisse Lambert-93 du point A.
    ay (float): Ordonnée Lambert-93 du point A.
    bx (float): Abscisse Lambert-93 du point B.
    by (float): Ordonnée Lambert-93 du point B.
    cat (str): Catégorie de POI recherchée.
    k (int): Nombre maximal de résultats retournés.
    corridor_m (float): Demi-largeur du corridor, en mètres.
    con (duckdb.DuckDBPyConnection | None): Connexion à réutiliser.

Returns:
    DataFrame: Colonnes `poi_id`, `poi_name`, `dist_a`, `dist_b`, `along`,
        `cross_m`, triées par écart latéral puis avancement.

TODO: `ab_len` vaut 0 si A et B coïncident, ce qui produit une division par
    zéro et des NaN silencieux ; garder le cas ou l'écarter en amont.
"""

D["make_question_betweenab"] = """
Génère les questions d'entre-deux « X entre A et B ».

Même tirage du point B que `make_question_towardsb` : un POI à moins de
1 000 m de l'ancre. La vérité terrain est ordonnée par distance à l'axe A-B,
et non par distance à l'ancre.

Args:
    df_osm (GeoDataFrame): POIs servant d'ancres, de points B et de cibles.
    corridor_m (float): Demi-largeur du corridor, en mètres.
    nb_q (int): Nombre de questions visé, stratifié par catégorie d'ancre.
    seed (int): Graine du générateur aléatoire.

Returns:
    DataFrame: Une ligne par question. Mêmes colonnes que
        `make_question_towardsb`, la distance rapportée étant l'écart latéral.

TODO: n'alimente ni `results_poi_dist` ni `results_poi_rank`, contrairement
    aux six autres générateurs — le schéma de sortie est incomplet.
TODO: même confusion position / `poi_id` que `make_question_towardsb`.
"""

# --------------------------------------------------------------- cellule 13
D["along_street"] = """
Retourne les POIs d'une catégorie les plus proches d'une rue.

La distance est mesurée du POI à la géométrie de la rue, pas à un point : pour
une rue en plusieurs tronçons, c'est la distance au tronçon le plus proche qui
est retenue.

Args:
    df (GeoDataFrame): POIs candidats en EPSG:2154.
    street_geom (LineString | MultiLineString): Géométrie de la rue.
    cat (str): Catégorie de POI recherchée.
    k (int): Nombre maximal de résultats retournés.

Returns:
    DataFrame: Colonnes `poi_id`, `poi_name`, `dist` (mètres), `rank`
        (1-based), triées par distance croissante.

TODO: aucun rayon maximal. « Le long de la rue » implique un couloir étroit ;
    sans borne, les 100 plus proches peuvent être à des centaines de mètres,
    surtout pour les catégories rares. Ajouter `max_dist` (~100 m).
"""

D["make_question_alongstreet"] = """
Génère les questions de linéaire « X le long de la rue R ».

Contrairement aux générateurs à ancre POI, la stratification porte ici sur la
catégorie cible : pour chaque catégorie, un quota de rues est tiré au hasard.

Args:
    df_osm (GeoDataFrame): POIs cibles en EPSG:2154.
    df_streets (GeoDataFrame): Rues, avec `name` et `geometry` en EPSG:2154.
    nb_q (int): Nombre total de questions visé, stratifié par catégorie cible.
    seed (int): Graine du générateur aléatoire.

Returns:
    DataFrame: Une ligne par question. Colonnes `query`, `category_query`,
        `street_index`, `street_name`, `street_geometry`, `function`, plus les
        quatre colonnes `results_poi_*`.

TODO: la question n'est résolvable que si le nom de rue désigne une voie
    unique dans l'emprise. Tirer uniquement des rues dont le libellé est
    unique, ou lever l'ambiguïté en nommant la commune dans la question.
TODO: tirage avec remise (`rng.choice` sur l'index à chaque tour) : une même
    rue peut produire deux questions identiques dans la même strate.
TODO: aucun filtre sur les résultats vides ou quasi vides ; les catégories
    rares produisent des questions à un seul POI très lointain.
"""

# --------------------------------------------------------------- cellule 15
D["touching_streets"] = """
Retourne les rues en contact avec une rue donnée.

Le critère est métrique et non topologique : toute rue dont la géométrie passe
à moins de `tol` mètres est retenue, ce qui couvre les nœuds partagés comme
les extrémités jointives imparfaitement numérisées.

Args:
    df_streets (GeoDataFrame): Rues en EPSG:2154.
    id_street: Index de la rue de référence dans `df_streets`.
    tol (float): Tolérance de contact, en mètres.

Returns:
    GeoDataFrame: Sous-ensemble de `df_streets` en contact, la rue de
        référence exclue.

TODO: coût linéaire en nombre de rues à chaque appel. Précalculer une fois
    l'ensemble des paires sécantes avec `gpd.sjoin(..., predicate="intersects")`.
"""

D["cross_streets"] = """
Retourne les POIs d'une catégorie les plus proches du croisement de deux rues.

Le croisement est l'intersection géométrique des deux rues.

Args:
    df (GeoDataFrame): POIs candidats en EPSG:2154.
    street_geom_a (LineString | MultiLineString): Première rue.
    street_geom_b (LineString | MultiLineString): Seconde rue.
    cat (str): Catégorie de POI recherchée.
    k (int): Nombre maximal de résultats retournés.

Returns:
    DataFrame: Colonnes `poi_id`, `poi_name`, `dist` (mètres au croisement),
        `rank` (1-based), triées par distance croissante.

TODO: `inter_point` porte mal son nom — l'intersection de deux rues
    multi-parties est un `MultiPoint`, et peut même être une `LineString` si
    deux voies sont superposées. `distance()` retient alors, pour chaque POI,
    le carrefour le plus proche de LUI : le classement mélange plusieurs
    carrefours distants de plusieurs kilomètres, sans jamais lever d'erreur.
    Valider que l'intersection est un `Point` unique et écarter la paire sinon.
TODO: aucun rayon maximal, cf. `along_street`.
"""

D["make_question_crossstreet"] = """
Génère les questions de carrefour « X au croisement de R1 et R2 ».

Pour chaque catégorie cible, tire une rue puis une rue sécante parmi celles en
contact avec elle. Les rues sans sécante sont ignorées, si bien que le nombre
de questions effectivement produites peut rester sous le quota.

Args:
    df_osm (GeoDataFrame): POIs cibles en EPSG:2154.
    df_streets (GeoDataFrame): Rues, avec `name` et `geometry` en EPSG:2154.
    nb_q (int): Nombre total de questions visé, stratifié par catégorie cible.
    seed (int): Graine du générateur aléatoire.

Returns:
    DataFrame: Une ligne par question. Colonnes `query`, `category_query`,
        `street_a_index`, `street_a_name`, `street_a_geometry`,
        `street_b_index`, `street_b_name`, `street_b_geometry`, `function`,
        plus les quatre colonnes `results_poi_*`.

TODO: ne garder que les couples de noms qui ne désignent qu'un seul carrefour
    dans toute l'emprise ; sinon la question a plusieurs réponses correctes.
TODO: les tirages infructueux ne sont pas rejoués, le quota par strate n'est
    donc pas atteint.
TODO: double espace dans le libellé, « of  {street_a} ».
"""

# --------------------------------------------------------------- cellule 17
D["side_of_street"] = """
Détermine de quel côté d'une rue se trouve un point.

Estime la tangente à la rue au droit du point, par deux points d'appui pris à
±5 m de son projeté, puis prend le signe du produit vectoriel 2D entre cette
tangente et le vecteur allant du premier point d'appui vers le point. Le signe
n'a de valeur que relative : deux points de signes opposés sont de part et
d'autre de la rue.

Args:
    street_geom (LineString | MultiLineString): Géométrie de la rue.
    pt (Point): Point à situer.

Returns:
    float: +1.0 ou -1.0 selon le côté.

TODO: `street_geom.geoms[0]` retient silencieusement le premier tronçon d'une
    `MultiLineString` — pour une rue en deux morceaux, le côté est calculé par
    rapport au mauvais tronçon. Projeter sur le tronçon le plus proche de `pt`.
"""

D["opposite_side"] = """
Retourne les POIs d'une catégorie situés en face d'un POI de référence, de l'autre côté d'une rue.

Un candidat est retenu s'il est du côté opposé au POI de référence, à moins de
`max_cross` mètres de la rue, et décalé d'au plus `max_along` mètres le long de
la rue. Le classement privilégie le vis-à-vis, c'est-à-dire le plus faible
écart d'abscisse curviligne avec le POI de référence.

Args:
    df (GeoDataFrame): POIs candidats en EPSG:2154.
    street_geom (LineString | MultiLineString): Rue traversée.
    poi_y_geom (Point): POI de référence, sur l'autre rive.
    cat (str): Catégorie de POI recherchée.
    k (int): Nombre maximal de résultats retournés.
    max_along (float): Décalage longitudinal maximal admis, en mètres.
    max_cross (float): Éloignement maximal à la rue, en mètres.

Returns:
    DataFrame: Colonnes `poi_id`, `poi_name`, `cross`, `along`, `score` et
        `rank` (1-based).

TODO: même troncature `geoms[0]` que `side_of_street`.
TODO: `sub.geometry.apply(...)` évalue `side_of_street` et `project` POI par
    POI, en Python, à chaque question. Vectoriser avec
    `GeoSeries.project`/`shapely.line_locate_point`.
"""

D["make_question_oppositeside"] = """
Génère les questions de vis-à-vis « X en face de Y, de l'autre côté de la rue R ».

Tire une rue à géométrie simple, puis un POI de référence parmi ceux situés à
moins de 40 m de cette rue.

Args:
    df_osm (GeoDataFrame): POIs servant de références et de cibles.
    df_streets (GeoDataFrame): Rues en EPSG:2154 ; seules les `LineString`
        sont retenues.
    nb_q (int): Nombre total de questions visé, stratifié par catégorie cible.
    seed (int): Graine du générateur aléatoire.

Returns:
    DataFrame: Une ligne par question. Colonnes `query`, `category_query`,
        `street_index`, `street_name`, `street_geometry`, `poi_y_id`,
        `poi_y_name`, `poi_y_geometry`, `function`, plus les quatre colonnes
        `results_poi_*`.

TODO: les tirages infructueux consomment le quota sans être rejoués — la
    boucle est un `for` sur le quota, pas un `while` sur le nombre de succès.
    Les compteurs `n` et `tries` sont d'ailleurs incrémentés sans être lus.
TODO: écarter les `MultiLineString` supprime les rues en plusieurs tronçons
    plutôt que de les traiter.
"""

# --------------------------------------------------------------- cellule 19
D["inside_area"] = """
Classe les POIs d'une catégorie par distance à une zone.

Ne filtre pas sur l'appartenance à la zone : le filtre `within` est appliqué
par l'appelant, cette fonction ne fait que trier ce qu'on lui donne.

Args:
    area (BaseGeometry): Polygone de la zone, en EPSG:2154.
    category (str): Catégorie de POI recherchée.
    pois (GeoDataFrame): POIs candidats, déjà restreints par l'appelant.
    cat_col (str): Nom de la colonne de catégorie.

Returns:
    DataFrame: `pois` filtré sur la catégorie, avec une colonne `dist`, trié
        par distance croissante et réindexé.

TODO: le tri est inopérant dans le cas « inside ». La distance d'un point
    intérieur à son polygone vaut 0 en géométrie shapely, donc `dist` est nul
    pour tous les POIs et `sort_values` conserve l'ordre d'insertion : le
    classement de la vérité terrain est arbitraire. Choisir un critère qui
    ordonne réellement — distance au centroïde, ou distance au bord via
    `area.exterior` — ou assumer que la relation « inside » est un ensemble
    non ordonné et le documenter comme tel dans le protocole d'évaluation.
TODO: corps identique à `outside_area` et `border_area` ; une seule fonction
    paramétrée par la géométrie de référence suffirait.
"""

D["make_question_insidearea"] = """
Génère les questions d'appartenance « X dans la zone Z ».

Pour chaque catégorie cible, tire des zones jusqu'à atteindre le quota, en
ne retenant que celles qui contiennent au moins un POI de la catégorie. Le
nombre de tentatives est plafonné à 200 par strate.

Args:
    df_osm (GeoDataFrame): POIs cibles en EPSG:2154.
    df_area (GeoDataFrame): Zones, avec `name` et `geometry` en EPSG:2154.
    min_size (float): Surface minimale d'une zone retenue, en m².
    nb_q (int): Nombre total de questions visé, stratifié par catégorie cible.
    seed (int): Graine du générateur aléatoire.

Returns:
    DataFrame: Une ligne par question. Colonnes `query`, `category_query`,
        `area_index`, `area_name`, `area_geometry`, `function`, plus les
        quatre colonnes `results_poi_*`.

TODO: `results_poi_rank` vaut `list(results.index)` après `reset_index`, donc
    0-based, alors que les générateurs rue sont 1-based.
TODO: le plafond de 200 tentatives est atteint silencieusement ; journaliser
    les strates incomplètes.
TODO: tirage avec remise, une même zone peut revenir dans la même strate.
TODO: `area.name` est le label d'index de la Series, `area["name"]` le nom de
    la zone. Les deux sont utilisés à trois lignes d'écart — correct, mais à
    désambiguïser pour la relecture.
"""

# --------------------------------------------------------------- cellule 21
D["outside_area"] = """
Classe les POIs d'une catégorie par distance à une zone.

Corps identique à `inside_area` ; c'est l'appelant qui restreint aux POIs
extérieurs. Ici la distance est réellement discriminante, puisqu'elle est
strictement positive hors du polygone.

Args:
    area (BaseGeometry): Polygone de la zone, en EPSG:2154.
    category (str): Catégorie de POI recherchée.
    pois (GeoDataFrame): POIs candidats, déjà restreints par l'appelant.
    cat_col (str): Nom de la colonne de catégorie.

Returns:
    DataFrame: `pois` filtré sur la catégorie, avec `dist`, trié par distance
        croissante et réindexé.
"""

D["make_question_outsidearea"] = """
Génère les questions d'exclusion « X hors de la zone Z ».

Même structure que `make_question_insidearea`, avec le complémentaire de la
zone.

Args:
    df_osm (GeoDataFrame): POIs cibles en EPSG:2154.
    df_area (GeoDataFrame): Zones, avec `name` et `geometry` en EPSG:2154.
    min_size (float): Surface minimale d'une zone retenue, en m².
    nb_q (int): Nombre total de questions visé, stratifié par catégorie cible.
    seed (int): Graine du générateur aléatoire.

Returns:
    DataFrame: Une ligne par question, mêmes colonnes que
        `make_question_insidearea`.

TODO: la vérité terrain contient TOUS les POIs de la catégorie hors de la
    zone, soit potentiellement des milliers d'entrées à l'échelle de la bbox,
    classés du plus proche du bord au plus lointain. C'est un ensemble, pas un
    classement : « hors du parc » n'ordonne rien. Décider si la relation est
    évaluée en rappel binaire plutôt qu'en ranking, ou borner à un rayon.
TODO: `results_poi_rank` 0-based, cf. `make_question_insidearea`.
"""

# --------------------------------------------------------------- cellule 23
D["border_area"] = """
Classe les POIs d'une catégorie par distance à une zone.

Contrairement à `inside_area` et `outside_area`, reçoit l'ensemble des POIs
sans restriction préalable.

Args:
    area (BaseGeometry): Polygone de la zone, en EPSG:2154.
    category (str): Catégorie de POI recherchée.
    df_osm (GeoDataFrame): POIs candidats, non filtrés.
    cat_col (str): Nom de la colonne de catégorie.

Returns:
    DataFrame: POIs de la catégorie, avec `dist`, triés par distance
        croissante et réindexés.

TODO: ne mesure pas la distance au bord mais au polygone plein. Tous les POIs
    intérieurs sont donc à distance 0 et occupent la tête du classement dans
    un ordre arbitraire, alors que la question porte précisément sur la
    bordure. Mesurer contre `area.boundary` (ou `area.exterior`), qui est la
    seule géométrie dont la distance est nulle exactement sur le pourtour.
"""

D["make_question_borderarea"] = """
Génère les questions de bordure « X en limite de la zone Z ».

Pour chaque catégorie cible, tire des zones jusqu'à atteindre le quota, dans
la limite de 200 tentatives.

Args:
    df_osm (GeoDataFrame): POIs cibles en EPSG:2154.
    df_area (GeoDataFrame): Zones, avec `name` et `geometry` en EPSG:2154.
    min_size (float): Surface minimale d'une zone retenue, en m².
    nb_q (int): Nombre total de questions visé, stratifié par catégorie cible.
    seed (int): Graine du générateur aléatoire.

Returns:
    DataFrame: Une ligne par question, mêmes colonnes que
        `make_question_insidearea`.

TODO: hérite du défaut de `border_area` — la vérité terrain est actuellement
    « les POIs intérieurs, en ordre arbitraire », pas « les POIs en bordure ».
TODO: aucun rayon maximal ; ajouter une bande (`|dist| <= 50 m` autour du
    bord) pour que la relation ait un sens.
TODO: `results_poi_rank` 0-based, cf. `make_question_insidearea`.
"""

# --------------------------------------------------------------- cellule 25
D["direction_area"] = """
Ordonne les POIs d'une zone selon un gradient directionnel.

Restreint aux POIs contenus dans la zone, puis les trie selon la coordonnée
projetée correspondant à l'axe demandé : l'ordonnée pour nord/sud, l'abscisse
pour est/ouest, en ordre décroissant vers le nord et vers l'est.

Args:
    pois (GeoDataFrame): POIs candidats en EPSG:2154.
    geom (BaseGeometry): Polygone de la zone.
    direction (str): "north", "south", "east" ou "west".

Returns:
    GeoDataFrame: POIs de la zone, avec une colonne `coord`, triés du plus
        conforme à la direction au moins conforme.

TODO: `direction in "north south"` est un test de sous-chaîne, pas
    d'appartenance à une collection. Il donne le bon résultat par accident
    ici, mais accepterait aussi `"north s"` ou `"h so"`. Idem pour
    `direction in ("south west")`, où l'absence de virgule fait de la
    parenthèse une simple chaîne et non un tuple. Écrire
    `direction in ("north", "south")` et `direction in ("south", "west")`.
TODO: le tri porte sur la coordonnée absolue, donc « nord du parc » désigne la
    moitié nord de la zone. Rapporter la coordonnée au centroïde de la zone
    rendrait la relation indépendante de la position de la zone dans la bbox.
"""

D["make_question_directionarea"] = """
Génère les questions de position relative dans une zone « X au nord de Z ».

Pour chaque catégorie cible, tire des zones et décline chacune sur les quatre
directions demandées.

Args:
    df_osm (GeoDataFrame): POIs cibles en EPSG:2154.
    df_area (GeoDataFrame): Zones, avec `name` et `geometry` en EPSG:2154.
    list_direction (list[str]): Directions à décliner.
    min_size (float): Surface minimale d'une zone retenue, en m².
    nb_q (int): Nombre total de questions visé, stratifié par catégorie cible.
    seed (int): Graine du générateur aléatoire.

Returns:
    DataFrame: Une ligne par (zone, direction). Colonnes `query`,
        `category_query`, `area_index`, `area_name`, `area_geometry`,
        `function`, `direction`, `results_poi_id`, `results_poi_x`,
        `results_poi_y`, `results_poi_name`, `results_poi_rank`.

TODO: la condition d'arrêt `while nq != n_queries_per_stratum` ne peut pas
    être satisfaite. La boucle interne incrémente `nq` d'un par direction, soit
    jusqu'à quatre par tour, si bien que `nq` saute par-dessus le quota — pour
    10 questions et 4 directions il vaut 0, 4, 8, 12 et n'atteint jamais 10. La
    boucle ne s'arrête donc que sur le plafond `i < 200`, produisant environ
    200 questions par catégorie au lieu de 10. Écrire `while nq < n_...`, et
    tirer la zone hors de la boucle des directions pour que les quatre
    questions portent sur la même zone.
TODO: schéma de sortie divergent — `results_poi_x`/`results_poi_y` au lieu de
    `results_poi_dist`, ce qui casse la concaténation avec les autres
    générateurs et l'outillage qui lit `results_poi_dist`.
TODO: `results_poi_rank` vaut `range(len(results))`, donc 0-based.
"""


# --------------------------------------------------------------------------- #

def sig_end(lines, start):
    """Index de la ligne qui referme la signature commencée en `start`."""
    depth = 0
    for j in range(start, len(lines)):
        code = re.sub(r"#.*$", "", lines[j])
        depth += code.count("(") - code.count(")")
        if depth <= 0 and code.rstrip().endswith(":"):
            return j
    raise RuntimeError(f"signature non refermée ligne {start}: {lines[start]!r}")


def drop_existing(lines, k, indent):
    """Supprime la docstring qui commence en `k`, si elle existe."""
    while k < len(lines) and not lines[k].strip():
        k += 1
    if k >= len(lines):
        return lines, 0
    s = lines[k].strip()
    if not (s.startswith('"""') or s.startswith("'''")):
        return lines, 0
    q = s[:3]
    end = k
    if not (len(s) > 3 and s.endswith(q)):
        end = k + 1
        while end < len(lines) and q not in lines[end]:
            end += 1
    return lines[:k] + lines[end + 1:], end + 1 - k


def insert(src, name, doc):
    lines = src.split("\n")
    pat = re.compile(r"^(\s*)def\s+" + re.escape(name) + r"\s*\(")
    for i, ln in enumerate(lines):
        m = pat.match(ln)
        if m:
            break
    else:
        raise KeyError(name)

    indent = m.group(1) + "    "
    end = sig_end(lines, i)
    tail, _ = drop_existing(lines[end + 1:], 0, indent)

    body = [indent + '"""' + doc.strip("\n").split("\n")[0]]
    for ln in doc.strip("\n").split("\n")[1:]:
        body.append(indent + ln if ln.strip() else "")
    body.append(indent + '"""')

    return "\n".join(lines[:end + 1] + body + tail)


def main():
    nb = json.load(open(NB))
    done = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        changed = False
        for name in re.findall(r"^def (\w+)", src, re.M):
            if name not in D:
                continue
            src = insert(src, name, D[name])
            done.append(name)
            changed = True
        if changed:
            cell["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]

    json.dump(nb, open(NB, "w"), ensure_ascii=False, indent=1)
    missing = sorted(set(D) - set(done))
    print(f"{len(done)} docstrings insérées")
    if missing:
        print("NON TROUVÉES :", missing, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
