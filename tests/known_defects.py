"""Inventaire des défauts connus des templates de type A.

Chaque entrée a été **constatée en exécutant** le générateur sur les fixtures de
`conftest.py`, jamais seulement déduite de la lecture du code.

Ces défauts ne sont pas corrigés : le rôle de la suite est de les constater. Ils
sont donc encodés en `xfail(strict=True)`, ce qui donne trois propriétés utiles :

* la suite sort **verte** — un défaut connu compte comme `xfailed`, pas `failed`,
  donc elle reste exploitable en intégration continue ;
* chaque défaut reste **visible et documenté**, avec son fichier et sa ligne ;
* `strict=True` fait échouer le test en **XPASS** dès que le défaut est corrigé.
  Le test dit alors « c'est réparé, retire le marqueur », au lieu de rester vert
  en silence et de laisser l'inventaire mentir.

Trois familles, parce qu'elles ne se corrigent pas au même endroit :

`BROKEN_TEMPLATES`
    Le générateur lève. Il ne produit **aucune** question, donc aucun invariant
    de cohérence ne peut être évalué.
`INCOHERENT_TEMPLATES`
    Le générateur tourne mais sa sortie viole le schéma.
`UNANSWERABLE_TEMPLATES`
    Le générateur tourne et sa sortie est cohérente, mais il laisse passer des
    questions sans aucune réponse — inévaluables pour un modèle de recherche.
"""

#: Templates dont le **module** ne s'importe même pas. Sous-ensemble strict de
#: `BROKEN_TEMPLATES` : distingué parce que seul ce défaut-là fait échouer
#: `test_module_imports`. Les autres modules s'importent très bien et ne cassent
#: qu'à l'appel du générateur.
IMPORT_ERRORS = {
    "street_cross": (
        "street_cross.py:106-107 — `bench4=make_question_crossstreet(df_osm, "
        "df_streets)` au niveau module : ancien nom de fonction et globales "
        "inexistantes, donc NameError dès l'import"
    ),
}

#: Templates dont le générateur lève avant de produire quoi que ce soit.
BROKEN_TEMPLATES = {
    "point_towards": (
        "point_towards.py:85 — `print(tree.index)` : cKDTree n'expose pas "
        "d'attribut `index`, donc AttributeError à chaque appel"
    ),
    "street_cross": (
        "street_cross.py:106-107 — `bench4=make_question_crossstreet(df_osm, "
        "df_streets)` au niveau module : ancien nom de fonction et globales "
        "inexistantes, donc NameError dès l'import"
    ),
    "area_inside": (
        "area_inside.py:86 — `area['name_area']` : orthographe qui n'existe "
        "dans aucun loader (le contrat porte `area_name`), donc KeyError"
    ),
    "area_outside": (
        "area_outside.py:69 — `area['name_area']`, cf. area_inside"
    ),
    "area_border": (
        "area_border.py:76 — `area['name_area']`, cf. area_inside"
    ),
    "area_direction": (
        "area_direction.py:94 — `area['name_area']`, cf. area_inside"
    ),
    "street_along": (
        "street_along.py:78 — `street['name']` alors que la f-string voisine "
        "(:75) utilise déjà `street['street_name']`, seul nom du contrat"
    ),
    "street_opposite_side": (
        "street_opposite_side.py:129 — `street['name']`, cf. street_along"
    ),
}

#: Templates qui tournent mais dont la sortie viole le schéma du benchmark.
INCOHERENT_TEMPLATES = {
    "point_between": (
        "point_between.py:126-127 — ne publie ni `results_poi_dist` ni "
        "`results_poi_rank`, alors que le SQL calcule déjà "
        "`row_number() OVER (ORDER BY cross_m, along) AS rank` (:60). "
        "Casse plot_question et tout invariant de rang"
    ),
}

#: Templates qui laissent passer des questions sans aucune réponse.
UNANSWERABLE_TEMPLATES = {
    "point_near_metric": (
        "point_near_metric.py:61 — aucun garde-fou sur un résultat vide : un "
        "rayon de 100 m autour d'une ancre isolée ne trouve rien, et la "
        "question est tout de même ajoutée au benchmark"
    ),
    "point_near_cardinal": (
        "point_near_cardinal.py:96 — idem : un secteur cardinal peut ne "
        "contenir aucun POI de la catégorie, la question est ajoutée quand même"
    ),
    "point_between": (
        "point_between.py:120 — idem : le corridor A→B peut être vide"
    ),
}

#: Défauts qui n'altèrent pas la cohérence d'une question prise isolément, mais
#: le volume ou la robustesse du jeu produit. Vérifiés un par un dans
#: `test_known_defects.py`.
SEMANTIC_DEFECTS = {
    "area_inside_dist_toujours_nulle": (
        "area_inside.py:34 — `sel.geometry.distance(area)` vaut 0.0 pour tout "
        "POI intérieur à un polygone, donc `sort_values('dist')` conserve "
        "l'ordre d'insertion : le classement de la vérité terrain est "
        "arbitraire. Classer par distance au centroïde, ou assumer que "
        "« inside » est un ensemble non ordonné"
    ),
    "area_border_mesure_le_polygone_plein": (
        "area_border.py:30 — même cause, mais ici la question porte "
        "précisément sur la bordure : il faut mesurer contre `area.boundary`, "
        "seule géométrie dont la distance est nulle exactement sur le pourtour"
    ),
    "street_cross_intersection_vide": (
        "street_cross.py:31 — `touching_streets(tol=1.0)` retient des rues qui "
        "ne se croisent pas ; `intersection()` est alors vide et "
        "`distance(vide)` vaut NaN dans results_poi_dist"
    ),
    "street_opposite_side_multilinestring_ignoree": (
        "street_opposite_side.py:107 — `'MultiString'` au lieu de "
        "`'MultiLineString'` : toutes les rues MultiLineString sont écartées "
        "du tirage"
    ),
    "point_near_metric_quota": (
        "point_near_metric.py:48 — `nb_q // (len(list_cat) + len(list_distance))` "
        "additionne là où il faut multiplier : la boucle parcourt le *produit* "
        "catégories × distances, donc le jeu produit est ~2,1× plus gros que "
        "`nb_q` (mesuré : 232 questions pour nb_q=110)"
    ),
    "area_direction_quota": (
        "area_direction.py — `while nq != n_queries_per_stratum` avec une "
        "boucle interne sur 4 directions : nq avance de 4 et saute le quota "
        "(0, 4, 8, 12… n'atteint jamais 10). Seul le plafond `i < 200` arrête "
        "la boucle"
    ),
    "area_direction_test_de_sous_chaine": (
        "area_direction.py:34-35 — `direction in 'north south'` est un test de "
        "sous-chaîne, pas d'appartenance : juste par accident pour les quatre "
        "cardinaux, mais accepterait aussi `'north s'` ou `'h so'`. "
        "`direction in ('south west')` est une chaîne, pas un tuple"
    ),
    "street_identity_incoherente": (
        "street_cross.py:81-82 utilise une *valeur* de `id_street` comme label "
        "de `.loc`, alors que street_along.py:71 tire dans `df_streets.index`. "
        "Les deux modules ne s'accordent pas sur ce qui identifie une rue"
    ),
}


def merge_reasons(*tables):
    """Fusionne des tables de défauts en `nom -> motif unique`.

    Un template peut figurer dans plusieurs familles — `point_between` est à la
    fois incohérent et inévaluable. Les motifs sont alors concaténés, pour
    qu'un seul `xfail` porte toute l'explication : empiler deux marqueurs sur le
    même cas rendrait le rapport ambigu.
    """
    merged = {}
    for table in tables:
        for name, reason in table.items():
            if name in merged:
                merged[name] = f"{merged[name]} ; et {reason}"
            else:
                merged[name] = reason
    return merged
