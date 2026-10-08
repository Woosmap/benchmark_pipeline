"""Inventaire des défauts connus des templates de type A.

Chaque entrée a été **constatée en exécutant** le générateur sur les fixtures de
`conftest.py`, jamais seulement déduite de la lecture du code. L'inventaire est
donc daté : il décrit l'état du 7 octobre 2026, après le commit f1be9c1
(« refonte des fonctions de calcul des pois pour les questions geospatiale »),
et se périme dès qu'un template est corrigé.

Ces défauts ne sont pas corrigés ici : le rôle de la suite est de les constater.
Ils sont encodés en `xfail(strict=True)`, ce qui donne trois propriétés utiles :

* la suite sort **verte** — un défaut connu compte comme `xfailed`, pas `failed`,
  donc elle reste exploitable en intégration continue ;
* chaque défaut reste **visible et documenté**, avec son fichier et sa ligne ;
* `strict=True` fait échouer le test en **XPASS** dès que le défaut est corrigé.
  Le test dit alors « c'est réparé, retire le marqueur », au lieu de rester vert
  en silence et de laisser l'inventaire mentir.

C'est ce qui s'est produit au passage de f1be9c1. Quatre défauts de la version
précédente de ce fichier ont disparu, et la suite les a signalés :

* `area_border_mesure_le_polygone_plein` — `border_area` est devenu
  `pois_near_border`, qui mesure contre `area.boundary`. Mesuré : les 24 POIs
  intérieurs du Parc Carre portent 20 distances distinctes, de 43,2 à 151,6 m,
  là où l'ancienne version les donnait toutes à 0,0 m ;
* `area_direction_test_de_sous_chaine` — `direction_area` est devenu
  `segregate_pois`, qui indexe un dictionnaire de masques : `'north s'`, `'h so'`
  et `''` lèvent maintenant KeyError au lieu d'être acceptés ;
* `point_near_cardinal_garde_fou_off_by_one` — le garde-fou `if nb_q <= 20` a
  été retiré ; le template tourne à n'importe quel `nb_q`, et
  `conftest.TEMPLATE_NB_Q`, qui n'existait que pour le contourner, a été
  supprimé avec lui ;
* `point_between_quota` — la stratification produit désormais exactement `nb_q`
  questions (10 pour nb_q=10, 14 pour nb_q=14).

Un cinquième a été corrigé pendant l'écriture de cette suite : `point_near_metric`
publiait `results` — les quatre lots de rayons concaténés — là où il calculait
`result`. Mesuré après correction : plus aucun doublon ni POI hors du rayon
annoncé. La correction a rendu visible le défaut qu'elle masquait, les réponses
vides au rayon de 100 m, désormais dans `UNANSWERABLE_TEMPLATES`.

Les tests qui les démontraient sont devenus des tests de non-régression
ordinaires dans `test_known_defects.py`.

Sept familles, parce qu'elles ne se corrigent pas au même endroit :

`BROKEN_TEMPLATES`
    Le générateur lève. Il ne produit **aucune** question, donc aucun invariant
    de cohérence ne peut être évalué.
`INCOHERENT_TEMPLATES`
    Le générateur tourne mais sa sortie viole le schéma.
`MISDECLARED_CONTEXT_COLUMNS`
    Le registre annonce des colonnes que le générateur ne publie pas (ou plus).
    La sortie, elle, reste cohérente.
`STALE_FUNCTION_NAMES`
    La colonne `function` nomme une fonction qui n'existe plus dans le module.
`UNANSWERABLE_TEMPLATES`
    Le générateur tourne et sa sortie est cohérente, mais il laisse passer des
    questions sans aucune réponse — inévaluables pour un modèle de recherche.
`DEAD_IMPORTS`
    Le module importe des noms qu'il n'utilise jamais — et paie une dépendance
    pour rien. Indexée par chemin de module, et non par nom de template.
`SEMANTIC_DEFECTS`
    La sortie est bien formée, mais le volume produit, le classement ou la
    robustesse du tirage sont faux.
"""

#: Templates dont le **module** ne s'importe même pas. Sous-ensemble strict de
#: `BROKEN_TEMPLATES` : distingué parce que seul ce défaut-là fait échouer
#: `test_module_imports`.
#:
#: Vide depuis la correction de `street_cross.py`, dont le bas de fichier avait
#: gardé l'appel du notebook (`bench4=make_question_crossstreet(...)`). Le
#: dictionnaire est conservé — et non supprimé — parce que la régression est
#: facile à réintroduire en sortant un nouveau template d'un notebook.
IMPORT_ERRORS = {}

#: Templates dont le générateur lève avant de produire quoi que ce soit.
#:
#: Vide au moment de la mesure, mais de justesse : `area_inside` et
#: `area_direction` lèvent dès que le corpus de zones compte moins de `nb_q`
#: entrées, ce qui était le cas des anciennes fixtures à trois zones. Le défaut
#: est tenu dans `SEMANTIC_DEFECTS` — `area_echantillonnage_borne` — et
#: démontré sur un corpus réduit, plutôt que de condamner les deux templates à
#: n'être testés que par leur plantage.
BROKEN_TEMPLATES = {}

#: Templates qui tournent mais dont la sortie viole le schéma du benchmark.
INCOHERENT_TEMPLATES = {
    "area_border": (
        "area_border.py:46 — `pois_near_border(df_osm, area.geometry, band)` "
        "ne reçoit pas `cat_q` et ne filtre donc rien, alors que l'énoncé "
        "(`f\"{cat_q} at the border of …\"`) et la colonne `category_query` "
        "annoncent tous deux une catégorie. La vérité terrain contient les "
        "cinq catégories du corpus : un modèle qui répond exactement ce que la "
        "question demande est pénalisé. Mesuré : 10 questions sur 10 à nb_q=10"
    ),
    "area_direction": (
        "area_direction.py:57-61 — publie `results_poi_x`/`results_poi_y` au "
        "lieu de `results_poi_dist`. Le schéma de sortie diverge de celui des "
        "onze autres templates : la concaténation en un benchmark unique perd "
        "la colonne, et tout l'outillage qui lit `results_poi_dist` "
        "(plot_question, calcul de nDCG) ne trouve rien"
    ),
}

#: Templates auxquels il manque une des quatre listes de vérité terrain.
#: **Sous-ensemble strict d'`INCOHERENT_TEMPLATES`**, sur le modèle
#: d'`IMPORT_ERRORS` ⊂ `BROKEN_TEMPLATES` : les deux autres incohérents
#: publient bien les quatre listes — `point_near_metric` les remplit mal,
#: `area_border` y met les mauvais POIs — donc les marquer ici les ferait
#: XPASS(strict) sur `test_result_lists_complete`, qu'ils passent.
INCOMPLETE_RESULT_LISTS = {
    "area_direction": INCOHERENT_TEMPLATES["area_direction"],
}

#: Templates dont le registre annonce des colonnes de contexte que le
#: générateur ne publie pas (ou plus).
#:
#: Famille distincte d'`INCOHERENT_TEMPLATES` : la divergence est entre le
#: *registre* et le module, pas dans la sortie elle-même. Le benchmark produit
#: reste cohérent, donc marquer ces templates incohérents ferait échouer en
#: XPASS(strict) `test_benchmark_is_coherent`, qu'ils passent. Seuls
#: `test_declared_columns_present` et `test_query_mentions_its_context` lisent
#: cette table.
#:
#: Deux causes y figuraient, toutes deux réglées côté schéma plutôt
#: qu'ici : neuf templates ne publiaient plus `category_query`, que
#: `BENCHMARK_CORE_COLUMNS` exigeait partout, et plus aucun ne produisait
#: `same_cat`, qu'annonçait `_ANCHOR_COLS`. C'était bien le registre qui était
#: périmé : les deux colonnes ont été retirées du schéma, `category_query`
#: devenant facultative (`OPTIONAL_CONTEXT_COLUMNS`) pour les trois templates
#: qui filtrent encore par catégorie. Les sept entrées que ces deux causes
#: justifiaient ont disparu avec elles.
#:
#: Ne reste que la troisième : le vocabulaire du second point — `point_towards`
#: publie `anchor_b_*` et `point_between` `anchor_a_*`/`anchor_b_*`, quand le
#: registre annonce `point_b_*`.
MISDECLARED_CONTEXT_COLUMNS = {
    "point_towards": (
        "point_towards.py:109-113 publie `anchor_b_index`, `anchor_b_name`, "
        "`anchor_b_category`, `anchor_b_x`, `anchor_b_y` là où le registre "
        "annonce `_POINT_B_COLS` (`point_b_*`) : les cinq colonnes déclarées "
        "sont absentes du DataFrame"
    ),
    "point_between": (
        "schema.py:126-129 déclare `point_between` avec `_ANCHOR_COLS + "
        "_POINT_B_COLS` (`anchor_index`, `anchor_name`…, `point_b_index`…), "
        "mais point_between.py:156-165 publie `anchor_a_*` et `anchor_b_*` : "
        "les dix colonnes annoncées sont absentes. Le template ayant deux "
        "ancres symétriques et non une ancre et un point B, c'est le registre "
        "qui est périmé — il a gardé le vocabulaire de `point_towards`. "
        "Conséquence en cascade : `label_column='anchor_name'` ne désigne "
        "aucune colonne, donc `test_query_mentions_its_context` lève KeyError "
        "au lieu de comparer l'énoncé à son libellé"
    ),
}

#: Templates dont la colonne désignée par `label_column` est absente.
#: **Sous-ensemble strict de `MISDECLARED_CONTEXT_COLUMNS`** : `point_towards`,
#: l'autre mal déclaré, publie bien son libellé (`anchor_name`), seules
#: d'*autres* colonnes lui manquent. `point_between` seul peut faire lever
#: `test_query_mentions_its_context`.
MISSING_LABEL_COLUMN = {
    "point_between": MISDECLARED_CONTEXT_COLUMNS["point_between"],
}

#: Templates dont la colonne `function` nomme une fonction absente du module.
#:
#: `function` est la seule trace, dans le benchmark enregistré, de la manière
#: dont la vérité terrain a été calculée : le nom doit pouvoir être résolu pour
#: que le jeu reste auditable. Le refactor a renommé les helpers sans toucher à
#: la chaîne publiée.
#:
#: À corriger des deux côtés en même temps : `targets/ellypses.py` aiguille sur
#: ces mêmes chaînes (`get_ellypse`, :69-141), donc renommer la valeur publiée
#: sans renommer la branche correspondante casserait le calcul des ellipses.
STALE_FUNCTION_NAMES = {
    "point_near_cardinal": (
        "publie `function='cardinal_azimuth_sql'`, mais le module ne définit "
        "que `pack_by_direction` (point_near_cardinal.py:20) — l'ancien nom a "
        "survécu au renommage du helper"
    ),
    "area_inside": (
        "publie `function='inside_area'`, nom d'une fonction qui n'existe plus "
        "du tout : `make_question_area_inside` calcule la vérité terrain en "
        "ligne (area_inside.py:37) depuis la fusion du helper"
    ),
    "area_border": (
        "publie `function='border_area'`, mais le helper s'appelle "
        "`pois_near_border` (area_border.py:10)"
    ),
    "area_direction": (
        "publie `function='direction_area'`, mais le helper s'appelle "
        "`segregate_pois` (area_direction.py:10)"
    ),
}

#: Templates dont la sortie ne satisfait pas les préconditions de
#: `plot_question`. **Sous-ensemble strict de `INCOHERENT_TEMPLATES`**, sur le
#: modèle d'`IMPORT_ERRORS` ⊂ `BROKEN_TEMPLATES` : toute violation du schéma
#: n'empêche pas d'afficher la question.
#:
#: `plot_question` a besoin des trois listes `id`, `name` et `rank` — pas de
#: `dist`. `area_direction`, à qui il ne manque que `dist`, reste donc
#: affichable et ne figure pas ici. La table est vide, et le test qu'elle pilote
#: devient une garantie plutôt qu'un constat.
UNPLOTTABLE_TEMPLATES = {}

#: Modules qui importent des noms qu'ils n'utilisent jamais.
#:
#: Famille à part des `IMPORT_ERRORS` : celle-ci est indexée par chemin de
#: module et ne concerne pas les douze templates, mais les cibles ajoutées par
#: f1be9c1.
#:
#: Vide depuis le nettoyage des imports. `ellypses.py` et `mask_anchors.py`
#: tiraient `sklearn` et `random.Random` sans jamais s'en servir ; comme
#: scikit-learn n'est pas dans les `dependencies` mais seulement dans l'extra
#: `nlp`, un `uv sync` sans extras — l'installation que décrit le README —
#: rendait `get_ellypse` et `make_mask` injoignables. Avec `--all-extras`, ce
#: que fait la CI, les deux modules s'importaient : le défaut était invisible
#: une fois sur deux, d'où le contrôle statique par AST de
#: `test_target_modules_import_only_what_they_use`, qui tombe pareil dans les
#: deux installations. Garder la table vide plutôt que la supprimer : le test
#: qui la lit reste en place et se repeuplera si la régression revient.
DEAD_IMPORTS = {}

#: Templates qui laissent passer des questions sans aucune réponse.
UNANSWERABLE_TEMPLATES = {
    "point_near_metric": (
        "point_near_metric.py:63-76 — aucun garde-fou sur un résultat vide : un "
        "rayon de 100 m autour d'une ancre isolée ne trouve rien, et la question "
        "est ajoutée au benchmark quand même. Mesuré : 3 questions vides sur 40 "
        "à nb_q=10, toutes au rayon de 100 m.\n"
        "  Ce défaut était masqué tant que le template publiait les quatre lots "
        "de rayons concaténés : aucune ligne n'était alors vide, puisque chacune "
        "portait la réponse des trois autres. Il est réapparu avec la correction "
        "de `results` en `result` — une correction en découvre une autre"
    ),
    "point_near_cardinal": (
        "point_near_cardinal.py:80-94 — les quatre secteurs sont publiés sans "
        "contrôle : un secteur cardinal peut ne contenir aucun POI, et la "
        "question est tout de même ajoutée au benchmark. Mesuré : 7 questions "
        "vides sur 40 à nb_q=10, 10 sur 56 à nb_q=14"
    ),
}

#: Défauts qui n'altèrent pas la cohérence d'une question prise isolément, mais
#: le volume ou la robustesse du jeu produit. Vérifiés un par un dans
#: `test_known_defects.py`.
SEMANTIC_DEFECTS = {
    "quota_multiplie_par_la_boucle_interne": (
        "`point_near_metric`, `point_near_cardinal` et `area_direction` "
        "bouclent `nb_q` fois sur l'ancre ou la zone, puis déclinent chaque "
        "tirage sur leur seconde dimension — quatre rayons, quatre secteurs, "
        "quatre directions — sans jamais diviser le quota. Ils rendent donc "
        "exactement 4 × nb_q questions. Mesuré : 40 pour nb_q=10 et 56 pour "
        "nb_q=14, pour les trois. `nb_q` étant un plafond, rendre moins "
        "serait légitime ; le dépasser d'un facteur 4 ne l'est pas, et "
        "déséquilibre le jeu final où ces trois templates pèsent quatre "
        "fois leur part"
    ),
    "area_echantillonnage_borne": (
        "area_inside.py:34 et area_direction.py:42 tirent `nb_q` indices "
        "**distincts** (`rng.choice(..., replace=False)`) dans un corpus de "
        "zones qui en compte souvent moins. Deux conséquences :\n"
        "  * sous `nb_q` zones, les deux templates lèvent `ValueError: Cannot "
        "take a larger sample than population` au lieu de rendre ce qu'ils "
        "peuvent. Mesuré : area_direction passe à nb_q=18 (18 zones retenues) "
        "et lève à nb_q=19 ;\n"
        "  * `area_inside` écrit `rng.choice(min(len(areas), nb_q), size=nb_q)`, "
        "donc tire dans `range(nb_q)` et non dans les zones : au-delà de la "
        "`nb_q`-ième, aucune zone n'est atteignable, quelle que soit la graine. "
        "Mesuré : 10 zones tirées sur 18, identiques sur 30 graines"
    ),
    "area_inside_dist_toujours_nulle": (
        "area_inside.py:48 — `results.geometry.distance(area.geometry)` vaut "
        "0.0 pour tout POI intérieur à un polygone. `results_poi_dist` est donc "
        "une colonne de zéros et `results_poi_rank` numérote l'ordre du corpus : "
        "le classement de la vérité terrain est arbitraire. Classer par "
        "distance au centroïde, ou assumer que « inside » est un ensemble non "
        "ordonné et cesser de publier un rang"
    ),
    "area_direction_rang_sans_classement": (
        "area_direction.py:61 — `segregate_pois` masque les POIs sans les "
        "trier, donc `results_poi_rank` numérote l'ordre d'apparition dans "
        "`df_osm`. Même défaut qu'`area_inside_dist_toujours_nulle` mais sans "
        "même une colonne de distance pour le signaler : deux modèles qui "
        "ordonnent différemment les mêmes bons POIs sont notés différemment "
        "sans raison. Trier par distance au centroïde, ou le long de l'axe de "
        "la direction demandée"
    ),
    "street_cross_intersection_vide": (
        "street_cross.py:26 — `touching_streets(tol=1.0)` retient des rues qui "
        "ne se croisent pas ; `intersection()` est alors vide et "
        "`distance(vide)` vaut NaN dans results_poi_dist"
    ),
    "street_opposite_side_multilinestring_ignoree": (
        "street_opposite_side.py:127 — `'MultiString'` au lieu de "
        "`'MultiLineString'` dans le filtre `geom_type.isin([...])` : toutes "
        "les rues MultiLineString sont écartées du tirage. Le module sait "
        "pourtant les traiter, `side_of_street` (:31) et `opposite_side` (:66) "
        "prennent explicitement `geoms[0]` pour ce cas"
    ),
    "mask_anchors_ignore_le_poi_de_reference": (
        "mask_anchors.py:8 — `ANCHOR_COL = r\"(anchor|area|street)(_[ab])?_name\"` "
        "ne reconnaît pas `poi_y_name`, la colonne où `street_opposite_side` "
        "range son POI de référence. Or son énoncé le nomme "
        "(`f\"{cat_q} across {street_name} from {poi_y_name}\"`) : les tokens de "
        "ce POI restent à 0 dans `mask_token_anchors`, donc traités comme du "
        "vocabulaire de la question et non comme une entité nommée. C'est le "
        "seul libellé d'un template qui échappe au masque"
    ),
    "street_identity_incoherente": (
        "street_cross.py:71-72 tire une *valeur* de `id_street` et la passe à "
        "`.loc`, alors que street_along.py:58 tire dans `df_streets.index`. "
        "Les deux modules ne s'accordent pas sur ce qui identifie une rue : sur "
        "un `df_streets` réindexé — ce que fait n'importe quel filtrage en "
        "amont — street_cross se trompe de rue en silence, ou lève"
    ),
}


def merge_reasons(*tables):
    """Fusionne des tables de défauts en `nom -> motif unique`.

    Un template peut figurer dans plusieurs familles — `area_direction` est à la
    fois incohérent, mal déclaré et porteur d'un nom de fonction périmé. Les
    motifs sont alors concaténés, pour qu'un seul `xfail` porte toute
    l'explication : empiler deux marqueurs sur le même cas rendrait le rapport
    ambigu.
    """
    merged = {}
    for table in tables:
        for name, reason in table.items():
            if name in merged:
                merged[name] = f"{merged[name]} ; et {reason}"
            else:
                merged[name] = reason
    return merged
