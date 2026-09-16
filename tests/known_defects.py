"""Inventaire des défauts connus des templates de type A.

Chaque entrée a été **constatée en exécutant** le générateur sur les fixtures de
`conftest.py`, jamais seulement déduite de la lecture du code. L'inventaire est
donc daté : il décrit l'état du 15 septembre 2026, et se périme dès qu'un
template est corrigé.

Ces défauts ne sont pas corrigés ici : le rôle de la suite est de les constater.
Ils sont encodés en `xfail(strict=True)`, ce qui donne trois propriétés utiles :

* la suite sort **verte** — un défaut connu compte comme `xfailed`, pas `failed`,
  donc elle reste exploitable en intégration continue ;
* chaque défaut reste **visible et documenté**, avec son fichier et sa ligne ;
* `strict=True` fait échouer le test en **XPASS** dès que le défaut est corrigé.
  Le test dit alors « c'est réparé, retire le marqueur », au lieu de rester vert
  en silence et de laisser l'inventaire mentir.

C'est exactement ce qui s'est produit : la version précédente de ce fichier
répertoriait huit templates cassés (`point_towards`, les quatre `area_*`, les
trois `street_*`) pour des noms de colonnes erronés — `area['name_area']`,
`street['name']` — et un appel resté au niveau module dans `street_cross.py`.
Tous ont été réparés depuis, et la suite l'a signalé par 142 `XPASS(strict)`.
Ils ne figurent plus ci-dessous ; les tests qui les démontraient sont devenus
des tests de non-régression ordinaires dans `test_known_defects.py`.

Cinq familles, parce qu'elles ne se corrigent pas au même endroit :

`BROKEN_TEMPLATES`
    Le générateur lève. Il ne produit **aucune** question, donc aucun invariant
    de cohérence ne peut être évalué.
`INCOHERENT_TEMPLATES`
    Le générateur tourne mais sa sortie viole le schéma.
`UNPLOTTABLE_TEMPLATES`
    Sous-ensemble du précédent : la sortie ne peut même pas être affichée.
`UNANSWERABLE_TEMPLATES`
    Le générateur tourne et sa sortie est cohérente, mais il laisse passer des
    questions sans aucune réponse — inévaluables pour un modèle de recherche.
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
#: Vide au moment de la mesure : les douze générateurs tournent. `point_between`
#: y a figuré brièvement pendant son refactor de stratification — `allocate`
#: était appelé sans son argument `n` — et en est sorti dès que l'appel a été
#: complété en `allocate(nb_q, ratio_cat_q)`.
BROKEN_TEMPLATES = {}

#: Templates qui tournent mais dont la sortie viole le schéma du benchmark.
INCOHERENT_TEMPLATES = {
    "area_direction": (
        "area_direction.py:104-106 — publie `results_poi_x`/`results_poi_y` "
        "au lieu de `results_poi_dist`. Le schéma de sortie diverge donc de "
        "celui des onze autres templates : la concaténation en un benchmark "
        "unique perd la colonne, et tout l'outillage qui lit `results_poi_dist` "
        "(plot_question, calcul de nDCG) ne trouve rien. Le module porte "
        "lui-même le TODO (:69-71)"
    ),
}

#: Templates dont la sortie ne satisfait pas les préconditions de
#: `plot_question`. **Sous-ensemble strict de `INCOHERENT_TEMPLATES`**, sur le
#: modèle d'`IMPORT_ERRORS` ⊂ `BROKEN_TEMPLATES` : toute violation du schéma
#: n'empêche pas d'afficher la question.
#:
#: `plot_question` a besoin des trois listes `id`, `name` et `rank` — pas de
#: `dist`. `area_direction`, à qui il ne manque que `dist`, reste donc
#: affichable et ne figure pas ici ; `point_between`, à qui il manque aussi
#: `rank`, ne l'est pas. Sans cette distinction, `area_direction` serait marqué
#: xfail sur un test qu'il passe, et le XPASS(strict) ferait échouer la suite
#: pour un défaut qui n'existe pas.
UNPLOTTABLE_TEMPLATES = {}

#: Templates qui laissent passer des questions sans aucune réponse.
UNANSWERABLE_TEMPLATES = {
    "point_near_metric": (
        "point_near_metric.py — aucun garde-fou sur un résultat vide : un "
        "rayon de 100 m autour d'une ancre isolée ne trouve rien, et la "
        "question est tout de même ajoutée au benchmark (mesuré : 4/20 à nb_q=10)"
    ),
    "point_near_cardinal": (
        "point_near_cardinal.py — idem : un secteur cardinal peut ne "
        "contenir aucun POI de la catégorie, la question est ajoutée quand "
        "même (mesuré : 9/40 à nb_q=10)"
    ),
}

#: Défauts qui n'altèrent pas la cohérence d'une question prise isolément, mais
#: le volume ou la robustesse du jeu produit. Vérifiés un par un dans
#: `test_known_defects.py`.
SEMANTIC_DEFECTS = {
    "area_inside_dist_toujours_nulle": (
        "area_inside.py — `sel.geometry.distance(area)` vaut 0.0 pour tout "
        "POI intérieur à un polygone, donc `sort_values('dist')` conserve "
        "l'ordre d'insertion : le classement de la vérité terrain est "
        "arbitraire (mesuré : 8/10 questions à distance constante). Classer "
        "par distance au centroïde, ou assumer que « inside » est un ensemble "
        "non ordonné"
    ),
    "area_border_mesure_le_polygone_plein": (
        "area_border.py — même cause, mais ici la question porte précisément "
        "sur la bordure : il faut mesurer contre `area.boundary`, seule "
        "géométrie dont la distance est nulle exactement sur le pourtour. "
        "Mesuré : les 5 POIs intérieurs du Parc Carre sont tous donnés à 0.0 m "
        "du bord, alors que l'oracle attend 46,8 à 53,0 m"
    ),
    "point_between_quota": (
        "point_between.py — le quota par catégorie est respecté mais le total "
        "dépasse `nb_q` d'exactement len(list_cat) : la boucle interne épuise "
        "sa catégorie avant de revérifier le compte. Mesuré : 15 questions "
        "pour nb_q=10 (×1,50), 45 pour nb_q=40 (×1,12), 115 pour nb_q=110 "
        "(×1,05). Le dépassement étant constant en valeur absolue, il ne se "
        "voit qu'aux petites tailles — d'où un test à nb_q=10 seulement"
    ),
    "point_near_cardinal_quota": (
        "point_near_cardinal.py — même faute que point_near_metric, avec le "
        "même effet en pire : la boucle parcourt catégories × 4 directions et "
        "le quota est divisé par leur somme. Mesuré : 40 questions pour "
        "nb_q=10, 160 pour nb_q=40, 440 pour nb_q=110 — soit exactement 4×, "
        "le nombre de directions"
    ),
    "area_direction_quota": (
        "area_direction.py — `while nq != n_queries_per_stratum` avec une "
        "boucle interne sur 4 directions : nq avance de 4 et saute le quota "
        "(0, 4, 8, 12… n'atteint jamais 10). Seul le plafond `i < 200` arrête "
        "la boucle, d'où un volume à la fois énorme et non monotone — mesuré : "
        "802 questions pour nb_q=10, 135 pour nb_q=40, 822 pour nb_q=110"
    ),
    "street_cross_sous_production": (
        "street_cross.py — produit moins de questions que demandé là où les "
        "autres templates atteignent leur quota : mesuré 7 pour nb_q=10, 25 "
        "pour 40, 64 pour 110. Le tirage exige deux rues sécantes *et* des POIs "
        "près du croisement ; il abandonne une strate au lieu de retirer une "
        "autre paire de rues. Sur un corpus réel le déficit est moins visible, "
        "mais le jeu reste déséquilibré entre templates"
    ),
    "street_cross_intersection_vide": (
        "street_cross.py — `touching_streets(tol=1.0)` retient des rues qui "
        "ne se croisent pas ; `intersection()` est alors vide et "
        "`distance(vide)` vaut NaN dans results_poi_dist"
    ),
    "street_opposite_side_multilinestring_ignoree": (
        "street_opposite_side.py:109 — `'MultiString'` au lieu de "
        "`'MultiLineString'` dans le filtre `geom_type.isin([...])` : toutes "
        "les rues MultiLineString sont écartées du tirage. Le module sait "
        "pourtant les traiter, `side_of_street` (:30) et `cross_along` (:65) "
        "prennent explicitement `geoms[0]` pour ce cas"
    ),
    "street_identity_incoherente": (
        "street_cross.py:83-84 tire une *valeur* de `id_street` et la passe à "
        "`.loc`, alors que street_along.py:73 tire dans `df_streets.index`. "
        "Les deux modules ne s'accordent pas sur ce qui identifie une rue : sur "
        "un `df_streets` réindexé — ce que fait n'importe quel filtrage en "
        "amont — street_cross se trompe de rue en silence, ou lève"
    ),
    "area_direction_test_de_sous_chaine": (
        "area_direction.py — `direction in 'north south'` est un test de "
        "sous-chaîne, pas d'appartenance : juste par accident pour les quatre "
        "cardinaux, mais accepterait aussi `'north s'` ou `'h so'`. "
        "`direction in ('south west')` est une chaîne, pas un tuple"
    ),
}


def merge_reasons(*tables):
    """Fusionne des tables de défauts en `nom -> motif unique`.

    Un template peut figurer dans plusieurs familles — `point_between` est à la
    fois incohérent, inaffichable et inévaluable. Les motifs sont alors concaténés,
    pour qu'un seul `xfail` porte toute l'explication : empiler deux marqueurs
    sur le même cas rendrait le rapport ambigu.
    """
    merged = {}
    for table in tables:
        for name, reason in table.items():
            if name in merged:
                merged[name] = f"{merged[name]} ; et {reason}"
            else:
                merged[name] = reason
    return merged
