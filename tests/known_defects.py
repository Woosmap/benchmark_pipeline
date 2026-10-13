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
#: Vide, et plus seulement de justesse : `area_inside` et `area_direction`
#: levaient dès que le corpus de zones comptait moins de `nb_q` entrées. Les
#: deux plafonnent désormais la taille de leur échantillon au corpus, et
#: `test_area_templates_survive_a_small_area_corpus` garde la correction.
BROKEN_TEMPLATES = {}

#: Templates qui tournent mais dont la sortie viole le schéma du benchmark.
#:
#: Vide. `area_border` ne filtrait pas par catégorie alors que son énoncé en
#: annonçait une, et `area_direction` publiait `results_poi_x`/`results_poi_y`
#: au lieu de `results_poi_dist` — la concaténation en un benchmark unique
#: perdait la colonne, et tout l'outillage qui la lit (`plot_question`, nDCG)
#: ne trouvait rien. Les deux sont corrigés.
INCOHERENT_TEMPLATES = {}

#: Templates auxquels il manque une des quatre listes de vérité terrain.
#: **Sous-ensemble strict d'`INCOHERENT_TEMPLATES`**, donc vide tant que
#: celle-ci l'est.
INCOMPLETE_RESULT_LISTS = {}

#: Templates dont le registre annonce des colonnes de contexte que le
#: générateur ne publie pas (ou plus).
#:
#: Famille distincte d'`INCOHERENT_TEMPLATES` : la divergence est entre le
#: *registre* et le module, pas dans la sortie elle-même. Seuls
#: `test_declared_columns_present` et `test_query_mentions_its_context` lisent
#: cette table.
#:
#: Vide : les trois causes qu'elle recensait ont toutes été corrigées dans
#: `schema.py`, qui était bien le périmé des deux côtés.
#:
#: 1. `category_query`, exigée partout par `BENCHMARK_CORE_COLUMNS` alors que
#:    neuf templates avaient cessé de la publier — devenue facultative
#:    (`OPTIONAL_CONTEXT_COLUMNS`) pour les trois qui filtrent par catégorie ;
#: 2. `same_cat`, annoncée par `_ANCHOR_COLS`, que plus aucun ne produisait —
#:    retirée ;
#: 3. le vocabulaire du second point : le registre annonçait `point_b_*` quand
#:    `point_towards` publie `anchor_b_*` et `point_between` `anchor_a_*` /
#:    `anchor_b_*`. `_POINT_B_COLS` a laissé place à `_ANCHOR_A_COLS` et
#:    `_ANCHOR_B_COLS`, et `point_between` étiquette ses questions par
#:    `anchor_a_name` — son `label_column` ne désignait aucune colonne, ce qui
#:    faisait lever `test_query_mentions_its_context` au lieu de comparer.
MISDECLARED_CONTEXT_COLUMNS = {}

#: Templates dont la colonne désignée par `label_column` est absente.
#: **Sous-ensemble strict de `MISDECLARED_CONTEXT_COLUMNS`**, donc vide tant que
#: celle-ci l'est.
MISSING_LABEL_COLUMN = {}

#: Templates dont la colonne `function` nomme une fonction absente du module.
#:
#: Vide : les quatre chaînes périmées ont été alignées sur les callables réels
#: — `cardinal_azimuth_sql` → `pack_by_direction`, `border_area` →
#: `pois_near_border`, `direction_area` → `segregate_pois` — et `area_inside`,
#: dont le calcul était en ligne et ne nommait donc rien, a vu son helper
#: extrait sous le nom `pois_inside_area`. Les branches correspondantes de
#: `targets.ellypses.get_ellypse` ont été renommées dans le même mouvement :
#: ces chaînes y servent de clés d'aiguillage, et les séparer casserait la
#: cible de tous les benchmarks du template concerné.
STALE_FUNCTION_NAMES = {}

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
#:
#: Vide. `point_near_metric` publiait ses quatre rayons sans contrôle — un
#: rayon de 100 m autour d'une ancre isolée ne trouve rien — et
#: `point_near_cardinal` ses quatre secteurs de même. Les deux sautent
#: désormais la question plutôt que de la livrer vide : un modèle ne peut pas
#: être noté sur une question qui n'a pas de bonne réponse.
UNANSWERABLE_TEMPLATES = {}

#: Défauts qui n'altèrent pas la cohérence d'une question prise isolément, mais
#: le volume ou la robustesse du jeu produit. Vérifiés un par un dans
#: `test_known_defects.py`.
SEMANTIC_DEFECTS = {}


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
