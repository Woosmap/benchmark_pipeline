"""Retire les TODO périmés de la cellule 7 et les remplace par la note associée."""
import json

NB = "question_typeA.ipynb"

OLD_BODY = """    L'azimut de chaque POI est mesuré en degrés horaires depuis le nord — d'où
    l'ordre `atan2(dx, dy)`, inversé par rapport à la convention mathématique. Un
    POI est retenu si son écart angulaire à la direction visée, ramené dans
    (-180°, 180°], ne dépasse pas `half_width`.
"""

NEW_BODY = """    L'azimut de chaque POI est mesuré dans le plan Lambert-93, en degrés horaires
    depuis le nord — d'où l'ordre `atan2(dx, dy)`, inversé par rapport à la
    convention mathématique. Un POI est retenu si son écart angulaire à la
    direction visée, ramené dans (-180°, 180°], ne dépasse pas `half_width`.
"""

OLD_TODO = """
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

NEW_NOTE = """
    Note:
        Distance euclidienne et azimut planaire, sans correction de latitude :
        c'est la forme correcte en CRS projeté. Le nord de la grille Lambert-93
        s'écarte du nord géographique d'environ 0,5° à Paris (convergence des
        méridiens), négligeable devant `half_width`.

    TODO: une connexion DuckDB jetable est ouverte à chaque appel quand `con`
        vaut None, et `INSTALL spatial` est rejoué 440 fois sur une génération
        complète. Passer une connexion unique depuis l'appelant.
"""

OLD_LABEL_TODO = """    TODO: le libellé `f"{cat_q} at {d} meters from ..."` insère une direction là
        où il annonce une distance ; écrire `f"{cat_q} to the {d} of ..."`.
"""

nb = json.load(open(NB))
cell = nb["cells"][7]
src = "".join(cell["source"])

for old, new in [(OLD_BODY, NEW_BODY), (OLD_TODO, NEW_NOTE), (OLD_LABEL_TODO, "")]:
    assert src.count(old) == 1, f"motif introuvable ou ambigu :\n{old[:80]}"
    src = src.replace(old, new)

cell["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
json.dump(nb, open(NB, "w"), ensure_ascii=False, indent=1)
print("cellule 7 : TODO périmés retirés, note ajoutée")
