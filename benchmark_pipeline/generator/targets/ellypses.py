import math

QUESTION_TYPE = ['NEAR', 'VERY_NEAR', 'CARDINAL_POINT']

def get_ellypse(row):
    """Paramètres de l'ellipse couvrant la zone de réponse d'une question.

    Le type est lu dans la colonne `function`, et la géométrie dans les colonnes
    que le template correspondant a produites.

    Args:
        row (pd.Series): Ligne de `df_question`.

    Returns:
        tuple[float, float, float, float, float]: décalage x et y du centre par
            rapport au point de référence de la question, demi-axe u, demi-axe v,
            inclinaison de u en degrés trigonométriques.

    Raises:
        ValueError: valeur de `function` inconnue.
    """
    # Longueurs en mètres Lambert-93. À inclinaison nulle, u porte les x et v les y.
    # Le point de référence dépend du type : l'ancre pour les questions de point, le
    # milieu de A-B pour `between`, le centroïde de la rue pour `along`, le
    # croisement pour `cross`, le POI de référence pour `opposite_side`, le
    # centroïde de la zone pour les `*_area`.
    def azimut_et_longueur(geom):
        ligne = geom.geoms[0] if geom.geom_type == "MultiLineString" else geom
        (x0, y0), (x1, y1) = ligne.coords[0], ligne.coords[-1]
        return math.degrees(math.atan2(y1 - y0, x1 - x0)), ligne.length

    def demi_emprise(geom):
        minx, miny, maxx, maxy = geom.bounds
        return (maxx - minx) / 2, (maxy - miny) / 2

    def ellipse_englobante(points):
        """Ellipse orientée selon l'axe principal d'un nuage de points.

        Args:
            points (list[tuple[float, float]]): Sommets, en Lambert-93.

        Returns:
            tuple: centre x, centre y, demi-axe u, demi-axe v, inclinaison en degrés.
        """
        n = len(points)
        moyenne_x = sum(x for x, _ in points) / n
        moyenne_y = sum(y for _, y in points) / n
        # Covariance 2x2, puis son vecteur propre dominant par la forme fermée :
        # l'ordre des tronçons d'une MultiLineString n'intervient nulle part.
        var_x = sum((x - moyenne_x) ** 2 for x, _ in points) / n
        var_y = sum((y - moyenne_y) ** 2 for _, y in points) / n
        cov = sum((x - moyenne_x) * (y - moyenne_y) for x, y in points) / n
        angle = 0.5 * math.atan2(2 * cov, var_x - var_y)

        projections_u = [(x - moyenne_x) * math.cos(angle) + (y - moyenne_y) * math.sin(angle)
                         for x, y in points]
        projections_v = [-(x - moyenne_x) * math.sin(angle) + (y - moyenne_y) * math.cos(angle)
                         for x, y in points]
        demi_u = (max(projections_u) - min(projections_u)) / 2
        demi_v = (max(projections_v) - min(projections_v)) / 2
        return moyenne_x, moyenne_y, demi_u, demi_v, math.degrees(angle)

    fonction = row["function"]

    if fonction == "near_sql":
        # k plus proches voisins, sans rayon : disque isotrope autour de l'ancre.
        return (0.0, 0.0, 600.0, 600.0, 0.0)

    if fonction == "near_metric_sql":
        # Le rayon est dans l'énoncé : l'ellipse est exactement le disque demandé.
        rayon = float(row["distance"])
        return (0.0, 0.0, rayon, rayon, 0.0)

    if fonction == "cardinal_azimuth_sql":
        dx, dy, inclinaison = {"north": (0.0, 500.0, 90.0),
                               "south": (0.0, -500.0, 90.0),
                               "east": (500.0, 0.0, 0.0),
                               "west": (-500.0, 0.0, 0.0)}[row["direction"]]
        return (dx, dy, 500.0, 300.0, inclinaison)

    if fonction == "towards_b_sql":
        angle = math.atan2(row["anchor_b_y"] - row["anchor_y"],
                           row["anchor_b_x"] - row["anchor_x"])
        return (500.0 * math.cos(angle), 300.0 * math.sin(angle),
                500.0, 300.0, math.degrees(angle))

    if fonction == "between_ab_sql":
        dx_ab = row["anchor_b_x"] - row["anchor_a_x"]
        dy_ab = row["anchor_b_y"] - row["anchor_a_y"]
        return (0.0, 0.0, math.hypot(dx_ab, dy_ab) / 2, 200.0,
                math.degrees(math.atan2(dy_ab, dx_ab)))

    if fonction == "along_street":
        rue = row["street_geometry"]
        parties = list(rue.geoms) if rue.geom_type == "MultiLineString" else [rue]
        sommets = [point for partie in parties for point in partie.coords]
        centre_x, centre_y, demi_u, _, inclinaison = ellipse_englobante(sommets)
        centroide = rue.centroid
        # Le petit axe reste la largeur de bande utile, pas la sinuosité de la rue.
        return (centre_x - centroide.x, centre_y - centroide.y, demi_u, 100.0, inclinaison)

    if fonction == "cross_streets":
        return (0.0, 0.0, 200.0, 200.0, 0.0)

    if fonction == "opposite_side":
        rue = row["street_geometry"]
        azimut, _ = azimut_et_longueur(rue)
        ligne = rue.geoms[0] if rue.geom_type == "MultiLineString" else rue
        poi_y = row["poi_y_geometry"]
        projete = ligne.interpolate(ligne.project(poi_y))
        normale = math.radians(azimut) + math.pi / 2
        # Signe du POI de référence sur la normale : on vise l'autre rive.
        cote = ((poi_y.x - projete.x) * math.cos(normale)
                + (poi_y.y - projete.y) * math.sin(normale))
        signe = -1.0 if cote >= 0 else 1.0
        return (signe * 40.0 * math.cos(normale), signe * 40.0 * math.sin(normale),
                80.0, 40.0, azimut)

    if fonction == "inside_area":
        demi_largeur, demi_hauteur = demi_emprise(row["area_geometry"])
        return (0.0, 0.0, demi_largeur, demi_hauteur, 0.0)

    if fonction == "direction_area":
        demi_largeur, demi_hauteur = demi_emprise(row["area_geometry"])
        dx, dy, demi_u, demi_v = {
            "north": (0.0, demi_hauteur / 2, demi_largeur, demi_hauteur / 2),
            "south": (0.0, -demi_hauteur / 2, demi_largeur, demi_hauteur / 2),
            "east": (demi_largeur / 2, 0.0, demi_largeur / 2, demi_hauteur),
            "west": (-demi_largeur / 2, 0.0, demi_largeur / 2, demi_hauteur),
        }[row["direction"]]
        return (dx, dy, demi_u, demi_v, 0.0)

    if fonction in ("border_area", "outside_area"):
        demi_largeur, demi_hauteur = demi_emprise(row["area_geometry"])
        marge = 1.2 if fonction == "border_area" else 2.0
        return (0.0, 0.0, demi_largeur * marge, demi_hauteur * marge, 0.0)

    raise ValueError(f"Type de question inconnu : {fonction}")

def add_ellypse(df_question):
    # `axis=1` : sans lui, apply passe les colonnes et non les lignes.
    df_question["ellypse"] = df_question.apply(get_ellypse, axis=1)
    return df_question