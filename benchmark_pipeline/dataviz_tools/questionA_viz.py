"""Visualisation d'une question du benchmark POI.

Corrections par rapport à la version précédente :
  - les polygones d'ancrage ne sont plus tracés comme des lignes (aplat rouge
    qui masquait le fond) mais en contour seul ;
  - la légende est construite à partir de ce qui est réellement tracé ;
  - la couleur encode le rang *relatif* aux résultats affichés, l'étiquette
    garde le rang corpus, pour que la structure ne soit plus écrasée par les
    outliers ;
  - le paramètre `basemap` est respecté ;
  - les invariants (listes parallèles, unicité de poi_id) sont vérifiés.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import Point, LineString

# Rampe séquentielle une seule teinte : rang 1 = foncé, dernier rang = clair.
# Tronquée à 0.30 pour que les derniers rangs restent visibles sur fond blanc.
_BLUES = plt.get_cmap("Blues")
RANK_CMAP = LinearSegmentedColormap.from_list(
    "rank", _BLUES(np.linspace(0.95, 0.30, 256))
)
ANCHOR_RED = "#D62728"

_POLY_TYPES = ("Polygon", "MultiPolygon")
_POINT_TYPES = ("Point", "MultiPoint")


# --------------------------------------------------------------------------- #
# géométries d'ancrage
# --------------------------------------------------------------------------- #

def anchor_geoms(row):
    """Sépare les géométries d'ancrage de la question en points/lignes/polygones.

    Lit ce qui est présent dans la ligne : anchor_x/y, point_b_x/y, et toute
    colonne `*_geometry` (area_geometry, street_geometry, street_a/b_geometry,
    poi_y_geometry...). Le tri par type est nécessaire parce que matplotlib
    remplit la face d'un Polygon là où il ne trace qu'un trait pour une
    LineString.

    Returns
    -------
    (points, lines, polygons) : trois listes de géométries shapely
    """
    pts, lines, polys = [], [], []

    if "anchor_x" in row.index and pd.notna(row.get("anchor_x")):
        pts.append(Point(row["anchor_x"], row["anchor_y"]))

    if "point_b_x" in row.index and pd.notna(row.get("point_b_x")):
        b = Point(row["point_b_x"], row["point_b_y"])
        if pts:
            lines.append(LineString([pts[0], b]))      # segment A→B
        pts.append(b)

    for col in row.index:
        if not col.endswith("_geometry"):
            continue
        g = row[col]
        if g is None or (hasattr(g, "is_empty") and g.is_empty):
            continue
        if g.geom_type in _POINT_TYPES:
            pts.append(g)
        elif g.geom_type in _POLY_TYPES:
            polys.append(g)
        else:
            lines.append(g)

    # carrefour : matérialise le point de croisement
    if {"street_a_geometry", "street_b_geometry"} <= set(row.index):
        a, b = row["street_a_geometry"], row["street_b_geometry"]
        if a is not None and b is not None:
            inter = a.intersection(b)
            if not inter.is_empty:
                pts.append(inter)

    return pts, lines, polys


# --------------------------------------------------------------------------- #
# carte d'une question
# --------------------------------------------------------------------------- #

def plot_question(row, df_osm, ax=None, label_top=3, basemap=True, pad=150,
                  color_by="local"):
    """Carte d'une question : résultats en dégradé par rang, ancre en rouge.

    Parameters
    ----------
    row : pd.Series
        Une ligne du benchmark (`bench.loc[i]`).
    df_osm : gpd.GeoDataFrame
        Doit contenir une colonne `poi_id` et une géométrie projetée.
    label_top : int
        Nombre de meilleurs rangs à étiqueter directement.
    basemap : bool
        Ajoute le fond CartoDB (nécessite contextily et un accès réseau).
    color_by : {"local", "corpus"}
        "local" colore par le rang parmi les résultats affichés — lisible même
        avec un outlier à 14000. "corpus" colore par le rang brut.
    """
    if isinstance(row, (int, np.integer)):
        raise TypeError("passe une ligne, pas un entier : bench.loc[i]")

    ids = list(row["results_poi_id"])
    ranks = list(row["results_poi_rank"])
    names = list(row["results_poi_name"])

    if not ids:
        raise ValueError(f"aucun résultat pour : {row['query']}")
    if not len(ids) == len(ranks) == len(names):
        raise ValueError(
            f"listes non parallèles : {len(ids)} ids, {len(ranks)} rangs, "
            f"{len(names)} noms — l'appariement id↔rang est cassé en amont"
        )
    if len(set(ranks)) != len(ranks):
        raise ValueError("rangs dupliqués : vérifie l'argsort qui les produit")

    geo = df_osm.set_index("poi_id")
    if not geo.index.is_unique:
        raise ValueError("poi_id non unique dans df_osm : .loc réordonne et duplique")
    missing = [i for i in ids if i not in geo.index]
    if missing:
        raise KeyError(f"{len(missing)} poi_id absents de df_osm, ex. {missing[:3]}")

    res = gpd.GeoDataFrame(
        {"rank": ranks, "name": names},
        geometry=list(geo.loc[ids, "geometry"]),
        crs=df_osm.crs,
    )
    res["rank_local"] = res["rank"].rank(method="first").astype(int)

    pts, lines, polys = anchor_geoms(row)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 9))

    # 1. ancre : polygones en contour, puis lignes, puis points au-dessus
    if polys:
        gpd.GeoSeries(polys, crs=df_osm.crs).plot(
            ax=ax, facecolor="none", edgecolor=ANCHOR_RED,
            linewidth=2, zorder=2,
        )
    if lines:
        gpd.GeoSeries(lines, crs=df_osm.crs).plot(
            ax=ax, color=ANCHOR_RED, linewidth=2, zorder=3,
        )
    if pts:
        gpd.GeoSeries(pts, crs=df_osm.crs).plot(
            ax=ax, color=ANCHOR_RED, marker="*", markersize=380,
            edgecolor="white", linewidth=1.2, zorder=5,
        )

    # 2. résultats : anneau blanc pour séparer les marques qui se recouvrent
    col = "rank_local" if color_by == "local" else "rank"
    vmax = max(int(res[col].max()), 2)          # évite vmin == vmax sur 1 résultat
    res.plot(
        ax=ax, column=col, cmap=RANK_CMAP, vmin=1, vmax=vmax,
        markersize=70, edgecolor="white", linewidth=1.0, zorder=4,
    )

    # 3. étiquettes directes sur les tout premiers rangs seulement
    for r in res.nsmallest(label_top, "rank").itertuples():
        ax.annotate(
            f"{r.rank}. {r.name}", (r.geometry.x, r.geometry.y),
            xytext=(7, 5), textcoords="offset points",
            fontsize=9, color="#333333", zorder=6,
        )

    # 4. cadrage sur l'ensemble ancre + résultats
    everything = list(res.geometry) + pts + lines + polys
    b = gpd.GeoSeries(everything, crs=df_osm.crs).total_bounds
    ax.set_xlim(b[0] - pad, b[2] + pad)
    ax.set_ylim(b[1] - pad, b[3] + pad)

    if basemap:
        try:
            import contextily as cx
            cx.add_basemap(
                ax, crs=df_osm.crs,
                source=cx.providers.CartoDB.PositronNoLabels,
            )
        except Exception as e:                  # réseau absent, clé, quota...
            print(f"[plot_question] fond de carte ignoré : {e}")

    # 5. barre de couleur
    label = ("rang parmi les résultats (1 = plus pertinent)"
             if color_by == "local" else "rang corpus (1 = plus pertinent)")
    sm = plt.cm.ScalarMappable(cmap=RANK_CMAP, norm=plt.Normalize(vmin=1, vmax=vmax))
    cb = ax.figure.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label(label)
    cb.ax.invert_yaxis()                        # rang 1 en haut, comme un classement

    # 6. légende construite d'après ce qui est réellement tracé
    handles = []
    if pts:
        handles.append(Line2D([], [], marker="*", color="none",
                              markerfacecolor=ANCHOR_RED, markeredgecolor="white",
                              markersize=16, label="ancre"))
    if lines:
        handles.append(Line2D([], [], color=ANCHOR_RED, linewidth=2, label="axe"))
    if polys:
        handles.append(Patch(facecolor="none", edgecolor=ANCHOR_RED,
                             linewidth=2, label="zone"))
    if handles:
        ax.legend(handles=handles, loc="upper left", frameon=False)

    n = len(ids)
    ax.set_title(
        f"{row['query']}\n{row['function']} — {n} résultat{'s' if n > 1 else ''}"
        f" — rang corpus {min(ranks)}–{max(ranks)}",
        fontsize=11, loc="left",
    )
    ax.set_axis_off()
    return ax


# --------------------------------------------------------------------------- #
# planche de contrôle : un exemple par template
# --------------------------------------------------------------------------- #

def plot_all_templates(bench, df_osm, seed=42, basemap=True, ncols=3):
    """Un exemple tiré au hasard par valeur de `function`, en grille."""
    first = bench.groupby("function").sample(1, random_state=seed)
    n = len(first)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, row) in zip(axes, first.iterrows()):
        try:
            plot_question(row, df_osm, ax=ax, label_top=1, basemap=basemap)
        except Exception as e:
            ax.text(0.5, 0.5, f"{row['function']}\n{type(e).__name__}: {e}",
                    ha="center", va="center", fontsize=8, wrap=True)
            ax.set_axis_off()

    for ax in axes[n:]:
        ax.set_axis_off()

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# diagnostic : le gradient directionnel est-il appris ?
# --------------------------------------------------------------------------- #

def direction_rank_correlation(row, df_osm, direction=None):
    """Spearman entre le rang modèle et le gradient directionnel attendu.

    Renvoie un rho dans [-1, 1] : proche de 1 le modèle a bien ordonné selon la
    direction, proche de 0 il l'ignore. À comparer entre `north` et `south` sur
    la même zone — un rho stable au changement de token veut dire que la tour
    requête n'utilise pas le token du tout.
    """
    from scipy.stats import spearmanr

    direction = direction or row.get("direction")
    axis, sign = {"north": ("y", 1), "south": ("y", -1),
                  "east": ("x", 1), "west": ("x", -1)}[direction]

    geo = df_osm.set_index("poi_id")
    g = geo.loc[list(row["results_poi_id"]), "geometry"]
    coord = sign * np.array([getattr(p, axis) for p in g])

    rho, p = spearmanr(-np.array(row["results_poi_rank"]), coord)
    return rho, p

#plot_question(bench4.loc[50], df_osm, basemap=True)