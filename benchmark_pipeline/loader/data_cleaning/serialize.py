import pandas as pd

from benchmark_pipeline.config import NAME_TAGS, CATEGORY_TAGS


def _clean(v) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if not s:
        return None
    # fast_food -> fast food ; convenience;supermarket -> convenience supermarket
    s = s.replace("_", " ").replace(";", " ")
    return " ".join(s.split())

def serialize_poi(
    row: pd.Series,
    list_features,
    include_address: bool = True,
    include_context: bool = False,
    include_names: bool = False,
    context_col: str = "nearby_stations",
) -> str:
    """Transforme une ligne en une chaîne unique.
 
    include_context : injecte les noms des entités voisines (stations de métro,
        parcs) dans le document. À laisser à False par défaut — voir la note
        en bas de fichier, c'est la décision qui décide de la validité de la
        comparaison spatiale.
    """
    parts: list[str] = []
 
    # --- nom
    if include_names:
        name = next((c for t in NAME_TAGS if (c := _clean(row.get(t)))), None)
        if name:
            parts.append(name)
 
    # --- catégories : on garde TOUTES celles présentes, pas seulement la
    # première. Un lieu shop=bakery + amenity=cafe doit matcher les deux.
    cats = [c for t in CATEGORY_TAGS if (c := _clean(row.get(t)))]
    parts.extend(dict.fromkeys(cats))  # dédoublonne en gardant l'ordre
 
    # --- attributs sémantiques utiles en retrieval
    for t in list_features:
        if (v := _clean(row.get(t))) and v not in parts:
            if _clean(row.get(t))=="yes":
                parts.append(t.replace("_", " ")) #pour enlever les underscore du nom des colonnes pour le tf-idf

            elif _clean(row.get(t))=="no":
                continue
            else:
                parts.append(v)

 
    # --- adresse
    if include_address:
        street = _clean(row.get("addr:street"))
        if street:
            parts.append(street)
        pc = _clean(row.get("addr:postcode"))
        if pc:
            parts.append(pc)
 
    if include_context:
        if ctx := _clean(row.get(context_col)):
            parts.append(ctx)
 
    doc = " ".join(parts)
    return doc.lower()
