
"""
MADE BY AI

cuisine_classifier.py
======================
Nettoie le tag OSM `cuisine` (texte libre, style "japanese;sushi;bar") et en
extrait UNE cuisine propre par POI.
 
Idée de fond : le champ `cuisine` mélange 4 choses différentes.
    - regional : la vraie cuisine     -> japanese, french, italian...
    - dish     : un type de plat       -> sushi, pizza, burger, poke...
    - venue    : un type de lieu        -> coffee_shop, bar, bakery... (pas une cuisine)
    - diet     : un régime              -> vegan, halal, kosher...
On range chaque bout dans sa catégorie ("facette"), puis on choisit la plus
pertinente.
 
Les 3 étapes du pipeline :
    1. split_tags(...)        découpe et normalise le champ brut en tokens propres
    2. facet_of(...)          dit à quelle catégorie appartient un token
    3. pick_cuisine(...)      choisit UN token, par ordre de priorite
 
Fonction tout-en-un : classify_cuisine_column(df).
Dependance obligatoire : pandas. Fallbacks (rapidfuzz / sentence-transformers) optionnels.
"""
 
import re
import unicodedata
from collections import Counter
 
import pandas as pd
 
 
# ===========================================================================
#  1. VOCABULAIRES  (a completer au fur et a mesure avec unknown_report)
# ===========================================================================
 
# La vraie cuisine regionale/nationale. C'est le plus informatif.
REGIONAL = {
    "japanese", "chinese", "korean", "vietnamese", "thai", "indian", "pakistani",
    "indonesian", "filipino", "cambodian", "malaysian", "asian", "oriental",
    "french", "italian", "spanish", "portuguese", "greek", "german", "english",
    "swiss", "austrian", "belgian", "dutch", "hungarian", "polish", "russian",
    "balkan", "basque", "corsican", "savoyard", "alsatian", "mediterranean",
    "american", "mexican", "argentinian", "brazilian", "peruvian", "cuban",
    "caribbean", "tex-mex", "cajun",
    "lebanese", "turkish", "moroccan", "tunisian", "algerian", "syrian",
    "iranian", "afghan", "georgian", "armenian", "israeli", "african",
    "ethiopian", "senegalese",
}
 
# Un type de plat. Moins precis que le regional, mais reste une cuisine valable.
DISH = {
    "pizza", "pasta", "burger", "sandwich", "kebab", "tacos", "sushi", "ramen",
    "noodle", "poke", "bowl", "dumpling", "dim_sum", "curry", "seafood", "fish",
    "chicken", "steak_house", "grill", "barbecue", "rotisserie", "friture",
    "crepe", "galette", "tapas", "fondue", "raclette", "salad", "soup",
    "fish_and_chips", "hot_dog", "bagel", "wings", "empanada", "wok", "teppanyaki",
}
 
# Un type de LIEU qui a fuite dans le champ cuisine. Pas une cuisine regionale,
# mais utile pour le retrieval. Ignore par defaut, garde si keep_venue=True.
VENUE = {
    "coffee_shop", "cafe", "bar", "pub", "wine_bar", "beer", "brewery",
    "ice_cream", "bakery", "pastry", "chocolate", "donut", "cake", "dessert",
    "bubble_tea", "tea", "juice", "smoothie", "breakfast", "brunch", "snack",
    "fast_food", "bistro", "brasserie", "food_court", "diner", "buffet",
}
 
# Un regime alimentaire. A sa place dans diet:*, mais souvent mis dans cuisine.
DIET = {"vegan", "vegetarian", "halal", "kosher", "gluten_free", "organic", "bio"}
 
 
# Table de correspondance : variante/faute connue -> forme officielle.
# (Les accents sont deja retires plus loin, donc on ecrit "francaise" sans accent.)
SYNONYMS = {
    # variantes francaises
    "francaise": "french", "italienne": "italian", "italien": "italian",
    "chinoise": "chinese", "chinois": "chinese", "japonaise": "japanese",
    "japonais": "japanese", "indienne": "indian", "libanaise": "lebanese",
    "grecque": "greek", "vietnamienne": "vietnamese", "coreenne": "korean",
    "coreen": "korean", "marocaine": "moroccan", "espagnole": "spanish",
    "portugaise": "portuguese", "turque": "turkish",
    # doublons / alias anglais
    "sushi_bar": "sushi", "steak": "steak_house", "steakhouse": "steak_house",
    "bbq": "barbecue", "burgers": "burger", "hamburger": "burger",
    "noodles": "noodle", "pastas": "pasta", "pizzas": "pizza",
    "coffee": "coffee_shop", "coffeeshop": "coffee_shop", "cofee_shop": "coffee_shop",
    "icecream": "ice_cream", "glacier": "ice_cream", "boulangerie": "bakery",
    "patisserie": "pastry", "salon_de_the": "tea", "the": "tea",
    "veg": "vegetarian", "veggie": "vegetarian",
}
 
# Suffixes inutiles a couper : "japanese_restaurant" -> "japanese".
SUFFIXES_PARASITES = ("_restaurant", "_restaurants", "_food", "_cuisine", "_place", "_house")
 
# Separe les tokens du champ brut : gere ;  ,  /  |
SEPARATEURS = re.compile(r"[;,/|]")
 
 
# ===========================================================================
#  2. NORMALISATION D'UN TOKEN
# ===========================================================================
 
def enlever_accents(texte):
    """'francaise' <- 'francaise' (retire les accents)."""
    decompose = unicodedata.normalize("NFD", texte)
    return "".join(c for c in decompose if unicodedata.category(c) != "Mn")
 
 
def normalize_token(token):
    """
    Un morceau brut -> forme canonique propre.
    Ex : ' Japanese Restaurant ' -> 'japanese'
    """
    # a) minuscules + retrait des accents
    t = enlever_accents(token.strip().lower())
 
    # b) espaces et tirets deviennent des underscores, on jette le reste du bruit
    t = re.sub(r"[\s\-]+", "_", t)      # "coffee shop" -> "coffee_shop"
    t = re.sub(r"[^a-z0-9_]", "", t)    # supprime ponctuation, symboles...
    t = re.sub(r"_+", "_", t).strip("_")  # underscores en trop
 
    # c) on coupe un eventuel suffixe parasite ("_restaurant", "_food"...)
    for suffixe in SUFFIXES_PARASITES:
        if t.endswith(suffixe) and len(t) > len(suffixe):
            t = t[: -len(suffixe)]
            break
 
    # d) on remplace par la forme officielle si c'est une variante connue
    return SYNONYMS.get(t, t)
 
 
def split_tags(champ_brut):
    """
    Champ cuisine brut -> liste de tokens propres, sans doublon, ordre conserve.
    Ex : 'Japanese;sushi;JAPANESE' -> ['japanese', 'sushi']
    """
    # champ vide ou NaN -> liste vide
    if not isinstance(champ_brut, str) or not champ_brut.strip():
        return []
 
    tokens_propres = []
    for morceau in SEPARATEURS.split(champ_brut):
        token = normalize_token(morceau)
        # on ignore les vides et les doublons
        if token and token not in tokens_propres:
            tokens_propres.append(token)
    return tokens_propres
 
 
# ===========================================================================
#  3. A QUELLE FACETTE APPARTIENT UN TOKEN ?
# ===========================================================================
 
def facet_of(token):
    """Renvoie 'regional', 'dish', 'venue', 'diet' ou 'unknown'."""
    if token in REGIONAL:
        return "regional"
    if token in DISH:
        return "dish"
    if token in VENUE:
        return "venue"
    if token in DIET:
        return "diet"
    return "unknown"
 
 
def group_by_facet(tokens):
    """
    Range les tokens par facette.
    Ex : ['japanese', 'sushi', 'bar'] -> {'regional': ['japanese'],
                                          'dish': ['sushi'], 'venue': ['bar']}
    """
    groupes = {}
    for token in tokens:
        facette = facet_of(token)
        groupes.setdefault(facette, []).append(token)
    return groupes
 
 
# ===========================================================================
#  4. CHOISIR UNE SEULE CUISINE
# ===========================================================================
 
def pick_cuisine(tokens, keep_venue=False, prefer="regional"):
    """
    Choisit UN token parmi la liste, par ordre de priorite de facette.
 
    prefer="regional" (defaut) : 'japanese;bowl;poke' -> 'japanese'
    prefer="dish"              : 'japanese;bowl;poke' -> 'bowl'
    keep_venue=True            : autorise 'coffee_shop'/'bar' comme repli.
 
    Renvoie None si rien de pertinent (ex : uniquement un venue en mode strict).
    """
    # ordre dans lequel on regarde les facettes
    if prefer == "dish":
        ordre_priorite = ["dish", "regional"]
    else:
        ordre_priorite = ["regional", "dish"]
    if keep_venue:
        ordre_priorite.append("venue")
 
    # on prend le premier token de la premiere facette non vide
    for facette in ordre_priorite:
        for token in tokens:
            if facet_of(token) == facette:
                return token
    return None
 
 
# ===========================================================================
#  5. FONCTION PRINCIPALE : nettoyer toute une colonne
# ===========================================================================
 
def classify_cuisine_column(df, col="cuisine", keep_venue=False, prefer="regional",
                            extra_synonyms=None):
    """
    Ajoute 3 colonnes au DataFrame :
        <col>_tokens : liste des tokens propres          (pour verifier)
        <col>_facets : tokens ranges par facette          (pour verifier)
        <col>_clean  : LA cuisine retenue (str ou None)   <-- le resultat utile
 
    extra_synonyms : dict de synonymes a ajouter avant de classer
                     (ex : le resultat valide de build_unknown_map).
    """
    # on peut enrichir la table de synonymes a la volee
    if extra_synonyms:
        for variante, forme_officielle in extra_synonyms.items():
            SYNONYMS[normalize_token(variante)] = forme_officielle
 
    resultat = df.copy()
    resultat[f"{col}_tokens"] = resultat[col].apply(split_tags)
    resultat[f"{col}_facets"] = resultat[f"{col}_tokens"].apply(group_by_facet)
    resultat[f"{col}"] = resultat[f"{col}_tokens"].apply(
        lambda tokens: pick_cuisine(tokens, keep_venue=keep_venue, prefer=prefer)
    )
    resultat[f"{col}"] = resultat[f"{col}"].str.replace("_", "", regex=False)
    return resultat
 
 
# ===========================================================================
#  6. RAPPORTS POUR INSPECTER LE RESULTAT
# ===========================================================================
 
def unknown_report(df, col="cuisine"):
    """
    Liste les tokens non reconnus, tries par frequence.
    C'est ce qui te dit quoi ajouter dans REGIONAL / DISH / SYNONYMS.
    """
    colonne_tokens = f"{col}_tokens"
    if colonne_tokens not in df:
        df = classify_cuisine_column(df, col=col)
 
    compteur = Counter()
    for tokens in df[colonne_tokens]:
        for token in tokens:
            if facet_of(token) == "unknown":
                compteur[token] += 1
 
    return pd.DataFrame(compteur.most_common(), columns=["token", "count"])
 
 
def coverage_report(df, col="cuisine"):
    """Combien de POIs ont fini avec une cuisine, combien sont restes a None."""
    colonne_clean = f"{col}_clean"
    if colonne_clean not in df:
        df = classify_cuisine_column(df, col=col)
 
    total = len(df)
    avec_cuisine = df[colonne_clean].notna().sum()
    return pd.Series({
        "n_pois": total,
        "avec_cuisine": avec_cuisine,
        "sans_cuisine_(None)": total - avec_cuisine,
        "taux_couverture": round(avec_cuisine / total, 3) if total else 0.0,
    })
 
 
# ===========================================================================
#  7. FALLBACK OPTIONNEL : proposer un mapping pour les tokens inconnus
# ===========================================================================

 
 
def _proposer_embedding(suspects, canoniques, seuil, model_name):
    """Rapproche par sens (gere mieux le FR/EN, ex 'japonaise' -> 'japanese')."""
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer(model_name)
    emb_canon = model.encode(canoniques, convert_to_tensor=True, normalize_embeddings=True)
    emb_suspect = model.encode(suspects, convert_to_tensor=True, normalize_embeddings=True)
 
    similarites = util.cos_sim(emb_suspect, emb_canon)
    meilleurs_index = similarites.argmax(dim=1)
    meilleurs_scores = similarites.max(dim=1).values
 
    lignes = []
    for i, suspect in enumerate(suspects):
        score = float(meilleurs_scores[i])
        match = canoniques[int(meilleurs_index[i])]
        if score >= seuil:
            lignes.append((suspect, match, facet_of(match), round(score, 3)))
        else:
            lignes.append((suspect, None, "unknown", round(score, 3)))
    return lignes
 
 
def build_unknown_map(unknowns, method="embedding", threshold=None,
                      model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    """
    Propose un mapping {token_inconnu -> token_officiel} A VALIDER a la main.
    Rien n'est applique automatiquement : l'outil propose, tu decides.
 
    method="fuzzy"     : ressemblance de chaine  (seuil par defaut 88, sur 100)
    method="embedding" : proximite de sens        (seuil par defaut 0.75)
 
    Renvoie un DataFrame [suspect, proposed, facet, score] trie par score.
    Ensuite : garder les bonnes lignes, en faire un dict, le passer a
    classify_cuisine_column(..., extra_synonyms=mon_dict).
    """
    canoniques = sorted(REGIONAL | DISH | VENUE | DIET)
    suspects = sorted(set(unknowns))
    if not suspects:
        return pd.DataFrame(columns=["suspect", "proposed", "facet", "score"])
 
    elif method == "embedding":
        seuil = 0.75 if threshold is None else threshold
        lignes = _proposer_embedding(suspects, canoniques, seuil, model_name)
    else:
        raise ValueError("method doit etre 'fuzzy' ou 'embedding'")
 
    tableau = pd.DataFrame(lignes, columns=["suspect", "proposed", "facet", "score"])
    return tableau.sort_values("score", ascending=False, ignore_index=True)