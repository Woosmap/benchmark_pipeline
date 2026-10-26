from functools import lru_cache

import mlx.core as mx
import pandas as pd
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

_load = lru_cache(maxsize=1)(load)

CONTEXT = "Paraphrase this spatial search query. Keep place names, place category and spatial relation unchanged. Answer with the paraphrase only."

MODELE_DEFAUT = "Qwen/Qwen2-1.5B-Instruct-MLX"

TAILLE_LOT = 128


def _prompt(tok, query: str) -> list[int]:
    return tok.apply_chat_template(
        [{"role": "system", "content": CONTEXT}, {"role": "user", "content": query}],
        add_generation_prompt=True,
    )

def paraphrases_par_lot(
        queries,
        nb_generated: int = 100,
        model_name: str = MODELE_DEFAUT,
        seed: int = 0,
        temperature: float = 0.5,
        taille_lot: int = TAILLE_LOT,
    ) -> list[list[str]]:
    """Paraphrase plusieurs énoncés d'un coup.

    Args:
        queries: énoncés à paraphraser.
        nb_generated: paraphrases produites par énoncé.
        model_name: modèle MLX, chargé une fois et gardé en cache.
        seed: graine du tirage.
        temperature: température d'échantillonnage.
        taille_lot: prompts envoyés ensemble au modèle.

    Returns:
        Une liste de `nb_generated` paraphrases par énoncé, dans l'ordre de `queries`.
    """
    queries = list(queries)
    if not queries:
        return []
    model, tok = _load(model_name)
    mx.random.seed(seed)
    sampler = make_sampler(temp=temperature)
    print('sampler')
    prompts = [_prompt(tok, query) for query in queries for _ in range(nb_generated)]
    print("prompts")
    textes = []
    i=0
    print(len(prompts)//taille_lot)
    for debut in range(0, len(prompts), taille_lot):
        print(i)
        reponse = batch_generate(model, tok, prompts=prompts[debut:debut + taille_lot],
                                 max_tokens=64, sampler=sampler)
        textes += [texte.strip() for texte in reponse.texts]
        i+=1
    return [textes[i:i + nb_generated] for i in range(0, len(textes), nb_generated)]


def make_paraphrase(
        dataset,
        key_query: str = "query_texte",
        nb_generated: int = 100,
        keep_original: bool = True,
        model_name: str = MODELE_DEFAUT,
        seed: int = 0,
        temperature: float = 0.5,
        ):
    """Remplace chaque question par ses paraphrases, une ligne chacune.

    Args:
        dataset: questions à paraphraser.
        key_query: colonne portant l'énoncé.
        nb_generated: paraphrases par énoncé.
        keep_original: conserve aussi les énoncés d'origine, marqués `is_paraphrase=False`.

    Returns:
        DataFrame avec `original_query` et `is_paraphrase` en plus des colonnes d'entrée.
    """
    out = dataset.copy()
    out["original_query"] = out[key_query]
    out[key_query] = paraphrases_par_lot(out[key_query], nb_generated=nb_generated,
                                         model_name=model_name, seed=seed,
                                         temperature=temperature)
    out = out.explode(key_query, ignore_index=True)
    out["is_paraphrase"] = True
    if keep_original:
        orig = dataset.assign(original_query=dataset[key_query], is_paraphrase=False)
        out = pd.concat([orig, out], ignore_index=True)
    return out
