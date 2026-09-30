# benchmark_pipeline

Génération de benchmarks de questions géospatiales en langage naturel à partir
d'OpenStreetMap. Chaque question est produite avec sa vérité terrain — la liste
ordonnée des POIs qui y répondent — pour évaluer un modèle de recherche de
points d'intérêt.

Trois familles de générateurs :

| Famille | Où | Ce qu'elle produit |
|---|---|---|
| **géospatiale** | `benchmark_pipeline/generator/template_question/geospatial/` | 12 templates : proximité (`point_near`, `point_near_metric`), direction (`point_near_cardinal`, `point_towards`, `point_between`), zone (`area_inside`, `area_outside`, `area_border`, `area_direction`), rue (`street_along`, `street_cross`, `street_opposite_side`) |
| **sémantique** | `benchmark_pipeline/generator/template_question/semantic/` | questions sur les attributs d'un POI (cuisine, terrasse, horaires) |
| **composite** | `benchmark_pipeline/generator/template_question/composite/` | `make_question_geo`, qui croise les deux |

Chaque template géospatial suit le même contrat :
`make_question_*(df, …, nb_q, seed) -> DataFrame`, et est déclaré dans
`TEMPLATE_REGISTRY`
(`benchmark_pipeline/generator/template_question/schema.py`), qui sert à la fois
à l'enregistrement en lot et à la paramétrisation des tests.

---

## Prérequis

- **Python 3.11** (`requires-python = ">=3.11"`). Version de référence : 3.11.15.
- Un accès réseau pour les chargeurs OSM : ils interrogent l'API Overpass via
  `osmnx`. Les tests, eux, ne touchent pas au réseau.

## Installation

```bash
uv sync
```

`uv sync` crée un `.venv/` à la racine et y installe les dépendances de base
plus le groupe `dev` (pytest, ruff, ipykernel). Deux extras sont optionnels,
parce qu'ils sont lourds et ne servent qu'à une partie du code :

```bash
uv sync --extra viz    # contextily : fonds de carte dans dataviz_tools/
uv sync --extra nlp    # sentence-transformers : similarité dans data_cleaning/
```

### Dans VS Code

Rien d'autre à installer : `⇧⌘P` → **Python: Select Interpreter** → choisir
`./.venv/bin/python`. Les fichiers `.py` s'exécutent alors directement, le
panneau **Testing** découvre `pytest`, et `demo.ipynb` s'ouvre dans l'éditeur de
notebooks intégré — c'est à ça que sert `ipykernel` dans le groupe `dev`.

Jupyter Lab n'est **pas** installé (`jupyterlab` n'est pas une dépendance du
projet) : `uv run jupyter lab` échoue. Si tu le veux malgré tout, ajoute-le au
groupe `dev`, sinon reste sur VS Code.

```bash
python -c "
from benchmark_pipeline.config import BBOX
from benchmark_pipeline.generator.template_question.schema import TEMPLATE_REGISTRY
print(len(TEMPLATE_REGISTRY), 'templates,  bbox', BBOX)
"
# -> 12 templates,  bbox (2.24, 48.8, 2.41, 48.9)
```

---

## Lancer les tests

Depuis la racine du dépôt :

```bash
pytest
```

Sortie attendue :

```
145 passed, 24 xfailed
```

Quelques invocations utiles :

```bash
pytest tests/test_template_contract.py   # un fichier
pytest -k point_near                     # un template
pytest -rxX                              # détail des xfail/xpass
```

La configuration est dans `[tool.pytest.ini_options]` du `pyproject.toml`.
`pythonpath = ["."]` met la racine sur le `sys.path`, ce qui rend importables à
la fois `benchmark_pipeline` et `tests` — les tests s'importent entre eux
(`from tests.conftest import …`) et `tests/` n'a pas d'`__init__.py`.

### Lire le résultat : le rôle des `xfailed`

**Un `xfailed` n'est pas un test ignoré, c'est un défaut inventorié.** Les
défauts connus des générateurs sont recensés dans `tests/known_defects.py`,
chacun avec son fichier, sa ligne et sa mesure. Les tests qui les démontrent
portent un `xfail(strict=True)`, ce qui donne trois propriétés :

- la suite sort **verte** tant que les défauts connus le restent, donc elle est
  exploitable en intégration continue ;
- chaque défaut reste **visible et documenté** — `pytest -rxX` en affiche le
  motif complet ;
- `strict=True` fait échouer le test en **XPASS** le jour où le défaut est
  corrigé : le test dit alors « c'est réparé, retire le marqueur », au lieu de
  rester vert en silence et de laisser l'inventaire mentir.

Donc : `24 xfailed` = 24 défauts connus et décrits, pas 24 tests morts. Un
**XPASS** n'est pas une bonne nouvelle silencieuse, c'est une tâche : retirer
l'entrée de `known_defects.py`.

Les tests n'utilisent ni le réseau ni le cache OSM. Ils travaillent sur un monde
synthétique construit dans `tests/conftest.py` — une grille de POIs, de zones et
de rues en Lambert-93 (EPSG:2154) dont la vérité terrain est recalculable à la
main. C'est ce qui les rend déterministes et rapides (~20 s).

### Lint

```bash
ruff check .
```

---

## Lancer le pipeline

Le périmètre géographique et les tags OSM sont dans
`benchmark_pipeline/config.py` (`BBOX`, `CATEGORIES`, `STREETS_TAGS`,
`AREA_TAGS`). `BBOX` couvre Paris ; `BBOX_TEST` est un petit rectangle autour du
Marais, utile pour itérer vite.

```python
from benchmark_pipeline.loader.osm_loaders import load_pois, load_streets, load_area
from benchmark_pipeline.generator.template_question.geospatial.point_near import (
    make_question_point_near,
)
from benchmark_pipeline.utils.dataset_io import save_benchmark

df_osm = load_pois()          # réseau : Overpass via osmnx
df_streets = load_streets()
df_area = load_area()

bench = make_question_point_near(df_osm, nb_q=50, seed=42)
save_benchmark(bench, "point_near.parquet", df_osm=df_osm)
```

- **Le premier appel télécharge** : `load_pois` interroge Overpass et n'utilise
  pas de cache Parquet (`use_cache=False`). `osmnx` met en cache les réponses
  HTTP dans `cache/` (~70 Mo pour la bbox Paris), ignoré par git — un clone neuf
  retélécharge donc.
- `save_benchmark` / `save_benchmark_suite`
  (`benchmark_pipeline/utils/dataset_io.py`) écrivent dans
  `data/benchmarks/geospatial/` avec un fichier de métadonnées (commit git,
  versions des bibliothèques, schéma). `data/` est ignoré par git.
- `make_question_geo`
  (`benchmark_pipeline/generator/template_question/composite/generator_qcomposite.py`)
  combine questions géospatiales et sémantiques ; c'est le point d'entrée du
  benchmark complet, et il attend `df_osm`, `df_area` et `df_streets`.
- Le nettoyage des types de cuisine peut appeler `sentence-transformers`
  (`paraphrase-multilingual-MiniLM-L12-v2`, extra `nlp`) : premier appel =
  téléchargement du modèle.

`demo.ipynb` montre l'enchaînement bout en bout : ouvrez-le dans VS Code avec
l'interpréteur `./.venv/bin/python`.

---

## Organisation

```
benchmark_pipeline/
  config.py                      # bbox, tags OSM, features — le seul fichier à éditer
  loader/                        # extraction et nettoyage OSM
    osm_extractor.py             #   appels Overpass, cache
    osm_loaders.py               #   load_pois / load_streets / load_area
    data_cleaning/
  generator/
    features.py
    template_question/
      schema.py                  # TEMPLATE_REGISTRY : contrat de sortie des 12 templates
      registry.py                # registre par décorateur @template
      geospatial/                # les 12 générateurs
      semantic/  composite/  multihop/
  utils/dataset_io.py            # save_benchmark / load_benchmark + métadonnées
  dataviz_tools/                 # plot_question et fonds de carte
tests/
  conftest.py                    # monde synthétique + fixtures
  known_defects.py               # inventaire des défauts, motifs des xfail
  test_template_contract.py      # invariants structurels, tous templates
  test_template_semantics.py     # vérité terrain recalculée en numpy/shapely
  test_known_defects.py          # démonstration ciblée de chaque défaut
  test_dataset_io.py             # aller-retour disque, cohérence du registre
```

Deux registres coexistent : `registry.REGISTRY`, peuplé à l'import par le
décorateur `@template`, et `schema.TEMPLATE_REGISTRY`, écrit à la main parce
qu'il porte en plus les colonnes de contexte. `test_dataset_io.py` vérifie
qu'ils ne divergent pas.

---

## Limites connues

Ce qui surprendra quelqu'un qui reprend le dépôt :

- **`workflows/tests.yml` n'est pas au bon endroit** : GitHub Actions lit
  `.github/workflows/`. Le fichier n'est donc jamais exécuté. Son étape
  d'installation (`pip install -e ".[dev]"`) est par ailleurs périmée depuis le
  passage à `uv_build` et aux `[dependency-groups]` — c'est `uv sync`.
- **`extract_woosmap_pois` contient des chemins absolus** vers la machine de
  l'auteur (`/Users/sese/Documents/code/benchmark_NL_POI/…`). La fonction n'est
  pas utilisable ailleurs en l'état ; le chemin OSM (`extract_osm_pois`), lui,
  fonctionne partout.
- **Les défauts des générateurs sont inventoriés, pas corrigés.** Voir les 24
  `xfailed` et `tests/known_defects.py` : `area_direction` publie un schéma
  divergent, plusieurs templates ne respectent pas `nb_q`, `point_between` a un
  registre périmé. C'est délibéré — la suite les constate pour qu'on sache quoi
  réparer et dans quel ordre.
