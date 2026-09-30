# benchmark_pipeline

Génération de benchmarks de questions géospatiales en langage naturel à partir
d'OpenStreetMap. Chaque question est produite avec sa vérité terrain — la liste
ordonnée des POIs qui y répondent — pour évaluer un modèle de recherche de
points d'intérêt.

Trois familles de générateurs :

| Famille | Où | Ce qu'elle produit |
|---|---|---|
| **géospatiale** | `src/generator/template_question/geospatial/` | 12 templates : proximité (`point_near`, `point_near_metric`), direction (`point_near_cardinal`, `point_towards`, `point_between`), zone (`area_inside`, `area_outside`, `area_border`, `area_direction`), rue (`street_along`, `street_cross`, `street_opposite_side`) |
| **sémantique** | `src/generator/template_question/semantic/` | questions sur les attributs d'un POI (cuisine, terrasse, horaires) |
| **composite** | `src/generator/template_question/composite/` | `make_question_geo`, qui croise les deux |

Chaque template géospatial suit le même contrat :
`make_question_*(df, …, nb_q, seed) -> DataFrame`, et est déclaré dans
`TEMPLATE_REGISTRY` (`src/generator/template_question/schema.py`), qui sert à la
fois à l'enregistrement en lot et à la paramétrisation des tests.

---

## Prérequis

- **Python 3.11** (`requires-python = ">=3.11"`). Version de référence : 3.11.15.
- Un accès réseau pour les chargeurs OSM : ils interrogent l'API Overpass via
  `osmnx`. Les tests, eux, ne touchent pas au réseau.
- Les dépendances lourdes (`geopandas`, `shapely`, `pyproj`, `duckdb`, `osmnx`,
  `sentence-transformers`) sont déclarées dans `pyproject.toml`.

## Installation

Avec [uv](https://docs.astral.sh/uv/) :

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Avec pip ou conda :

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

`src/` est la racine des paquets : `config`, `loader`, `generator`, `utils` et
`dataviz_tools` s'importent en top-level. **Pour tout ce qui n'est pas la suite
de tests, mettez `src/` sur le `PYTHONPATH`** — l'installation éditable ne
suffit pas encore (cf. [Limites connues](#limites-connues)) :

```bash
export PYTHONPATH=src
```

Vérification que la chaîne d'imports est complète :

```bash
PYTHONPATH=src python -c "
from config import BBOX
from loader.osm_loaders import load_pois
from generator.template_question.schema import TEMPLATE_REGISTRY
print(len(TEMPLATE_REGISTRY), 'templates,  bbox', BBOX)
"
# -> 12 templates,  bbox (2.24, 48.8, 2.41, 48.9)
```

---

## Lancer les tests

Depuis la racine du dépôt :

```bash
python -m pytest
```

Sortie attendue :

```
145 passed, 24 xfailed
```

> **`python -m pytest`, pas `pytest`.** Les tests s'importent entre eux
> (`from tests.conftest import …`), `tests/` n'a pas d'`__init__.py`, et
> `pytest.ini` ne met que `src` sur le `pythonpath`. La forme `python -m` ajoute
> le répertoire courant à `sys.path` ; `pytest` seul échoue à la collecte avec
> `ModuleNotFoundError: No module named 'tests'`.

Quelques invocations utiles :

```bash
python -m pytest tests/test_template_contract.py   # un fichier
python -m pytest -k point_near                     # un template
python -m pytest -rxX                              # détail des xfail/xpass
```

### Lire le résultat : le rôle des `xfailed`

**Un `xfailed` n'est pas un test ignoré, c'est un défaut inventorié.** Les
défauts connus des générateurs sont recensés dans `tests/known_defects.py`,
chacun avec son fichier, sa ligne et sa mesure. Les tests qui les démontrent
portent un `xfail(strict=True)`, ce qui donne trois propriétés :

- la suite sort **verte** tant que les défauts connus le restent, donc elle est
  exploitable en intégration continue ;
- chaque défaut reste **visible et documenté** — `python -m pytest -rxX` en
  affiche le motif complet ;
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

Le périmètre géographique et les tags OSM sont dans `src/config.py` (`BBOX`,
`CATEGORIES`, `STREETS_TAGS`, `AREA_TAGS`). `BBOX` couvre Paris ;
`BBOX_TEST` est un petit rectangle autour du Marais, utile pour itérer vite.

```python
# PYTHONPATH=src
from loader.osm_loaders import load_pois, load_streets, load_area
from generator.template_question.geospatial.point_near import make_question_point_near
from utils.dataset_io import save_benchmark

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
- `save_benchmark` / `save_benchmark_suite` (`src/utils/dataset_io.py`) écrivent
  dans `data/benchmarks/geospatial/` avec un fichier de métadonnées (commit git,
  versions des bibliothèques, schéma). `data/` est ignoré par git.
- `make_question_geo`
  (`src/generator/template_question/composite/generator_qcomposite.py`) combine
  questions géospatiales et sémantiques ; c'est le point d'entrée du benchmark
  complet, et il attend `df_osm`, `df_area` et `df_streets`.
- Le nettoyage des types de cuisine peut appeler `sentence-transformers`
  (`paraphrase-multilingual-MiniLM-L12-v2`) : premier appel = téléchargement du
  modèle.

`demo.ipynb` montre l'enchaînement bout en bout. Lancez Jupyter depuis la racine
du dépôt, avec `src/` sur le `PYTHONPATH`.

---

## Organisation

```
src/
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

- **`config` n'est pas installé par `pip install -e .`.** `pyproject.toml`
  déclare `[tool.setuptools.packages.find] where = ["src"]`, qui découvre bien
  les paquets (`loader`, `generator`, `utils`, `dataviz_tools`) mais pas le
  module `src/config.py` — dont dépend `loader/osm_loaders.py`. D'où le
  `PYTHONPATH=src`. Correctif : ajouter `[tool.setuptools] py-modules =
  ["config"]`.
- **`pytest` seul ne collecte pas** (voir plus haut). Correctif : `pythonpath =
  src .` dans `pytest.ini`.
- **`workflows/tests.yml` n'est pas au bon endroit** : GitHub Actions lit
  `.github/workflows/`. Le fichier n'est donc jamais exécuté, et sa dernière
  étape (`pytest`, sans `python -m`) échouerait à la collecte.
- **`extract_woosmap_pois` contient des chemins absolus** vers la machine de
  l'auteur (`/Users/sese/Documents/code/benchmark_NL_POI/…`). La fonction n'est
  pas utilisable ailleurs en l'état ; le chemin OSM (`extract_osm_pois`), lui,
  fonctionne partout.
- **Les défauts des générateurs sont inventoriés, pas corrigés.** Voir les 24
  `xfailed` et `tests/known_defects.py` : `area_direction` publie un schéma
  divergent, plusieurs templates ne respectent pas `nb_q`, `point_between` a un
  registre périmé. C'est délibéré — la suite les constate pour qu'on sache quoi
  réparer et dans quel ordre.
