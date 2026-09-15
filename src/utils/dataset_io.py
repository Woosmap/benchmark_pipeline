"""Enregistrement et relecture des benchmarks de questions.

Un benchmark tel que produit par `src.template_question.type_A.make_question_*`
ne passe pas tel quel dans un parquet : il mélange des colonnes-listes
(`results_poi_id`…) et des géométries shapely dans des colonnes objet
(`area_geometry`, `street_geometry`…). Les géométries sont donc encodées en WKB à
l'écriture et redécodées à la lecture.

Ne pas utiliser `safe_to_parquet` (`src.data_collection.osm_extractor`) pour
cela : ce helper fait `astype(str)` sur toute colonne que pyarrow refuse, donc
il transformerait `results_poi_id` en la chaîne `"[123, 456]"`, et comme il ne
protège que la colonne nommée exactement `geometry`, il stringifierait aussi
`area_geometry`. Il est fait pour les extractions OSM brutes, pas pour ces
frames.

Chaque écriture produit deux fichiers :

* `<nom>.parquet` — les questions ;
* `<nom>.meta.json` — la provenance : graine, effectifs, répartition par
  template, CRS, commit git, versions des bibliothèques. Sans ce fichier, un
  benchmark sur disque ne dit pas avec quelle graine il a été tiré, et n'est donc
  pas reproductible.
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import from_wkb, to_wkb

from src.template_question.schema import (
    GEOMETRY_COLUMNS,
    RESULT_LIST_COLUMNS,
    validate_benchmark,
)

logger = logging.getLogger(__name__)

#: Emplacement par défaut des benchmarks produits. Le projet n'a pas de dossier
#: `data/` : il est créé au besoin.
DEFAULT_OUTDIR = Path("data/benchmarks/type_A")

#: Version du format sur disque. À incrémenter si l'encodage change, pour que la
#: lecture puisse refuser un fichier qu'elle ne sait pas interpréter.
FORMAT_VERSION = 1

_CRS_LAMBERT93 = 2154


# --------------------------------------------------------------------------- #
# métadonnées
# --------------------------------------------------------------------------- #

def _git_commit():
    """Renvoie le commit courant, ou None hors dépôt git."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parents[2],
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _library_versions():
    """Versions des bibliothèques dont dépend la vérité terrain."""
    versions = {"pandas": pd.__version__, "numpy": np.__version__}
    for name in ("geopandas", "shapely", "duckdb", "scipy"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = None
    return versions


def _build_meta(bench, geometry_columns, extra=None):
    """Assemble le contenu du sidecar."""
    meta = {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_questions": int(len(bench)),
        "columns": list(bench.columns),
        "geometry_columns": list(geometry_columns),
        "list_columns": [c for c in RESULT_LIST_COLUMNS if c in bench.columns],
        "crs": _CRS_LAMBERT93,
        "git_commit": _git_commit(),
        "library_versions": _library_versions(),
    }

    if "function" in bench.columns:
        meta["questions_by_function"] = {
            str(k): int(v) for k, v in bench["function"].value_counts().items()
        }
    if "category_query" in bench.columns:
        meta["questions_by_category"] = {
            str(k): int(v) for k, v in bench["category_query"].value_counts().items()
        }

    # effectifs de réponses : un benchmark dont la médiane est à 0 est inutilisable
    if "results_poi_id" in bench.columns:
        sizes = bench["results_poi_id"].apply(
            lambda v: 0 if v is None else len(v)
        )
        meta["answer_sizes"] = {
            "min": int(sizes.min()), "max": int(sizes.max()),
            "mean": round(float(sizes.mean()), 2),
            "median": float(sizes.median()),
            "empty": int((sizes == 0).sum()),
        }

    if extra:
        meta.update(extra)
    return meta


# --------------------------------------------------------------------------- #
# écriture
# --------------------------------------------------------------------------- #

def _encode_geometries(bench):
    """Remplace les géométries shapely par leur WKB. Renvoie (frame, colonnes)."""
    out = bench.copy()
    encoded = []
    for col in bench.columns:
        if col not in GEOMETRY_COLUMNS and not col.endswith("_geometry"):
            continue
        out[col] = [None if g is None or (hasattr(g, "is_empty") and g.is_empty)
                    else to_wkb(g) for g in bench[col]]
        encoded.append(col)
    return out, encoded


def _normalise_lists(bench):
    """Force les colonnes-listes en listes Python.

    Les générateurs produisent parfois des `np.ndarray` (via `list(series)` sur
    une colonne numpy) ; pyarrow les accepte, mais autant figer un type unique
    pour que l'aller-retour soit exactement l'identité.
    """
    out = bench.copy()
    for col in RESULT_LIST_COLUMNS:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda v: [] if v is None else list(v)
            )
    return out


def save_benchmark(bench, path, *, df_osm=None, meta=None, validate="warn",
                   overwrite=False):
    """Écrit un benchmark en parquet, avec son sidecar de métadonnées.

    Args:
        bench (DataFrame): Le benchmark à enregistrer.
        path (str | Path): Chemin du parquet. L'extension est ajoutée si absente.
            Les dossiers parents sont créés au besoin.
        df_osm (GeoDataFrame | None): Corpus de POIs. Fourni, il permet à la
            vérification de contrôler aussi l'existence des `poi_id`,
            l'appariement id↔nom et la catégorie des résultats.
        meta (dict | None): Métadonnées à joindre au sidecar — au minimum la
            graine utilisée, qu'aucun générateur n'enregistre de lui-même.
        validate ({"warn", "raise", None}): Que faire d'un benchmark incohérent.
            `"warn"` journalise et écrit quand même, `"raise"` refuse d'écrire,
            `None` ne vérifie pas.
        overwrite (bool): Autorise l'écrasement d'un fichier existant.

    Returns:
        Path: Le chemin du parquet écrit.

    Raises:
        FileExistsError: Le fichier existe et `overwrite` est faux.
        ValueError: `validate="raise"` et le benchmark est incohérent, ou
            `validate` a une valeur inattendue.
    """
    if validate not in ("warn", "raise", None):
        raise ValueError(
            f"validate attend 'warn', 'raise' ou None, reçu {validate!r}"
        )

    path = Path(path)
    if path.suffix != ".parquet":
        path = path.with_suffix(".parquet")
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} existe déjà ; passe overwrite=True pour l'écraser"
        )

    if validate is not None:
        problems = validate_benchmark(bench, df_osm)
        if problems:
            summary = "\n  - ".join(problems[:20])
            more = f"\n  … et {len(problems) - 20} autres" if len(problems) > 20 else ""
            message = (
                f"{len(problems)} incohérence(s) dans le benchmark "
                f"{path.stem} :\n  - {summary}{more}"
            )
            if validate == "raise":
                raise ValueError(message)
            logger.warning("%s", message)

    encoded, geom_cols = _encode_geometries(_normalise_lists(bench))

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.to_parquet(path, index=False)

    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(_build_meta(bench, geom_cols, extra=meta),
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("benchmark écrit : %s (%d questions)", path, len(bench))
    return path


def save_benchmark_suite(benchmarks, outdir=DEFAULT_OUTDIR, **kwargs):
    """Écrit plusieurs benchmarks dans un même dossier.

    Args:
        benchmarks (dict[str, DataFrame]): Nom de template → benchmark. Le nom
            sert de nom de fichier.
        outdir (str | Path): Dossier de destination, créé au besoin.
        **kwargs: Transmis à `save_benchmark` (`df_osm`, `validate`,
            `overwrite`, `meta`).

    Returns:
        dict[str, Path]: Nom de template → chemin écrit.
    """
    outdir = Path(outdir)
    written = {}
    for name, bench in benchmarks.items():
        written[name] = save_benchmark(bench, outdir / f"{name}.parquet", **kwargs)
    return written


# --------------------------------------------------------------------------- #
# lecture
# --------------------------------------------------------------------------- #

def load_benchmark(path, with_meta=False):
    """Relit un benchmark écrit par `save_benchmark`.

    Restaure les géométries shapely depuis leur WKB et rend les colonnes-listes
    en `list` Python — parquet les restitue en `np.ndarray`, ce qui suffit à
    `plot_question` (qui fait `list(...)`) mais casse une comparaison d'égalité.

    Args:
        path (str | Path): Chemin du parquet.
        with_meta (bool): Renvoie aussi le contenu du sidecar.

    Returns:
        DataFrame, ou (DataFrame, dict) si `with_meta`. Le sidecar vaut `{}` s'il
        est absent.
    """
    path = Path(path)
    if path.suffix != ".parquet":
        path = path.with_suffix(".parquet")

    bench = pd.read_parquet(path)

    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    version = meta.get("format_version", FORMAT_VERSION)
    if version > FORMAT_VERSION:
        raise ValueError(
            f"{path} est au format v{version}, cette version n'en lit que "
            f"v{FORMAT_VERSION} — mets à jour src/utils/dataset_io.py"
        )

    # Se fier au sidecar s'il liste les colonnes encodées, sinon retomber sur la
    # convention de nommage : un benchmark copié sans son .meta.json reste lisible.
    geom_cols = meta.get("geometry_columns") or [
        c for c in bench.columns
        if c in GEOMETRY_COLUMNS or c.endswith("_geometry")
    ]
    for col in geom_cols:
        if col in bench.columns:
            bench[col] = [None if b is None else from_wkb(bytes(b))
                          for b in bench[col]]

    for col in RESULT_LIST_COLUMNS:
        if col in bench.columns:
            bench[col] = bench[col].apply(
                lambda v: [] if v is None else list(v)
            )

    return (bench, meta) if with_meta else bench


def load_benchmark_suite(outdir=DEFAULT_OUTDIR, with_meta=False):
    """Relit tous les benchmarks d'un dossier.

    Args:
        outdir (str | Path): Dossier écrit par `save_benchmark_suite`.
        with_meta (bool): Transmis à `load_benchmark`.

    Returns:
        dict: Nom de template → benchmark (ou `(benchmark, meta)`), trié par nom.

    Raises:
        FileNotFoundError: Le dossier n'existe pas.
    """
    outdir = Path(outdir)
    if not outdir.is_dir():
        raise FileNotFoundError(f"dossier de benchmarks introuvable : {outdir}")
    return {
        p.stem: load_benchmark(p, with_meta=with_meta)
        for p in sorted(outdir.glob("*.parquet"))
    }
