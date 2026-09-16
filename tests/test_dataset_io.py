"""Enregistrement et relecture d'un benchmark.

Ce qui doit survivre à l'aller-retour : les colonnes-listes (la vérité terrain
n'est rien d'autre que quatre listes parallèles) et les géométries shapely (que
parquet ne sait pas stocker telles quelles). Le reste du fichier vérifie le
sidecar de provenance et le comportement de la vérification à l'écriture.
"""

import json

import pandas as pd
import pytest

from src.template_question.schema import TEMPLATES_BY_NAME
from src.utils.dataset_io import (
    FORMAT_VERSION,
    load_benchmark,
    load_benchmark_suite,
    save_benchmark,
    save_benchmark_suite,
)

gpd = pytest.importorskip("geopandas")
from shapely.geometry import LineString, Point, box  # noqa: E402


@pytest.fixture(scope="module")
def bench(df_osm):
    """Un vrai benchmark, produit par le seul template aujourd'hui sain."""
    from src.template_question.geospatial.point_near import make_question_point_near

    return make_question_point_near(df_osm, nb_q=10, seed=42)


@pytest.fixture
def bench_with_geometries():
    """Benchmark artificiel couvrant tous les cas tordus de l'encodage.

    Les templates sains ne produisent pas de colonne géométrique ; ceux qui en
    produisent sont cassés. Cette frame reproduit donc à la main ce que
    `area_*` et `street_*` publieront une fois réparés — plusieurs colonnes
    `*_geometry` sur la même ligne, des types différents, et des trous.
    """
    return pd.DataFrame({
        "query": ["cafe inside Parc", "bar across rue X from poi7"],
        "category_query": ["cafe", "bar"],
        "function": ["inside_area", "opposite_side"],
        "area_geometry": [box(0, 0, 10, 10), None],
        "street_geometry": [None, LineString([(0, 0), (5, 5)])],
        "poi_y_geometry": [None, Point(3, 4)],
        "results_poi_id": [[3, 1, 2], [9]],
        "results_poi_name": [["a", "b", "c"], ["z"]],
        "results_poi_dist": [[0.0, 1.5, 2.5], [4.0]],
        "results_poi_rank": [[1, 2, 3], [1]],
    })


# --------------------------------------------------------------------------- #
# aller-retour
# --------------------------------------------------------------------------- #

def test_roundtrip_preserves_list_columns(bench, tmp_path, df_osm):
    """Les quatre listes parallèles reviennent identiques.

    C'est le point sur lequel `safe_to_parquet` échouerait : il stringifierait
    `results_poi_id` en `"[123, 456]"`, et la visualisation comme l'évaluation
    ne sauraient plus quoi en faire.
    """
    path = save_benchmark(bench, tmp_path / "point_near.parquet", df_osm=df_osm)
    back = load_benchmark(path)

    for column in ("results_poi_id", "results_poi_name",
                   "results_poi_dist", "results_poi_rank"):
        assert back[column].tolist() == bench[column].tolist(), (
            f"{column} n'a pas survécu à l'aller-retour"
        )


def test_roundtrip_preserves_plain_columns(bench, tmp_path, df_osm):
    """Le contexte de la question revient inchangé, types compris."""
    path = save_benchmark(bench, tmp_path / "b.parquet", df_osm=df_osm)
    back = load_benchmark(path)

    assert list(back.columns) == list(bench.columns)
    pd.testing.assert_frame_equal(
        back.drop(columns=list(back.filter(like="results_poi"))),
        bench.drop(columns=list(bench.filter(like="results_poi"))),
        check_dtype=True,
    )


def test_roundtrip_restores_shapely_geometries(bench_with_geometries, tmp_path):
    """Les géométries repassent de WKB à shapely, trous compris.

    Une ligne `area_*` n'a pas de `street_geometry` et réciproquement : les
    `None` doivent rester des `None`, pas devenir des géométries vides.
    """
    path = save_benchmark(bench_with_geometries, tmp_path / "g.parquet",
                          validate=None)
    back = load_benchmark(path)

    assert back["area_geometry"][0].equals(bench_with_geometries["area_geometry"][0])
    assert back["street_geometry"][1].equals(bench_with_geometries["street_geometry"][1])
    assert back["poi_y_geometry"][1].equals(bench_with_geometries["poi_y_geometry"][1])

    assert back["area_geometry"][1] is None
    assert back["street_geometry"][0] is None
    assert back["poi_y_geometry"][0] is None


def test_loaded_benchmark_is_still_plottable(bench, tmp_path, df_osm):
    """Un benchmark relu satisfait encore les préconditions de `plot_question`.

    Sans cela, un jeu enregistré serait exploitable par l'évaluation mais plus
    relisible à l'œil, et les erreurs de vérité terrain deviendraient
    invisibles.
    """
    pytest.importorskip("matplotlib")
    from src.template_question.schema import validate_benchmark

    path = save_benchmark(bench, tmp_path / "b.parquet", df_osm=df_osm)
    back = load_benchmark(path)

    assert not validate_benchmark(back, df_osm, allow_empty=True)


# --------------------------------------------------------------------------- #
# sidecar de provenance
# --------------------------------------------------------------------------- #

def test_sidecar_records_provenance(bench, tmp_path, df_osm):
    """Le `.meta.json` conserve ce que le parquet ne dit pas.

    La graine en premier lieu : aucun générateur ne l'enregistre, or sans elle
    un benchmark n'est pas reproductible.
    """
    path = save_benchmark(bench, tmp_path / "point_near.parquet",
                          df_osm=df_osm, meta={"seed": 42, "nb_q": 10})
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))

    assert meta["seed"] == 42
    assert meta["nb_q"] == 10
    assert meta["n_questions"] == len(bench)
    assert meta["format_version"] == FORMAT_VERSION
    assert meta["crs"] == 2154
    assert meta["questions_by_function"] == {"near_sql": len(bench)}
    assert set(meta["columns"]) == set(bench.columns)
    assert meta["library_versions"]["pandas"] == pd.__version__
    assert "created_at" in meta


def test_sidecar_reports_answer_sizes(bench, tmp_path, df_osm):
    """Le sidecar résume la taille des réponses.

    Un jeu dont beaucoup de questions sont vides est inévaluable ; le compte
    doit être lisible sans rouvrir le parquet.
    """
    path = save_benchmark(bench, tmp_path / "b.parquet", df_osm=df_osm)
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))

    sizes = bench["results_poi_id"].apply(len)
    assert meta["answer_sizes"]["min"] == int(sizes.min())
    assert meta["answer_sizes"]["max"] == int(sizes.max())
    assert meta["answer_sizes"]["empty"] == int((sizes == 0).sum())


def test_load_works_without_sidecar(bench_with_geometries, tmp_path):
    """Un parquet copié sans son `.meta.json` reste lisible.

    La relecture retombe alors sur la convention de nommage `*_geometry` pour
    retrouver les colonnes à décoder.
    """
    path = save_benchmark(bench_with_geometries, tmp_path / "g.parquet",
                          validate=None)
    path.with_suffix(".meta.json").unlink()

    back = load_benchmark(path)
    assert back["area_geometry"][0].equals(bench_with_geometries["area_geometry"][0])


def test_load_refuses_a_future_format(bench, tmp_path, df_osm):
    """Une version de format inconnue doit être refusée, pas mal interprétée."""
    path = save_benchmark(bench, tmp_path / "b.parquet", df_osm=df_osm)
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["format_version"] = FORMAT_VERSION + 1
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValueError, match="format v"):
        load_benchmark(path)


# --------------------------------------------------------------------------- #
# vérification à l'écriture
# --------------------------------------------------------------------------- #

def test_save_warns_on_incoherent_benchmark(tmp_path, caplog, df_osm):
    """Par défaut, une incohérence est signalée mais n'empêche pas d'écrire.

    Les templates ont encore des défauts connus : bloquer l'écriture rendrait
    l'outil inutilisable. L'avertissement, lui, doit être explicite.
    """
    broken = pd.DataFrame({
        "query": ["cafe near poi0_cafe"],
        "category_query": ["cafe"],
        "function": ["near_sql"],
        "results_poi_id": [[0, 0, 999999]],          # doublon + id inconnu
        "results_poi_name": [["a", "b", "c"]],
        "results_poi_dist": [[5.0, 1.0, 2.0]],       # décroissante
        "results_poi_rank": [[1, 2, 3]],
    })

    with caplog.at_level("WARNING"):
        path = save_benchmark(broken, tmp_path / "broken.parquet", df_osm=df_osm)

    assert path.exists(), "l'écriture doit avoir lieu malgré l'avertissement"
    assert "incohérence" in caplog.text
    assert "dupliqués" in caplog.text


def test_save_can_refuse_an_incoherent_benchmark(tmp_path, df_osm):
    """`validate="raise"` bloque l'écriture d'un jeu incohérent."""
    broken = pd.DataFrame({
        "query": ["cafe near poi0_cafe"],
        "category_query": ["cafe"],
        "function": ["near_sql"],
        "results_poi_id": [[0, 0]],
        "results_poi_name": [["a", "b"]],
        "results_poi_dist": [[1.0, 2.0]],
        "results_poi_rank": [[1, 2]],
    })

    with pytest.raises(ValueError, match="incohérence"):
        save_benchmark(broken, tmp_path / "broken.parquet",
                       df_osm=df_osm, validate="raise")
    assert not (tmp_path / "broken.parquet").exists()


def test_save_rejects_an_unknown_validate_mode(bench, tmp_path):
    """Un mode de vérification mal orthographié ne doit pas passer pour `None`."""
    with pytest.raises(ValueError, match="validate attend"):
        save_benchmark(bench, tmp_path / "b.parquet", validate="oui")


# --------------------------------------------------------------------------- #
# écrasement et lots
# --------------------------------------------------------------------------- #

def test_save_refuses_to_overwrite_by_default(bench, tmp_path, df_osm):
    """Un benchmark coûte cher à produire : on ne l'écrase pas par accident."""
    path = save_benchmark(bench, tmp_path / "b.parquet", df_osm=df_osm)

    with pytest.raises(FileExistsError):
        save_benchmark(bench, path, df_osm=df_osm)

    save_benchmark(bench, path, df_osm=df_osm, overwrite=True)


def test_save_creates_missing_directories(bench, tmp_path, df_osm):
    """Le projet n'a pas de dossier `data/` : il doit être créé au besoin."""
    path = save_benchmark(bench, tmp_path / "data" / "benchmarks" / "geospatial" / "b",
                          df_osm=df_osm)
    assert path.exists()
    assert path.suffix == ".parquet", "l'extension doit être ajoutée si absente"


def test_suite_roundtrip(bench, tmp_path, df_osm):
    """Un lot de benchmarks s'écrit et se relit par nom de template."""
    benchmarks = {"point_near": bench, "point_near_bis": bench}
    written = save_benchmark_suite(benchmarks, tmp_path / "suite", df_osm=df_osm)

    assert set(written) == set(benchmarks)
    assert all(p.exists() for p in written.values())

    back = load_benchmark_suite(tmp_path / "suite")
    assert set(back) == set(benchmarks)
    assert back["point_near"]["results_poi_id"].tolist() == bench["results_poi_id"].tolist()


def test_load_suite_reports_a_missing_directory(tmp_path):
    """Un dossier absent doit se dire, pas renvoyer un lot vide."""
    with pytest.raises(FileNotFoundError, match="introuvable"):
        load_benchmark_suite(tmp_path / "nexiste_pas")


# --------------------------------------------------------------------------- #
# le registre des templates reste aligné sur le code
# --------------------------------------------------------------------------- #

#: Modules de `geospatial/` qui ne sont pas des templates de question et n'ont donc
#: rien à faire dans le registre. `make_question_geo` est un combinateur encore
#: en chantier — il croise features et questions géo — et ne suit pas le contrat
#: `make_question_*(df, …, nb_q, seed) -> DataFrame`. Le recenser ici plutôt que
#: de relâcher le test garde la garantie : tout *autre* module ajouté à `geospatial/`
#: fera échouer `test_every_template_module_is_in_the_registry`.
NON_TEMPLATE_MODULES = {"make_question_geo"}


def test_registry_matches_the_modules_on_disk():
    """Chaque entrée du registre désigne un module et un générateur réels.

    Le registre pilote la paramétrisation des tests *et* l'enregistrement en
    lot. Une entrée périmée ferait silencieusement disparaître un template de
    la couverture.
    """
    import importlib
    import pathlib

    from src.template_question.schema import TEMPLATE_REGISTRY

    for template in TEMPLATE_REGISTRY:
        module_path = pathlib.Path(
            "src/template_question/geospatial") / f"{template.name}.py"
        assert module_path.exists(), f"{module_path} est introuvable"

        module = importlib.import_module(template.module)
        assert callable(getattr(module, template.generator, None)), (
            f"{template.generator} absent de {template.module}"
        )


def test_every_template_module_is_in_the_registry():
    """Aucun template sur disque n'échappe au registre."""
    import pathlib

    on_disk = {
        p.stem for p in pathlib.Path("src/template_question/geospatial").glob("*.py")
        if not p.stem.startswith("_")
    } - NON_TEMPLATE_MODULES
    assert on_disk == set(TEMPLATES_BY_NAME), (
        f"registre et disque divergent : "
        f"seulement sur disque {sorted(on_disk - set(TEMPLATES_BY_NAME))}, "
        f"seulement au registre {sorted(set(TEMPLATES_BY_NAME) - on_disk)}"
    )


def test_the_two_registries_describe_the_same_templates():
    """Le registre par décorateur et celui du schéma ne doivent pas diverger.

    Le projet en tient deux : `registry.REGISTRY`, peuplé à l'import par le
    décorateur `@template` de chaque module, et `schema.TEMPLATE_REGISTRY`,
    écrit à la main parce qu'il porte en plus les colonnes de contexte et le
    libellé attendu dans l'énoncé. Ils sont indexés différemment — le premier
    par nom de fonction, le second par nom court — ce qui rend la divergence
    invisible à l'œil.

    Un template décoré mais absent du schéma échappe à toute la suite ; l'
    inverse ferait échouer la paramétrisation. Ce test est le seul endroit qui
    les confronte.
    """
    import src.template_question.geospatial  # noqa: F401 — peuple REGISTRY
    from src.template_question.registry import REGISTRY

    decorated = set(REGISTRY)
    declared = {t.generator for t in TEMPLATES_BY_NAME.values()}

    assert decorated == declared, (
        f"les deux registres divergent : décorés mais absents du schéma "
        f"{sorted(decorated - declared)}, déclarés au schéma mais non décorés "
        f"{sorted(declared - decorated)}"
    )


def test_registry_generators_share_the_nb_q_and_seed_contract():
    """Tout générateur accepte `nb_q` et `seed` en mot-clé.

    C'est le contrat sur lequel reposent `run_template` et l'enregistrement en
    lot : appeler un générateur sans savoir lequel c'est. Les signatures
    diffèrent par ailleurs — `list_direction`, `min_size`, `corridor_m`,
    `ratio_cat_q` — mais ces deux-là doivent rester communs, sinon le lot ne
    peut plus être produit d'un seul appel paramétré.
    """
    import importlib
    import inspect

    from src.template_question.schema import TEMPLATE_REGISTRY

    for template in TEMPLATE_REGISTRY:
        module = importlib.import_module(template.module)
        params = inspect.signature(getattr(module, template.generator)).parameters
        missing = [p for p in ("nb_q", "seed") if p not in params]
        assert not missing, (
            f"{template.generator} n'accepte pas {missing} : le lot ne peut "
            f"pas être produit par un appel uniforme"
        )
