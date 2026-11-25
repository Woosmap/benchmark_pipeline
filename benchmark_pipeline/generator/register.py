import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
import numpy as np
from pandas import DataFrame
import inspect

import benchmark_pipeline.config as config
import geopandas as gpd
from benchmark_pipeline.config import (
    ANSWERS_COLUMN, DESAMBIGUIOUS_NAME, FORMAT_DIRECTION, IS_PARAPHRASE, LATERAL_AREA_BUFFER,
    LATERAL_NB_QUERY_FEATURE, LATERAL_NB_QUERY_TEMPLATE, LEN_PATH, MAX_NEW_QUERIES, MAX_QUERY_NODES, NB_HARD_QUESTION,
    NB_HARD_UNMATCHED, NB_POIS, NB_QUERY_FEATURE, NB_QUERY_TEMPLATE, NB_QUESTIONS, NB_QUESTIONS_REL, NB_RELATIONS,
    QUESTION_COLUMN, RATIO_FEATURES, RATIO_TEST, RATIO_WITH_FEATURES, ROWS_TARGET, SEED, SPATIAL_ANSWERS_COLUMN,
    TEMPLATE_MULTIHOP, USE_CACHE,
)

CONFIG_DIR = Path(config.__file__).parent

@dataclass
class BenchmarkConfig:
    """What is chosen for the benchmark of one city; defaults come from `benchmark_pipeline.config`."""
    # --- city
    name: str
    bbox: tuple[float, float, float, float]   # WGS84 (west, south, east, north)
    ecrs: int                                 # EPSG code of the metric CRS of the city
    seed: int = SEED

    # --- OSM loading and name deduplication
    use_cache: bool = USE_CACHE
    desambiguous_name: bool = DESAMBIGUIOUS_NAME

    # --- simple questions: spatial questions, train/test split, attributes
    nb_q_by_temp: int = NB_QUERY_TEMPLATE
    ratio_test: float = RATIO_TEST            # also splits the multihop starting questions and the relational graphs
    nb_q_by_feat: int = NB_QUERY_FEATURE
    ratio_features: dict[int, float] = field(default_factory=lambda: dict(RATIO_FEATURES))
    part_with_features: float = RATIO_WITH_FEATURES
    is_paraphrase: bool = IS_PARAPHRASE

    # --- simple questions: train/test rows
    rows_target: list[str] = field(default_factory=lambda: list(ROWS_TARGET))
    nb_hard: int = NB_HARD_QUESTION
    nb_unmatches: int = NB_HARD_UNMATCHED
    colonne_question: str = QUESTION_COLUMN
    colonne_pois: str = ANSWERS_COLUMN
    colonne_pois_spatiaux: str = SPATIAL_ANSWERS_COLUMN

    # --- multihop
    nb_seed_questions: int = NB_QUESTIONS
    multihop_template: dict[int, int] = field(default_factory=lambda: dict(TEMPLATE_MULTIHOP))
    max_questions_per_node: int = MAX_QUERY_NODES
    max_new_queries: int = MAX_NEW_QUERIES
    lateral_area_buffer: float = LATERAL_AREA_BUFFER
    lateral_nb_q_by_temp: int = LATERAL_NB_QUERY_TEMPLATE
    lateral_nb_q_by_feat: int = LATERAL_NB_QUERY_FEATURE

    # --- relational
    relational_nb_questions: int = NB_QUESTIONS_REL
    relational_nb_pois: int = NB_POIS
    relational_nb_relations: int = NB_RELATIONS
    relational_path_length: int = LEN_PATH
    relational_format_direction: str = FORMAT_DIRECTION


def _modify_geo_column(df, ecrs):
    geometry_columns = [column for column in df.columns if "geometry" in column]
    return gpd.GeoDataFrame(
        # astype(object): a geometry column with no value in a split comes as float64 NaN, refused by GeoSeries
        df.assign(**{column: gpd.GeoSeries(df[column].astype(object), crs=f"EPSG:{ecrs}")
                                for column in geometry_columns}),
        geometry=geometry_columns[0],
    )


def _save_config(save_path, conf):
    """Copy the shared config modules and write `conf` in `<name>.json`."""
    config_dir = save_path / "config"
    config_dir.mkdir(exist_ok=True)
    for path in CONFIG_DIR.glob("*.py"):
        if path.name not in ("__init__.py", "cities.py"):
            shutil.copy(path, config_dir / path.name)
    (config_dir / f"{save_path.name}.json").write_text(json.dumps(asdict(conf), indent=2))


def register(
        save_path: Path,
        questions_by_split,
        df_osm,
        df_street,
        df_area,
        df_all,
        conf,
        formating_function,
        overwritte: bool = False,
    ):
    """Write the config, the shared tables and, for each split, its questions in `question/<split>/simple_question`:
    `df_questions.parquet`, the benchmark rows made by `formating_function`, and `df_question_full_table.parquet`,
    every column of the generated questions (the `index` of a row is its position in this table).

    Args:
        questions_by_split (dict[str, DataFrame]): {"train": df_question, "test": df_question}.
        formating_function: called on each split with `is_test`, e.g. `format_composit_samples`.
    """
    save_path.mkdir(parents=True, exist_ok=overwritte)
    (save_path / "tables").mkdir(exist_ok=True)
    _save_config(save_path, conf)

    rng = np.random.default_rng(conf.seed)
    accepted = inspect.signature(formating_function).parameters
    kwargs = {k: v for k, v in asdict(conf).items() if k in accepted}
    for split, df_question in questions_by_split.items():
        split_path = save_path / "question" / split / "simple_question"
        split_path.mkdir(parents=True, exist_ok=True)
        questions = formating_function(df_question, df_osm, rng, is_test=split == "test", **kwargs)
        DataFrame(questions).to_parquet(split_path / "df_questions.parquet")
        _modify_geo_column(df_question, conf.ecrs).to_parquet(split_path / "df_question_full_table.parquet")

    tables = {"df_osm": df_osm, "df_street": df_street, "df_area": df_area, "df_all": df_all}
    for name, table in tables.items():
        _modify_geo_column(table, conf.ecrs).to_parquet(save_path / "tables" / f"{name}.parquet")
