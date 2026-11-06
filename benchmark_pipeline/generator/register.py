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
    ANSWERS_COLUMN, IS_PARAPHRASE, NB_HARD_QUESTION, NB_HARD_UNMATCHED, NB_QUERY_FEATURE, NB_QUERY_TEMPLATE,
    NB_SEED_QUESTIONS, QUESTION_COLUMN, RATIO_FEATURES, RATIO_TEST, RATIO_WITH_FEATURES, ROWS_TARGET, SEED,
    SPATIAL_ANSWERS_COLUMN, TEMPLATE_MULTIHOP,
)

CONFIG_DIR = Path(config.__file__).parent

@dataclass
class BenchmarkConfig:
    """What is chosen for the benchmark of one city; defaults come from `benchmark_pipeline.config`."""
    name: str
    bbox: tuple[float, float, float, float]   # WGS84 (west, south, east, north)
    ecrs: int                                 # EPSG code of the metric CRS of the city
    seed: int = SEED
    nb_q_by_temp: int = NB_QUERY_TEMPLATE
    nb_q_by_feat: int = NB_QUERY_FEATURE
    ratio_features: dict[int, float] = field(default_factory=lambda: dict(RATIO_FEATURES))
    part_with_features: float = RATIO_WITH_FEATURES
    is_paraphrase: bool = IS_PARAPHRASE
    ratio_test: float = RATIO_TEST
    rows_target: list[str] = field(default_factory=lambda: list(ROWS_TARGET))
    nb_hard: int = NB_HARD_QUESTION
    nb_unmatches: int = NB_HARD_UNMATCHED
    colonne_question: str = QUESTION_COLUMN
    colonne_pois: str = ANSWERS_COLUMN
    colonne_pois_spatiaux: str = SPATIAL_ANSWERS_COLUMN
    nb_seed_questions: int = NB_SEED_QUESTIONS
    multihop_template: dict[int, int] = field(default_factory=lambda: dict(TEMPLATE_MULTIHOP))

def _modify_geo_column(df, ecrs):
    geometry_columns = [column for column in df.columns if "geometry" in column]
    return gpd.GeoDataFrame(
        df.assign(**{column: gpd.GeoSeries(df[column], crs=f"EPSG:{ecrs}")
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
        df_question,
        df_osm,
        df_street,
        df_area,
        df_all,
        conf,
        formating_function,
        overwritte: bool = False,
    ):
    save_path.mkdir(parents=True, exist_ok=overwritte)
    (save_path / "question").mkdir(exist_ok=True)
    (save_path / "tables").mkdir(exist_ok=True)
    _save_config(save_path, conf)

    rng = np.random.default_rng(conf.seed)
    accepted = inspect.signature(formating_function).parameters
    train_set, test_set = formating_function(df_question, df_osm, rng, **{k: v for k, v in asdict(conf).items() if k in accepted})
    DataFrame(train_set).to_parquet(save_path / "question" / "train_set.parquet")
    DataFrame(test_set).to_parquet(save_path / "question" / "test_set.parquet")

    tables = {"df_question": df_question, "df_osm": df_osm, "df_street": df_street, "df_area": df_area, "df_all": df_all}
    for name, table in tables.items():
        _modify_geo_column(table, conf.ecrs).to_parquet(save_path / "tables" / f"{name}.parquet")
