import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from pandas import DataFrame
import inspect

import benchmark_pipeline.config as config
import geopandas as gpd
from benchmark_pipeline.config import SEED

CONFIG_DIR = Path(config.__file__).parent

@dataclass
class BenchmarkConfig:
    ratio_test: float
    ratio_features: dict[int, float] | None
    ratio_category: dict[str, float] | None
    rows_target: list[str]
    nb_hard: int
    nb_unmatches: int
    seed: int

def _modify_geo_column(df, ecrs):
    geometry_columns = [column for column in df.columns if "geometry" in column]
    return gpd.GeoDataFrame(
        df.assign(**{column: gpd.GeoSeries(df[column], crs=f"EPSG:{ecrs}")
                                for column in geometry_columns}),
        geometry=geometry_columns[0],
    )


def _save_config(save_path, bbox, ecrs, conf):
    """Copy the shared config modules and write what is specific to this benchmark in `<name>.json`."""
    config_dir = save_path / "config"
    config_dir.mkdir(exist_ok=True)
    for path in CONFIG_DIR.glob("*.py"):
        if path.name not in ("__init__.py", "cities.py"):
            shutil.copy(path, config_dir / path.name)
    specific = {"name": save_path.name, "bbox": list(bbox), "ecrs": ecrs, **asdict(conf)}
    (config_dir / f"{save_path.name}.json").write_text(json.dumps(specific, indent=2))


def register(
        save_path: Path,
        df_question,
        df_osm,
        df_street,
        df_area,
        df_all,
        conf,
        formating_function,
        bbox,
        ecrs,
        overwritte: bool = False,
    ):
    save_path.mkdir(parents=True, exist_ok=overwritte)
    (save_path / "question").mkdir(exist_ok=True)
    (save_path / "tables").mkdir(exist_ok=True)
    _save_config(save_path, bbox, ecrs, conf)

    kwargs = asdict(conf)
    rng = np.random.default_rng(kwargs.pop("seed"))
    accepted = inspect.signature(formating_function).parameters
    train_set, test_set = formating_function(df_question, df_osm, rng, **{k: v for k, v in kwargs.items() if k in accepted})
    DataFrame(train_set).to_parquet(save_path / "question" / "train_set.parquet")
    DataFrame(test_set).to_parquet(save_path / "question" / "test_set.parquet")

    tables = {"df_question": df_question, "df_osm": df_osm, "df_street": df_street, "df_area": df_area, "df_all": df_all}
    for name, table in tables.items():
        _modify_geo_column(table, ecrs).to_parquet(save_path / "tables" / f"{name}.parquet")
