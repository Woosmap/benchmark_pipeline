import json
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from pandas import DataFrame

from benchmark_pipeline.config import ECRS
import geopandas as gpd

@dataclass
class BenchmarkConfig:
    ratio_test: float
    ratio_features: dict[int, float] | None
    ratio_category: dict[str, float] | None
    rows_target: list[str]
    nb_hard: int
    nb_unmatches: int
    seed: int

def _modify_geo_column(df):
    geometry_columns = [column for column in df.columns if "geometry" in column]
    return gpd.GeoDataFrame(
        df.assign(**{column: gpd.GeoSeries(df[column], crs=f"EPSG:{ECRS}")
                                for column in geometry_columns}),
        geometry=geometry_columns[0],
    )


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

    kwargs = asdict(conf)
    rng = np.random.default_rng(kwargs.pop("seed"))
    train_set, test_set = formating_function(df_question, df_osm, rng, **kwargs)
    DataFrame(train_set).to_parquet(save_path / "question" / "train_set.parquet")
    DataFrame(test_set).to_parquet(save_path / "question" / "test_set.parquet")

    tables = {"df_question": df_question, "df_osm": df_osm, "df_street": df_street, "df_area": df_area, "df_all": df_all}
    for name, table in tables.items():
        _modify_geo_column(table).to_parquet(save_path / "tables" / f"{name}.parquet")

    with open(save_path / "config.json", "w") as f:
        json.dump(asdict(conf), f, indent=2)
