import json
from dataclasses import asdict, dataclass

import numpy as np
from pandas import DataFrame

from benchmark_pipeline.generator.template_question.semantic.batcher import (
    make_benchmark_question,
)


@dataclass
class BenchmarkConfig:
    ratio_test: float
    ratio_features: dict[int, float] | None
    ratio_category: dict[str, float] | None
    nb_hard: int
    nb_unmatches: int
    seed: int

def register_semantic_query(
    save_path,
    df_question,
    df_osm,
    conf,
    ):
    kwargs = asdict(conf)
    rng = np.random.default_rng(kwargs.pop("seed"))
    train_set, test_set = make_benchmark_question(df_question, df_osm, rng, **kwargs)

    DataFrame(train_set).to_parquet(save_path + "train_set.parquet")
    DataFrame(test_set).to_parquet(save_path + "test_set.parquet")
    df_question.to_parquet(save_path + "df_question.parquet")
    df_osm.to_parquet(save_path + "df_osm.parquet")
    with open(save_path + "config.json", "w") as f:
        json.dump(asdict(conf), f, indent=2)
