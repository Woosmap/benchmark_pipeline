from pandas import DataFrame
from numpy import random as rng
from dataclasses import dataclass, asdict

from generator.template_question.semantic.batcher import make_benchmark_question

@dataclass
class BenchmarkConfig:
    ratio_test: float
    ratio_features: dict[int, float]
    batch_size: int
    nb_hard: int
    nb_unmatches: int
    n_times: int
    nb_answers: int

def register_semantic_query(
        save_path,
        df_question,
        df_osm,
        rng,
        conf,
    ):
    train_set, test_set = make_benchmark_question(df_question, df_osm, rng, **asdict(conf))

    DataFrame(train_set).to_parquet(save_path + "train_set.parquet")
    DataFrame(test_set).to_parquet(save_path + "test_set.parquet")
    DataFrame(asdict(conf)).to_parquet(save_path + "config.parquet")
