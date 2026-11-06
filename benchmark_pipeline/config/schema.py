from benchmark_pipeline.generator.features import *

#Features used
FEATURES = [Category(), CookingType(), OutDoorSeating(), InDoorSeating()]
MANDATORY_FEATURES = "category"

MULTIHOP_COLUMNS = ["query_texte", "features", "dic_query", "values", "answers", "template"]
