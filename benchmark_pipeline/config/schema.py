from benchmark_pipeline.generator.features import *

#Features used
FEATURES = [Category(), CookingType(), OutDoorSeating(), InDoorSeating()]
MANDATORY_FEATURES = "category"

MULTIHOP_COLUMNS = ["query_texte", "features", "dic_query", "values", "answers", "template"]

# Columns of df_question used by make_rows to build the train/test rows
QUESTION_COLUMN = "query_texte"                       # question text published ("query_geo_semantic": tagged column=value version)
ANSWERS_COLUMN = "results_features_pois_id"           # answers: POIs satisfying the spatial relation and the attributes
SPATIAL_ANSWERS_COLUMN = "results_poi_id"             # POIs satisfying the spatial relation only; hard negatives are drawn there
