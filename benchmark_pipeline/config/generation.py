SEED = 42

NB_QUERY_TEMPLATE = 1000
NB_QUERY_FEATURE = 10
RATIO_FEATURES = {1: 1, 2: 1, 3: 1}
RATIO_WITH_FEATURES = 1.0
IS_PARAPHRASE = False

#Questions singles parameters
NB_HARD_QUESTION = 20
NB_HARD_UNMATCHED = 1

RATIO_TEST = 0.1                                   # share of the questions kept for the test set
ROWS_TARGET = ["anchors_mask", "anchors", "ellypse"]                                   # df_question columns copied into each train/test row, e.g. ["ellypse"]
