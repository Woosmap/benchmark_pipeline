#Multihop
LIST_TEMPLATE_FORWARD = ['make_question_point_between', 'make_question_point_near_cardinal',
                         'make_question_point_near_metric', 'make_question_point_near',
                         'make_question_point_towards']

TEMPLATE_MULTIHOP = {8:0, 7:2, 6:1, 5: 3, 4: 1, 3: 2, 2: 1, 1: 2}
NB_QUESTIONS = 50

MAX_QUERY_NODES = 3
MAX_NEW_QUERIES = 50
LATERAL_AREA_BUFFER = 200                          # add_lateral_hop: meters around the answers where lateral anchors are searched
LATERAL_NB_QUERY_TEMPLATE = 10                     # add_lateral_hop: nb_q of each template for the lateral questions
LATERAL_NB_QUERY_FEATURE = 10                      # add_lateral_hop: attribute constraints drawn per lateral question
