LIST_TEMPLATE = ['make_question_area_border',
                 'make_question_area_inside', 'make_question_area_direction',
                 'make_question_point_between', 'make_question_point_near_cardinal',
                 'make_question_point_near_metric', 'make_question_point_near',
                 'make_question_point_towards', 'make_question_street_along',
                 'make_question_street_cross']

DISTANCE_MAX = 1000 #en mètres
DISTANCE_STREET_MAX = 50


#SQL DEFINITION FUNCTION
#Template parameters (meters unless stated)
DIRECTIONS = ["north", "east", "west", "south"]   # area_direction, point_near_cardinal
METRIC_DISTANCES = [100, 300, 500, 1000]           # point_near_metric

AREA_DIRECTION_MIN_SIZE = 20000                    # m², area_direction
AREA_BORDER_MIN_SIZE = 20000                       # m², area_border
AREA_BORDER_BAND = 200                             # area_border: distance to the border of the area
AREA_INSIDE_MIN_SIZE = 1000                        # m², area_inside
AREA_OUTSIDE_MIN_SIZE = 1000                       # m², area_outside
MAX_TRIES = 200                       # area_outside: draws per category street_opposite_side

BETWEEN_CORRIDOR = 200.0                           # point_between: half-width of the corridor between A and B
BETWEEN_NB_CANDIDATE_ANCHORS = 300                 # point_between: anchors A sampled per category
TOWARDS_HALF_WIDTH = 70.0                          # degrees, point_towards: half-angle of the cone
STREET_ALONG_MIN_LENGTH = 500                      # street_along
STREET_CROSS_TOLERANCE = 1.0                       # street_cross: touching_streets

OPPOSITE_SIDE_TANGENT_STEP = 5.0                   # street_opposite_side: support points around the projection
OPPOSITE_SIDE_NB_RESULTS = 100                     # street_opposite_side: k
OPPOSITE_SIDE_MAX_ALONG = 80.0                     # street_opposite_side
OPPOSITE_SIDE_MAX_CROSS = 40.0                     # street_opposite_side
OPPOSITE_SIDE_REFERENCE_RADIUS = 40                # street_opposite_side: reference POI near the street

# Keyword arguments passed to each template of LIST_TEMPLATE by make_question_composite
TEMPLATE_PARAMS = {
    "make_question_area_border": {"min_size": AREA_BORDER_MIN_SIZE, "band": AREA_BORDER_BAND},
    "make_question_area_inside": {"min_size": AREA_INSIDE_MIN_SIZE},
    "make_question_area_direction": {"list_direction": DIRECTIONS, "min_size": AREA_DIRECTION_MIN_SIZE},
    "make_question_point_between": {"corridor_m": BETWEEN_CORRIDOR, "distance_max": DISTANCE_MAX,
                                    "nb_ancres_candidates": BETWEEN_NB_CANDIDATE_ANCHORS},
    "make_question_point_near_cardinal": {"list_direction": DIRECTIONS, "distance_max": DISTANCE_MAX},
    "make_question_point_near_metric": {"list_distance": METRIC_DISTANCES},
    "make_question_point_near": {"distance_max": DISTANCE_MAX},
    "make_question_point_towards": {"half_width": TOWARDS_HALF_WIDTH, "distance_max": DISTANCE_MAX},
    "make_question_street_along": {"max_dist": DISTANCE_STREET_MAX, "min_length": STREET_ALONG_MIN_LENGTH},
    "make_question_street_cross": {"tol": STREET_CROSS_TOLERANCE, "distance_max": DISTANCE_MAX},
}
