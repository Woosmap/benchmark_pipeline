from benchmark_pipeline.generator.features import *

VILLE = "Paris, France"
ECRS = 2154
BBOX = (2.24, 48.8, 2.41, 48.9)


# Tags OSM à récupérer par catégorie
CATEGORIES = {
    "amenity": ["cafe", "restaurant", "bar", "pub"],
    "tourism": ["hotel"],
    "railway": ["station"],
    "public_transport": ["station"],
}
STREETS_TAGS = {"highway": ["primary", "secondary", "tertiary", "unclassified",
                    "residential", "pedestrian", "living_street"]}

AREA_TAGS = {"leisure": ["park", "nature_reserve", "garden"],
        "boundary": "administrative"
        }


DISTANCE_MAX = 1000 #en mètres
DISTANCE_STREET_MAX = 50


FEATURES = [Category(), CookingType(), OutDoorSeating(), InDoorSeating()]

CATEGORY_TAGS = ["amenity", "shop", "tourism", "leisure", "station", "public_transport", "railway"]

NAME_TAGS = ["poi_name", "name", "name:fr", "brand", "operator", "alt_name"]




LIST_TEMPLATE = ['make_question_area_border',
                 'make_question_area_inside', 'make_question_area_direction',
                 'make_question_point_between', 'make_question_point_near_cardinal',
                 'make_question_point_near_metric', 'make_question_point_near',
                 'make_question_point_towards', 'make_question_street_along',
                 'make_question_street_cross']

LIST_TEMPLATE_FORWARD = ['make_question_point_between', 'make_question_point_near_cardinal', 
                         'make_question_point_near_metric', 'make_question_point_near', 
                         'make_question_point_towards']

MANDATORY_FEATURES = "category"
MULTIHOP_COLUMNS = ["query_texte", "features", "dic_query", "values", "answers", "template"]