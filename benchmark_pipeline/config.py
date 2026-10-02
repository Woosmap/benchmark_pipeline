from benchmark_pipeline.generator.features import *

VILLE = "Paris, France"
OUTPUT_PATH = "pois_paris.parquet"

# Tags OSM à récupérer par catégorie
CATEGORIES = {
    "amenity": ["cafe", "restaurant", "bar", "pub", "park"],
    "tourism": ["hotel"],
    "railway": ["station", "subway_entrance"],
    "leisure": ["park", "nature_reserve", "garden"],
    "landuse": ["grass"],
}
 
FEATURES_OSM = {
    "general": ["opening_hour", "dog_accepted", "close_to_the_seine", "price", "toilets", "wifi", "decoration", "atmosphere", "disabled access"],
    "restaurant": ["cuisine_type"],
    "hotel": ["star_rating", "breakfast"],
    "cafe": ["coworking"],
}

FEATURES = [Category(), CookingType(), OutDoorSeating(), InDoorSeating()]


STREETS_TAGS = {"highway": ["primary", "secondary", "tertiary", "unclassified",
                    "residential", "pedestrian", "living_street"]}

AREA_TAGS = {"leisure": ["park", "nature_reserve", "garden"],
        "boundary": "administrative"
        }

CATEGORY_TAGS = ["amenity", "shop", "tourism", "leisure", "station", "public_transport", "railway"]

# Ancres spatiales pour calculer les distances
ANCRES = {
    "metro": {"railway": "subway_entrance"},
    "parc":  {"leisure": "park"},
}

BBOX = (2.24, 48.8, 2.41, 48.9)
BBOX_TEST = (2.346, 48.853, 2.362, 48.862)
BBOX_OUEST = (2.24, 48.8156, 2.3470, 48.9022)
BBOX_EST   = (2.3470, 48.8156, 2.41, 48.9022)

NAME_TAGS = ["poi_name", "name", "name:fr", "brand", "operator", "alt_name"]