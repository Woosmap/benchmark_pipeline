# OSM Infos colonnes
CATEGORIES = {
    "amenity": ["cafe", "restaurant", "bar", "pub"],
    "tourism": ["hotel"],
    "railway": ["station"],
    "public_transport": ["station"],
}
CATEGORIES_EXCLUDED = [
    'bench', 'waste_basket', 'bicycle_parking',
    'vending_machine', 'yes', 'parking_space',
    'ticket_validator', 'recycling'
    ]

STREETS_TAGS = {"highway": ["primary", "secondary", "tertiary", "unclassified",
                    "residential", "pedestrian", "living_street"]}

AREA_TAGS = {"leisure": ["park", "nature_reserve", "garden"],
        "boundary": "administrative"
        }
CATEGORY_TAGS = ["amenity", "shop", "tourism", "leisure", "station", "public_transport", "railway"]

NAME_TAGS = ["poi_name", "name", "name:fr", "brand", "operator", "alt_name"]
ADMIN_LEVEL = 9

#data loading choice
DISTANCE_STREET_UNION = 50
MERGE_DISTANCE_STREET = 500

# Text document of each POI (bow_poi), built by serialize_poi
BOW_INCLUDE_ADDRESS = True    # add the street and postcode of the POI
BOW_INCLUDE_CONTEXT = False   # add the names of the nearby entities (stations, parks): leaks the location into the text
BOW_INCLUDE_NAMES = False     # add the name of the POI

# cooking_type, picked by classify_cuisine_column among the OSM cuisine tags
CUISINE_PREFER = "regional"   # "regional": 'japanese;bowl;poke' -> 'japanese'; "dish": -> 'bowl'
CUISINE_KEEP_VENUE = False    # True accepts venue words ('coffee_shop', 'bar') as a fallback cuisine

USE_CACHE = False             # True rereads data/pois/pois_osm_<bbox>.parquet instead of querying OSM again
