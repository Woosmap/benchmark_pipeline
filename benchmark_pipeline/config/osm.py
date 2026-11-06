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
