from pyproj import Transformer
from shapely.geometry import box

def bbox_wgs84_to_lambert(bbox):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    min_x, min_y = transformer.transform(bbox[0], bbox[1])
    max_x, max_y = transformer.transform(bbox[2], bbox[3])
    return box(min_x, min_y, max_x, max_y)

def bbox_to_polygon(bbox):
    return box(bbox[0], bbox[1], bbox[2], bbox[3])