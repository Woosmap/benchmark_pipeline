import math

from benchmark_pipeline.config import FORMAT_DIRECTION


def cardinal_point(angle):
    """The cardinal point of an angle in degrees (0° = north, clockwise).

    Each of the eight directions covers 45°: north goes from -22.5° to 22.5°, northeast from 22.5° to 67.5°, etc.
    """
    directions = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
    return directions[int((angle % 360 + 22.5) // 45) % 8]


def compute_relation(x_start, y_start, x_end, y_end):
    """Distance (in meters) and direction from a start point to an end point, the direction as two labels.

    The coordinates are already in meters. The angle goes from north to the arrow start -> end,
    clockwise (0° = north, 90° = east, ...), and is given in two ways:
    rounded to the degree, and as a cardinal point.
    """
    eastward = x_end - x_start
    northward = y_end - y_start
    distance = math.sqrt(eastward ** 2 + northward ** 2)
    angle = math.degrees(math.atan2(eastward, northward)) % 360
    return round(distance), round(angle) % 360, cardinal_point(angle)


def direction_text(relation, format_direction=FORMAT_DIRECTION):
    """The direction of a relation as written, depending on `format_direction`: "219°" or "southwest"."""
    if format_direction == "degrees":
        return f"{relation['direction_degrees']}°"
    if format_direction == "cardinal":
        return relation["direction_cardinal"]
    raise ValueError('format_direction must be "degrees" or "cardinal"')
