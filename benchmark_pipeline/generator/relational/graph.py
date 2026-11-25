def shortest_path(relations, start, end):
    """The relations of the shortest path between two POIs, in order, reading the relations both ways.

    A relation A -> B also leads from B to A (same distance, opposite direction).
    Returns None if the two POIs are not connected.
    """
    # for each POI, the relations touching it, in one direction or the other
    poi_relations = {}
    for relation in relations:
        poi_relations.setdefault(relation["from"], []).append(relation)
        poi_relations.setdefault(relation["to"], []).append(relation)

    # breadth-first search: for each POI reached, keep the relation it was reached through
    reached_by = {start: None}
    to_visit = [start]
    while to_visit:
        poi = to_visit.pop(0)
        if poi == end:
            break
        for relation in poi_relations.get(poi, []):
            if relation["from"] == poi:
                neighbor = relation["to"]
            else:
                neighbor = relation["from"]
            if neighbor not in reached_by:
                reached_by[neighbor] = relation
                to_visit.append(neighbor)
    if end not in reached_by:
        return None

    # walk back from the end to the start
    path = []
    poi = end
    while poi != start:
        relation = reached_by[poi]
        path.insert(0, relation)
        if relation["to"] == poi:
            poi = relation["from"]
        else:
            poi = relation["to"]
    return path


def path_subgraph(path, start):
    """The subgraph of a path: its POIs in order from `start`, and its relations in order.

    A relation of the path can be read backwards: the next POI is then its `from`.
    """
    pois = [start]
    for relation in path:
        if relation["from"] == pois[-1]:
            pois.append(relation["to"])
        else:
            pois.append(relation["from"])
    return {"pois": pois, "relations": path}
