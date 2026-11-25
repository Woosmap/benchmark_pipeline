import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from benchmark_pipeline.config import FORMAT_DIRECTION
from benchmark_pipeline.generator.relational.graph import path_subgraph, shortest_path
from benchmark_pipeline.generator.relational.relation import direction_text


def plot_graph(graph, answer, streets, format_direction=FORMAT_DIRECTION):
    """The graph on a map, with the shortest path between the two POIs of the answer.

    On the map: the OSM streets in grey, the POIs and the known relations in light blue, the relations
    of the shortest path in orange, the relation to find in red (dashed). The POIs of the path are
    numbered from 0 (start of the answer) to the end of the path (end of the answer).
    On the right: the POIs of the path and each of its relations, with its distance and direction.
    """
    coordinates = graph["coordinates"]
    path = path_subgraph(shortest_path(graph["relations"], answer["from"], answer["to"]), answer["from"])
    path_pois = path["pois"]   # in order, from the start to the end of the answer

    figure, (map_ax, legend_ax) = plt.subplots(1, 2, figsize=(20, 10), width_ratios=[2, 1], layout="constrained")

    # the background map: the streets around the POIs
    xs = [x for x, y in coordinates.values()]
    ys = [y for x, y in coordinates.values()]
    margin = 300
    nearby_streets = streets.cx[min(xs) - margin:max(xs) + margin, min(ys) - margin:max(ys) + margin]
    nearby_streets.plot(ax=map_ax, color="lightgrey", linewidth=0.5)

    # the known relations and the POIs, in light blue
    for relation in graph["relations"]:
        map_ax.add_patch(FancyArrowPatch(coordinates[relation["from"]], coordinates[relation["to"]],
                                         arrowstyle="->", mutation_scale=10, color="tab:blue", alpha=0.25))
    map_ax.scatter(xs, ys, color="tab:blue", s=15, zorder=3)

    # the relations of the shortest path, in orange
    for relation in path["relations"]:
        map_ax.add_patch(FancyArrowPatch(coordinates[relation["from"]], coordinates[relation["to"]],
                                         arrowstyle="->", mutation_scale=20, color="darkorange", linewidth=2.5,
                                         shrinkA=10, shrinkB=10, zorder=4))

    # the relation to find, in red
    map_ax.add_patch(FancyArrowPatch(coordinates[answer["from"]], coordinates[answer["to"]],
                                     arrowstyle="->", mutation_scale=20, color="red", linewidth=2.5,
                                     linestyle="--", shrinkA=10, shrinkB=10, zorder=4))

    # the POIs of the path, numbered: the two POIs of the answer in red
    for number, poi in enumerate(path_pois):
        x, y = coordinates[poi]
        color = "darkorange"
        if poi == answer["from"] or poi == answer["to"]:
            color = "red"
        map_ax.text(x, y, str(number), color="white", fontsize=9, fontweight="bold", ha="center", va="center",
                    zorder=5, bbox={"facecolor": color, "edgecolor": "white", "boxstyle": "circle"})
    map_ax.set_title(f"Shortest path from 0 to {len(path["relations"])}: {len(path["relations"])} relations")
    map_ax.set_aspect("equal")
    map_ax.set_axis_off()

    # on the right, the whole path written out
    lines = [f"To find: from 0 to {len(path["relations"])}, {answer['distance']} m in direction "
             f"{direction_text(answer, format_direction)}", "", "POIs of the path:"]
    for number, poi in enumerate(path_pois):
        lines.append(f"  {number}. {poi}")
    lines.append("")
    lines.append(f"Relations of the path ({len(path["relations"])}):")
    for number, relation in enumerate(path["relations"], start=1):
        start = path_pois.index(relation["from"])
        end = path_pois.index(relation["to"])
        lines.append(f"  {number}. from {start} to {end}: {relation['distance']} m in direction "
                     f"{direction_text(relation, format_direction)}")
    legend_ax.text(0, 1, "\n".join(lines), va="top", fontsize=10, family="monospace")
    legend_ax.set_axis_off()
    plt.show()
