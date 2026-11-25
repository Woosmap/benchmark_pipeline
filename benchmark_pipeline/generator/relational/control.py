import math

import pandas as pd

from benchmark_pipeline.generator.relational.relation import cardinal_point


def check_benchmark(benchmark):
    """Checks, for each question, that chaining the relations of the shortest path gives back the answer.

    Each relation is a move of `distance` meters in the direction `direction_degrees`. They are added up
    along the path (a relation read backwards counts the other way): the total move must have
    the distance and the angle of the answer, and the angle must fall in the cardinal point of the label.
    Distances are rounded to the meter and angles to the degree, so a small gap is accepted: on a relation
    of d meters, rounding moves the end by at most 0.5 m + d x 0.5° (in radians). These shifts are added up
    over the whole path to get the allowed gap.
    If this gap exceeds the distance of the answer, the angle cannot be checked: the question is
    marked as not checkable, and the angle and label counts only cover the checkable questions.
    The same sum is also done keeping only the cardinal point of each relation (north = 0°, northeast = 45°...),
    as a model reading only the cardinal points would, and the two answers found are compared.
    Returns a table with one row per question.
    """
    directions = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
    rows = []
    for question_number, row in benchmark.iterrows():
        path = row["path"]
        answer = row["answer"]

        # chain the relations of the path
        eastward = 0
        northward = 0
        allowed_shift = 0
        eastward_by_cardinals = 0
        northward_by_cardinals = 0
        for step, relation in enumerate(path["relations"]):
            angle = math.radians(relation["direction_degrees"])
            sign = 1
            if relation["from"] != path["pois"][step]:   # relation read backwards
                sign = -1
            eastward += sign * relation["distance"] * math.sin(angle)
            northward += sign * relation["distance"] * math.cos(angle)
            allowed_shift += 0.5 + relation["distance"] * math.radians(0.5)
            # the same relation, with the angle of its cardinal point instead of its degrees
            cardinal_angle = math.radians(directions.index(relation["direction_cardinal"]) * 45)
            eastward_by_cardinals += sign * relation["distance"] * math.sin(cardinal_angle)
            northward_by_cardinals += sign * relation["distance"] * math.cos(cardinal_angle)
        computed_distance = math.sqrt(eastward ** 2 + northward ** 2)
        computed_angle = math.degrees(math.atan2(eastward, northward)) % 360
        distance_by_cardinals = math.sqrt(eastward_by_cardinals ** 2 + northward_by_cardinals ** 2)
        angle_by_cardinals = math.degrees(math.atan2(eastward_by_cardinals, northward_by_cardinals)) % 360

        # the gaps with the answer, and the gaps allowed by rounding
        distance_gap = abs(computed_distance - answer["distance"])
        allowed_distance_gap = allowed_shift + 0.5
        angle_gap = abs((computed_angle - answer["direction_degrees"] + 180) % 360 - 180)

        # the angle can only be checked if the allowed shift stays below the distance of the answer
        minimal_distance = answer["distance"] - 0.5
        checkable = allowed_shift < minimal_distance
        allowed_angle_gap = None
        angle_ok = None
        cardinal_ok = None
        if checkable:
            allowed_angle_gap = math.degrees(math.asin(allowed_shift / minimal_distance)) + 0.5
            angle_ok = angle_gap <= allowed_angle_gap
            # the cardinal label: the one of the computed angle, or of a neighboring angle near a boundary
            possible_cardinals = [cardinal_point(computed_angle - allowed_angle_gap), cardinal_point(computed_angle),
                                  cardinal_point(computed_angle + allowed_angle_gap)]
            cardinal_ok = row["label_cardinal"]["direction"] in possible_cardinals
            allowed_angle_gap = round(allowed_angle_gap, 1)

        rows.append({
            "question": question_number,
            "computed_angle": round(computed_angle, 1),
            "expected_angle": answer["direction_degrees"],
            "angle_gap": round(angle_gap, 1),
            "allowed_angle_gap": allowed_angle_gap,
            "computed_distance": round(computed_distance),
            "expected_distance": answer["distance"],
            "distance_gap": round(distance_gap, 1),
            "allowed_distance_gap": round(allowed_distance_gap, 1),
            "label_cardinal": row["label_cardinal"]["direction"],
            "checkable": checkable,
            "angle_ok": angle_ok,
            "distance_ok": distance_gap <= allowed_distance_gap,
            "cardinal_ok": cardinal_ok,
            # the answer computed with the degrees, compared to the one computed with the cardinal points
            "computed_cardinal": cardinal_point(computed_angle),
            "cardinal_by_cardinals": cardinal_point(angle_by_cardinals),
            "same_cardinal": cardinal_point(computed_angle) == cardinal_point(angle_by_cardinals),
            "angle_gap_cardinals": round(abs((angle_by_cardinals - computed_angle + 180) % 360 - 180), 1),
            "distance_gap_cardinals": round(abs(distance_by_cardinals - computed_distance)),
        })
    report = pd.DataFrame(rows)
    checkable_rows = report[report["checkable"]]
    print(f"distances found: {report['distance_ok'].sum()} / {len(report)}")
    print(f"angles found: {checkable_rows['angle_ok'].sum()} / {len(checkable_rows)} checkable questions")
    print(f"cardinal labels found: {checkable_rows['cardinal_ok'].sum()} / {len(checkable_rows)} checkable questions")
    print(f"questions not checkable: {len(report) - len(checkable_rows)} "
          "(answer shorter than the gap allowed by rounding)")
    print(f"same cardinal point through the degrees and through the cardinal points: "
          f"{report['same_cardinal'].sum()} / {len(report)}")
    print(f"mean gap through the cardinal points: {report['angle_gap_cardinals'].mean():.1f}° "
          f"and {report['distance_gap_cardinals'].mean():.0f} m")
    return report
