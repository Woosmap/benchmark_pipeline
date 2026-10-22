def find_features()

def make_question_multihop(question, df_area, nb_hop):
    features = find_features(question["pois"])
    area = find_area(question["pois"])
    anchors = find_achors(area, df_area)

