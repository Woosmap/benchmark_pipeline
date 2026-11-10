def dicts_to_lists(df):
    # Parquet only takes str dict keys: node i is stored at position i - 1, each dic_query value as a list
    def as_list(value):
        return value if isinstance(value, list) else [] if value is None else [value]
    # anchors of a multihop question follow its graph like dic_query; those of a single-hop question are flat lists
    anchors = {column: df[column].map(lambda d: [d[node] or [] for node in sorted(d)] if isinstance(d, dict) else d)
               for column in ("anchors_type", "anchors_id") if column in df}
    return df.assign(
        template=df["template"].map(lambda d: [d[node] for node in sorted(d)] if isinstance(d, dict) else None),
        dic_query=df["dic_query"].map(lambda d: [as_list(d[node]) for node in sorted(d)] if isinstance(d, dict) else None),
        **anchors)