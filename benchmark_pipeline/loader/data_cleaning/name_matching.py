"""Make POI, street and area names unique, so that each name points to a single entity."""

import unicodedata
import pandas as pd


def disambiguate_names(
        df_osm,
        df_street,
        df_area
    ):
    """Rename streets and areas whose name is exactly a POI or area name.

    Args:
        df_osm (DataFrame): POIs, with `poi_name`. Left unchanged.
        df_street (DataFrame): Streets, with `street_name`. Modified in place: "Rue " is
            prepended when the name equals a POI or area name and does not contain "Rue".
        df_area (DataFrame): Areas, with `area_name` and `category`. Modified in place:
            the category is prepended when the name equals a POI name.

    Returns:
        tuple: (df_osm, df_street, df_area), the same objects as the inputs.
    """
    for _, poi_row in df_osm.iterrows():
        matches = df_street[df_street["street_name"] == poi_row["poi_name"]]
        for street_label, street_row in matches.iterrows():
            street_name = street_row["street_name"]
            if "Rue" not in street_name:
                df_street.at[street_label, "street_name"] = f"Rue {street_name}"

        area_similarity = df_area[df_area["area_name"]==poi_row["poi_name"]]
        for area_num_row, area_row in area_similarity.iterrows():
            area_name = area_row["area_name"]
            area_category = area_row["category"]
            df_area.at[area_num_row, "area_name"] = f"{area_category} {area_name}"

    for _, area_row in df_area.iterrows():
            matches = df_street[df_street["street_name"] == area_row["area_name"]]
            for street_label, street_row in matches.iterrows():
                street_name = street_row["street_name"]
                if "Rue" not in street_name:
                    df_street.at[street_label, "street_name"] = f"Rue {street_name}"
    return df_osm, df_street, df_area

def name_filtering_poi(df_osm, key):
    """Drop rows without a name, then rows whose normalized name is already used by an earlier row.

    Names are compared without accents, case or repeated spaces.

    Args:
        df_osm (DataFrame): Table to filter, left unchanged.
        key (str): Name column.

    Returns:
        DataFrame: Kept rows sorted by index.
    """
    def normaliser(s):
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        return " ".join(s.lower().split())

    df_osm = df_osm.copy()
    df_osm[key + "_norm"] = df_osm[key].dropna().map(normaliser)
    a_nom = df_osm[key].notna()
    df_filtre = df_osm[a_nom].drop_duplicates(subset=key + "_norm", keep="first").sort_index()
    return df_filtre.drop(columns=key + "_norm")

def drop_removed_rows(df_all, df_all_filtered, df_osm, df_street, df_area):
    """Drop from each table the rows that the name filter removed from `df_all`.

    Args:
        df_all (DataFrame): Stacked table before filtering, with `id` and `source`.
        df_all_filtered (DataFrame): `df_all` after `name_filtering_poi`, same index labels.
        df_osm (DataFrame): POIs, with `poi_id`.
        df_street (DataFrame): Streets, with `street_id`.
        df_area (DataFrame): Areas, with `area_id`.

    Returns:
        tuple: (df_osm, df_street, df_area) without the removed rows.
    """
    removed = df_all[~df_all.index.isin(df_all_filtered.index)]
    print(f"lignes supprimées par source : {removed['source'].value_counts().to_dict()}")
    def drop(table, id_col, source):
        removed_ids = removed.loc[removed["source"] == source, "id"]
        return table[~table[id_col].isin(removed_ids)]

    return (
        drop(df_osm, "poi_id", "poi"),
        drop(df_street, "street_id", "street"),
        drop(df_area, "area_id", "area"),
    )

def name_matching_cleaning(df_osm, df_street, df_area):
    """Make names unique within and across the POI, street and area tables.

    Exact matches with a POI or area name are first renamed by `disambiguate_names`;
    the remaining collisions keep the area, then the street, then the POI.

    Args:
        df_osm (DataFrame): POIs, with `poi_id` and `poi_name`.
        df_street (DataFrame): Streets, with `street_id` and `street_name`.
        df_area (DataFrame): Areas, with `area_id`, `area_name` and `category`.

    Returns:
        tuple: (df_all, (df_osm, df_street, df_area)). `df_all` stacks the kept rows of
            the three tables under shared `id`, `name` and `source` columns.
    """
    df_osm_name_filter = name_filtering_poi(df_osm, "poi_name")
    df_street_name_filter = name_filtering_poi(df_street, "street_name")
    df_area_name_filter = name_filtering_poi(df_area, "area_name")
    df_osm_clean, df_street_clean, df_area_clean = disambiguate_names(df_osm_name_filter, df_street_name_filter, df_area_name_filter)

    df_all = pd.concat([
        df_area_clean.rename(columns={"area_id": "id", "area_name": "name"}).assign(source="area"),
        df_street_clean.rename(columns={"street_id": "id", "street_name": "name"}).assign(source="street"),
        df_osm_clean.rename(columns={"poi_id": "id", "poi_name": "name"}).assign(source="poi"),
    ], ignore_index=True)

    premieres = ["id", "name", "source", "geometry"]
    df_all = df_all[premieres + [c for c in df_all.columns if c not in premieres]]

    df_all_name_filter = name_filtering_poi(df_all, key="name")
    return df_all_name_filter, drop_removed_rows(df_all, df_all_name_filter, df_osm_clean, df_street_clean, df_area_clean)