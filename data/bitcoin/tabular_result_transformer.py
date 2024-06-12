def transform_result_set(result):
    if not result:
        return [{"columns": [], "rows": []}]

    # Dynamically generate columns based on the keys in the first result item
    columns = [{"name": key, "label": key.title()} for key in result[0].keys()]

    # Populate the content based on the result set
    rows = [{**item} for item in result]

    return [{"columns": columns, "rows": rows}]