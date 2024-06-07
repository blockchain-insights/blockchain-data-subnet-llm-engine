def transform_result_set(result):
    if not result:
        return {"columns": [], "content": []}

    # Dynamically generate columns based on the keys in the first result item
    columns = [{"name": key, "label": key.title().replace('_',' ')} for key in result[0].keys()]

    # Populate the content based on the result set
    content = [{**item} for item in result]

    return [{"columns": columns, "content": content}]