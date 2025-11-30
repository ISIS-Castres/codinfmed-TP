from src.config import FILE_NAME

def convert_labelstudio_to_entities(ls_data):
    if FILE_NAME == "example":
        return ls_data
    # Label Studio exports a list of tasks → we take the first one
    task = ls_data[0]

    text = task["data"]["text"]
    entities = []

    # Each task may have multiple annotations, each with multiple results
    for ann in task.get("annotations", []):
        for res in ann.get("result", []):
            val = res.get("value", {})
            entities.append({
                "label": val.get("labels", [""])[0],   # first label
                "spans": [val.get("start"), val.get("end")],
                "entity": val.get("text")
            })

    return {"entities": entities, "text": text}
