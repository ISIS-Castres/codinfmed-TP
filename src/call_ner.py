import json
from src.ner_model import NERModel
from src.config import (
    ENTITY_TYPES,
    DATA_PATH,
    MODELS,
    THRESHOLD,
    PRED_PATH,
)
from src.utils import load_txt


def launch_gliner(model_conf):
    model = NERModel(
        model_conf["name"],
        backend=model_conf["backend"],
        uniner_model_path="./universal-ner",
    )
    text = load_txt(DATA_PATH)

    result = {}
    preds = model.predict_entities(text, ENTITY_TYPES, threshold=THRESHOLD)
    entities = []
    for ent in preds:
        entities.append(
                {
                    "label": ent["label"],
                    "spans": [ent["start"], ent["end"]],
                    "entity": text[ent["start"] : ent["end"]],
                }
            )
    result["entities"] = entities
    result["text"] = text

    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PRED_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n Saved predictions to: {PRED_PATH}")


if __name__ == "__main__":
    launch_gliner(MODELS[0])