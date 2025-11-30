from spacy import displacy
import spacy
from src.utils import load_txt, load_json
from src.config import (
    DATA_PATH,
    PRED_PATH,
    TRUE_PATH,
    PRED_COLORS,
    TRUE_COLORS,
    ENTITY_TYPES,
)

def visualize_entities(text: str, true_entities, pred_entities):
    # prepare data
    nlp = spacy.blank("fr")  # create blank Language class
    doc = nlp.make_doc(text)

    # prepare annotations in format (start_character, end_character, label)
    spans = []
    for start, end, label in true_entities:
        span = doc.char_span(start, end, label=f"{label}_true")
        if span is not None:
            spans.append(span)

    for start, end, label in pred_entities:
        span = doc.char_span(start, end, label=f"{label}_pred")
        if span is not None:
            spans.append(span)

    # stocker les spans dans doc.spans (clé arbitraire, ex: "sc")
    doc.spans["sc"] = spans
    
    colors ={}
    ents = []
    i = 0
    for label in ENTITY_TYPES:
        i +=1
        if i == len(PRED_COLORS):
            i = 0
        ents.append(f"{label}_pred")
        ents.append(f"{label}_true")
        colors[f"{label}_pred"] = PRED_COLORS[i]
        colors[f"{label}_true"] = TRUE_COLORS[i]
        
    # display (open localhost:5100 in your browser)
    options = {
        "ents": ents,
        "colors": colors,
    }
    displacy.serve(doc, style="span", host="localhost", port=5100, options=options)



if __name__ == "__main__":
    # Example usage
    text = load_txt(DATA_PATH)
    pred_json = load_json(PRED_PATH)
    true_json = load_json(TRUE_PATH)
    
    pred_entities = [(ent["spans"][0], ent["spans"][1], ent.get("label", "UNK"))for ent in pred_json["entities"]]
    true_entities = [(ent["spans"][0], ent["spans"][1], ent.get("label", "UNK"))for ent in true_json["entities"]]
    text = load_txt(DATA_PATH)
    visualize_entities(text, true_entities, pred_entities)

