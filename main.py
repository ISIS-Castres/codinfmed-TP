
from src.utils import load_json, load_txt
from src.config import DATA_PATH, PRED_PATH, TRUE_PATH, MODELS, NOMENCLATURE_PATH, OUTPUT_DIR
from src.viz import visualize_entities
from src.call_ner import launch_gliner
from src.parsing import convert_labelstudio_to_entities
from src.matching import match_entities, display_similarity_matrix
import seaborn as sns
import matplotlib.pyplot as plt

######### Partie 1 : Modèle NER #########
text = load_txt(DATA_PATH)
launch_gliner(MODELS[0])

pred_json = load_json(PRED_PATH)
true_json = convert_labelstudio_to_entities(load_json(TRUE_PATH))

pred_entities = [(ent["spans"][0], ent["spans"][1], ent.get("label", "UNK"))for ent in pred_json["entities"]]
true_entities = [(ent["spans"][0], ent["spans"][1], ent.get("label", "UNK"))for ent in true_json["entities"]]

visualize_entities(text, true_entities, pred_entities)


######### Partie 2 : Similarité sémantique et mapping #########
# terms = load_json(NOMENCLATURE_PATH)

# # Calcul du matching
# results, similarity_matrix = match_entities(
#     entities=pred_json["entities"],
#     terms=terms,
#     top_k=3
# )

# df = display_similarity_matrix(
#     entities=pred_json["entities"], terms=terms, similarity_matrix=similarity_matrix)


# plt.figure(figsize=(20, 15))  # Increase width and height
# sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu")
# plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
# plt.yticks(rotation=0)  # Keep y-axis labels horizontal
# plt.tight_layout()  # Adjust layout so labels don’t get cut off
# plt.savefig(f"{OUTPUT_DIR}/similarity_heatmap.png", dpi=300)
