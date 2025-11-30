import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def load_json(path):
    """Charge un fichier JSON depuis un chemin."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_embeddings(data_list, model):
    """Retourne les embeddings d'une liste de textes."""
    return model.encode(data_list, convert_to_numpy=True, normalize_embeddings=True)

def display_similarity_matrix(entities, terms, similarity_matrix, max_col_width=30):
    
    # --- noms des lignes (entités) ---
    row_labels = [f"{e['entity']} ({e['label']})" for e in entities]

    # --- noms des colonnes (termes) ---
    col_labels = [f"{t['code']} - {t['name']}" for t in terms]

    # --- Construction du DataFrame ---
    df = pd.DataFrame(similarity_matrix, index=row_labels, columns=col_labels)

    # Formatage lisible
    pd.set_option('display.max_colwidth', max_col_width)
    pd.set_option('display.float_format', lambda x: f"{x:.3f}")

    print("\n====== MATRICE DE SIMILARITÉ ENTITÉS × TERMES ======\n")
    print(df)
    print("\n====================================================\n")

    return df

def match_entities(entities, terms, top_k=3):
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    # ---- Préparation des textes ----
    entity_texts = [e["entity"] for e in entities]
    term_texts = [f"{c['name']} - {c['definition']}" for c in terms]

    # ---- Embeddings ----
    entity_emb = compute_embeddings(entity_texts, model)
    term_emb = compute_embeddings(term_texts, model)

    # ---- Similarités ----
    similarity_matrix = cosine_similarity(entity_emb, term_emb)

    results = []

    for i, ent in enumerate(entities):
        # scores pour une entité donnée
        scores = similarity_matrix[i]
        
        # indices des meilleurs scores
        best_idx = scores.argsort()[::-1][:top_k]

        matches = []
        for idx in best_idx:
            matches.append({
                "entity": ent["entity"],
                "label": ent["label"],
                "term_code": terms[idx]["code"],
                "term_name": terms[idx]["name"],
                "similarity": float(scores[idx])
            })

        results.append({
            "entity": ent["entity"],
            "label": ent["label"],
            "matches": matches
        })

    return results, similarity_matrix



