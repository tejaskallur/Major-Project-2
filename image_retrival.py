import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import matplotlib.pyplot as plt

RESULTS_FILE = r"archive\flickr30k_images\results.csv"
IMAGE_FOLDER = r"archive\flickr30k_images\flickr30k_images"

_df = None
_vectorizer = None
_caption_vectors = None


def initialize_retrieval():
    global _df, _vectorizer, _caption_vectors
    if _df is not None and _vectorizer is not None and _caption_vectors is not None:
        return

    df = pd.read_csv(RESULTS_FILE, sep="|")
    df.columns = ["image", "index", "caption"]
    df["image"] = df["image"].astype(str).str.strip()
    df["caption"] = df["caption"].astype(str).str.strip()
    df = df.dropna(subset=["caption"])
    df = df[df["caption"] != ""]

    vectorizer = TfidfVectorizer(stop_words="english")
    caption_vectors = vectorizer.fit_transform(df["caption"])

    _df = df
    _vectorizer = vectorizer
    _caption_vectors = caption_vectors


def search_images(query, top_k=8):
    initialize_retrieval()
    query_vec = _vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, _caption_vectors).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = _df.iloc[top_indices].copy()
    results["similarity"] = similarities[top_indices]
    return results


def search_images_payload(query, top_k=8):
    results = search_images(query, top_k=top_k)
    payload = []
    for row in results.itertuples():
        payload.append(
            {
                "image": row.image,
                "caption": row.caption,
                "similarity": float(row.similarity),
                "image_path": os.path.join(IMAGE_FOLDER, row.image),
            }
        )
    return payload


def show_results(query, top_k=5):
    results = search_images(query, top_k=top_k)
    plt.figure(figsize=(15, 5))
    for i, row in enumerate(results.itertuples()):
        img_path = os.path.join(IMAGE_FOLDER, row.image)
        img = Image.open(img_path)
        plt.subplot(1, top_k, i + 1)
        plt.imshow(img)
        plt.title(row.caption[:30])
        plt.axis("off")
    plt.suptitle(f"Query: {query}")
    plt.show()


if __name__ == "__main__":
    query = input("Enter search query: ").strip()
    if query:
        show_results(query)