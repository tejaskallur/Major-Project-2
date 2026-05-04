import base64
import io
import os
import urllib.request
from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image
from image_caption import (
    IMAGE_FOLDER as CAPTION_IMAGE_FOLDER,
    img_features,
    get_random_dataset_image,
    generate_caption_for_dataset_image,
    generate_caption_from_bytes,
)
from image_retrival import search_images_payload

app = Flask(__name__, static_folder=".", static_url_path="")


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/styles.css")
def styles():
    return send_from_directory(".", "styles.css")


@app.route("/app.js")
def js():
    return send_from_directory(".", "app.js")


@app.route("/api/image/<path:image_name>")
def dataset_image(image_name):
    image_path = os.path.join(CAPTION_IMAGE_FOLDER, image_name)
    if not os.path.exists(image_path):
        return jsonify({"error": "Image not found"}), 404
    return send_file(image_path)


@app.route("/api/random-image", methods=["GET"])
def random_image():
    image_name, _ = get_random_dataset_image()
    return jsonify(
        {
            "image_name": image_name,
            "image_url": f"/api/image/{image_name}",
        }
    )


def _caption_from_data_url(data_url):
    if "," not in data_url:
        raise ValueError("Invalid data URL")
    payload = data_url.split(",", 1)[1]
    image_bytes = base64.b64decode(payload)
    return generate_caption_from_bytes(image_bytes)


@app.route("/api/caption", methods=["POST"])
def generate_caption():
    mode = request.form.get("mode") or (request.json or {}).get("mode")
    if not mode:
        return jsonify({"error": "mode is required"}), 400

    try:
        if mode == "dataset":
            image_name = request.form.get("image_name") or (request.json or {}).get("image_name")
            if not image_name:
                image_name, _ = get_random_dataset_image()
            if image_name not in img_features:
                return jsonify({"error": "Invalid dataset image"}), 400
            caption = generate_caption_for_dataset_image(image_name)
            return jsonify({"caption": caption, "image_name": image_name})

        if mode == "local":
            uploaded = request.files.get("image")
            if uploaded is None:
                return jsonify({"error": "image file is required"}), 400
            caption = generate_caption_from_bytes(uploaded.read())
            return jsonify({"caption": caption})

        if mode == "webcam":
            data_url = request.form.get("data_url") or (request.json or {}).get("data_url")
            if not data_url:
                return jsonify({"error": "data_url is required"}), 400
            caption = _caption_from_data_url(data_url)
            return jsonify({"caption": caption})

        if mode == "url":
            image_url = request.form.get("image_url") or (request.json or {}).get("image_url")
            if not image_url:
                return jsonify({"error": "image_url is required"}), 400
            with urllib.request.urlopen(image_url, timeout=15) as response:
                image_bytes = response.read()
            Image.open(io.BytesIO(image_bytes)).convert("RGB")
            caption = generate_caption_from_bytes(image_bytes)
            return jsonify({"caption": caption})

        return jsonify({"error": "Unsupported mode"}), 400
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@app.route("/api/search", methods=["GET"])
def search():
    query = request.args.get("q", "").strip()
    top_k = int(request.args.get("top_k", 8))
    if not query:
        return jsonify({"results": []})

    try:
        items = search_images_payload(query, top_k=top_k)
        results = []
        for item in items:
            results.append(
                {
                    "label": item["caption"],
                    "score": round(item["similarity"], 4),
                    "image": item["image"],
                    "src": f"/api/image/{item['image']}",
                }
            )
        return jsonify({"results": results})
    except Exception as error:
        return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
