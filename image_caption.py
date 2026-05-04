import io
import os
import pickle
import random
import tempfile
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image

IMAGE_FOLDER = "D:/image_features (2)/archive/flickr30k_images/flickr30k_images"
FEATURE_FILE = "D:/image_features (2)/image_features (2).pkl"
ENC_PATH = "D:/image_features (2)/encoder_attention.weights.h5"
DEC_PATH = "D:/image_features (2)/decoder_attention.weights.h5"
TOKENIZER_PATH = "D:/image_features (2)/tokenizer.pkl"

EMBEDDING_DIM = 256
UNITS = 512
MAX_LEN = 20
BEAM_WIDTH = 5

tokenizer = pickle.load(open(TOKENIZER_PATH, "rb"))
vocab_size = len(tokenizer.word_index) + 1

with open(FEATURE_FILE, "rb") as f:
    img_features = pickle.load(f)


class CNN_Encoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc = tf.keras.layers.Dense(EMBEDDING_DIM)

    def call(self, x):
        x = self.fc(x)
        return tf.expand_dims(x, 1)


class BahdanauAttention(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(UNITS)
        self.W2 = tf.keras.layers.Dense(UNITS)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden))
        attn = tf.nn.softmax(self.V(score), axis=1)
        context = tf.reduce_sum(attn * features, axis=1)
        return context


class RNN_Decoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM)
        self.gru = tf.keras.layers.GRU(UNITS, return_sequences=True, return_state=True)
        self.fc1 = tf.keras.layers.Dense(UNITS)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention()

    def call(self, x, features, hidden):
        context = self.attention(features, hidden)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        output, state = self.gru(x, initial_state=hidden)
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)
        return x, state


encoder = CNN_Encoder()
decoder = RNN_Decoder()
sample_feature = tf.random.uniform((1, 2048))
sample_seq = tf.random.uniform((1, 1), maxval=vocab_size, dtype=tf.int32)
_ = decoder(sample_seq, encoder(sample_feature), tf.zeros((1, UNITS)))
encoder.load_weights(ENC_PATH)
decoder.load_weights(DEC_PATH)

cnn_model = InceptionV3(weights="imagenet")
cnn_model = tf.keras.Model(cnn_model.input, cnn_model.layers[-2].output)


def extract_feature(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return cnn_model.predict(img, verbose=0)


def generate_caption_core(feature):
    feature = encoder(feature)
    hidden = tf.zeros((1, UNITS))
    sequences = [[["<start>"], 0.0, hidden]]

    for _ in range(MAX_LEN):
        all_candidates = []
        for seq, score, hidden_state in sequences:
            last_word = tokenizer.word_index.get(seq[-1], 0)
            seq_input = tf.expand_dims([last_word], 0)
            preds, new_hidden = decoder(seq_input, feature, hidden_state)
            probs = tf.nn.softmax(preds[0]).numpy()
            top_ids = probs.argsort()[-BEAM_WIDTH:]

            for idx in top_ids:
                word = tokenizer.index_word.get(idx, "")
                if word in ["", "<pad>", "<start>"]:
                    continue
                if word == "<end>":
                    all_candidates.append([seq + [word], score, new_hidden])
                    continue
                if word in seq:
                    continue
                new_seq = seq + [word]
                new_score = score + np.log(probs[idx] + 1e-10)
                all_candidates.append([new_seq, new_score, new_hidden])

        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:BEAM_WIDTH]
        if not sequences:
            break

    if not sequences:
        return "No caption generated"

    best_seq = sequences[0][0]
    words = [w for w in best_seq if w not in ["<start>", "<end>", "<pad>", ""]]
    words = list(dict.fromkeys(words))
    clean = []
    for w in words:
        if not clean or clean[-1] != w:
            clean.append(w)
    words = clean
    if "and" in words:
        words = words[:words.index("and")]
    sentence = " ".join(words)
    return sentence.capitalize() + "." if sentence else "No caption generated"


def refine_caption(caption):
    words = caption.split()

    replacements = {
        "man": "person",
        "woman": "person",
        "men": "people",
        "women": "people"
    }

    clean_words = []
    for w in words:
        clean_words.append(replacements.get(w, w))

    # remove duplicates
    final_words = []
    for w in clean_words:
        if w not in final_words:
            final_words.append(w)

    return " ".join(final_words)


def generate_caption_from_path(img_path):
    feature = extract_feature(img_path)
    caption = generate_caption_core(feature)
    return refine_caption(caption)


def generate_caption_from_bytes(image_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(image_bytes)
        temp_path = temp_file.name
    try:
        return generate_caption_from_path(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def generate_caption_from_pil(pil_image):
    buffer = io.BytesIO()
    pil_image.convert("RGB").save(buffer, format="JPEG")
    return generate_caption_from_bytes(buffer.getvalue())


def get_random_dataset_image():
    img_name = random.choice(list(img_features.keys()))
    img_path = os.path.join(IMAGE_FOLDER, img_name)
    return img_name, img_path


def generate_caption_for_dataset_image(img_name):
    caption = generate_caption_core(np.expand_dims(img_features[img_name], 0))
    return refine_caption(caption)


if __name__ == "__main__":
    print("Model loaded successfully.")
    image_name, _ = get_random_dataset_image()
    print("Random image:", image_name)
    print("Caption:", generate_caption_for_dataset_image(image_name))