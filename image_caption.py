"""
Image Captioning using InceptionV3 + LSTM (Kaggle Ready - FINAL)
"""

import os
import gc
import re
import pickle
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image

# ===============================
# KAGGLE PATHS (IMPORTANT)
# ===============================

BASE_DIR = "/kaggle/input/datasets/raghudinkavijaykumar/flickr-images-dataset"

dataset_images_path = "/kaggle/input/datasets/raghudinkavijaykumar/flickr-images-dataset/flickr30k_images"
captions_file = BASE_DIR + "/results.csv"

# Save outputs here
WORK_DIR = "/kaggle/working"

features_cache = WORK_DIR + "/image_features.pkl"
encoder_model_file = WORK_DIR + "/encoder_model.weights.h5"
decoder_model_file = WORK_DIR + "/decoder_model.weights.h5"

# ===============================
# FORCE DELETE OLD CACHE (IMPORTANT)
# ===============================

if os.path.exists(features_cache):
    print("Deleting old cached features...")
    os.remove(features_cache)

# ===============================
# CONFIG
# ===============================

img_height = 224
img_width = 224

BATCH_SIZE = 8
EPOCHS = 10   # reduce for speed (increase later)

EMBEDDING_DIM = 256
UNITS = 256
TOP_K = 5000
FEATURE_DIM = 2048
MAX_CAPTION_LENGTH = 20

validation_split = 0.2

# ===============================
# RANDOM IMAGE FUNCTION
# ===============================

def get_random_image(folder_path):
    images = [img for img in os.listdir(folder_path) if img.endswith(".jpg")]
    return os.path.join(folder_path, random.choice(images))

# ===============================
# CAPTION PREPROCESS
# ===============================

def preprocess_caption(caption):
    caption = caption.lower()
    caption = re.sub(r"[^a-z ]", "", caption)
    caption = re.sub(r"\s+", " ", caption).strip()
    return "<start> " + caption + " <end>"

# ===============================
# LOAD CAPTIONS (FIXED)
# ===============================

print("Loading captions...")

images_captions_dict = {}

with open(captions_file, "r", encoding="utf-8") as f:
    
    next(f)  # skip header

    for line in f:
        parts = line.strip().split("|")

        if len(parts) < 3:
            continue

        image_name = parts[0].strip()
        caption_text = parts[2].strip()

        caption = preprocess_caption(caption_text)

        images_captions_dict.setdefault(image_name, []).append(caption)

print("Total images:", len(images_captions_dict))

# ===============================
# FEATURE EXTRACTOR
# ===============================

def build_feature_extractor():
    base = tf.keras.applications.InceptionV3(
        include_top=False,
        weights="imagenet",
        input_shape=(img_height, img_width, 3),
    )
    out = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    return tf.keras.Model(base.input, out)

# ===============================
# EXTRACT FEATURES
# ===============================

print("Checking sample image path:")

for key in list(images_captions_dict.keys())[:5]:
    print(os.path.join(dataset_images_path, key))

print("Extracting features...")

extractor = build_feature_extractor()
images_features = {}

for img_name in tqdm(images_captions_dict.keys()):
    img_path = os.path.join(dataset_images_path, img_name)

    try:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = tf.keras.applications.inception_v3.preprocess_input(img)

        feature = extractor.predict(img, verbose=0)
        images_features[img_name] = feature[0]

    except:
        continue

# SAVE FEATURES
with open(features_cache, "wb") as f:
    pickle.dump(images_features, f)

print("Features ready:", len(images_features))

# ===============================
# DATA PREP
# ===============================

image_filenames = list(images_features.keys())

train_imgs, test_imgs = train_test_split(
    image_filenames,
    test_size=validation_split,
    random_state=42
)

def build_pairs(img_list):
    X, Y = [], []

    for img in img_list:
        for cap in images_captions_dict[img]:
            X.append(images_features[img])
            Y.append(cap)

    return X, Y

X_train, y_train_raw = build_pairs(train_imgs)

# ===============================
# TOKENIZER
# ===============================

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=TOP_K, oov_token="<unk>")
tokenizer.fit_on_texts(y_train_raw)

tokenizer.word_index["<pad>"] = 0
tokenizer.index_word[0] = "<pad>"

sequences = tokenizer.texts_to_sequences(y_train_raw)

y_train = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding="post")

vocab_size = len(tokenizer.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices((np.array(X_train), y_train))
dataset = dataset.shuffle(1000).batch(BATCH_SIZE)

# ===============================
# MODEL
# ===============================

class CNN_Encoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc = tf.keras.layers.Dense(EMBEDDING_DIM)

    def call(self, x):
        return self.fc(x)

class RNN_Decoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM)
        self.lstm = tf.keras.layers.LSTM(UNITS, return_sequences=True)
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, features):
        x = self.embedding(x)
        features = tf.expand_dims(features, 1)
        x = tf.concat([features, x], axis=1)
        x = self.lstm(x)
        return self.fc(x)

enc = CNN_Encoder()
dec = RNN_Decoder()

optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# ===============================
# TRAIN
# ===============================

print("Training started...")

for epoch in range(EPOCHS):
    total_loss = 0

    for img_feat, target in dataset:
        with tf.GradientTape() as tape:
            features = enc(img_feat)
            preds = dec(target[:, :-1], features)

            loss = loss_fn(target[:, 1:], preds[:, 1:, :])

        variables = enc.trainable_variables + dec.trainable_variables
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))

        total_loss += loss

    print(f"Epoch {epoch+1}, Loss: {total_loss.numpy()}")

# SAVE MODELS
enc.save_weights(encoder_model_file)
dec.save_weights(decoder_model_file)

print("Training Done ✅")

# ===============================
# GENERATE CAPTION
# ===============================

def generate_caption(image_path):

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224,224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.inception_v3.preprocess_input(img)

    feature = extractor.predict(img, verbose=0)
    feature = enc(feature)

    caption = "<start>"

    for i in range(MAX_CAPTION_LENGTH):
        seq = tokenizer.texts_to_sequences([caption])[0]
        seq = tf.keras.preprocessing.sequence.pad_sequences([seq], maxlen=MAX_CAPTION_LENGTH)

        pred = dec(seq, feature)
        pred_id = np.argmax(pred[0][i])

        word = tokenizer.index_word.get(pred_id, "")

        if word == "<end>":
            break

        caption += " " + word

    return caption.replace("<start>", "")

# ===============================
# DEMO
# ===============================

sample_path = get_random_image(dataset_images_path)

print("Random Image:", sample_path)


img = Image.open(sample_path)
plt.imshow(img)
plt.axis("off")

caption = generate_caption(sample_path)

plt.title(caption)
plt.show()

print("\nGenerated Caption:", caption)