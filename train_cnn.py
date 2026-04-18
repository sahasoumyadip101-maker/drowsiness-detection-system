import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ========= YOUR FOLDER STRUCTURE =========
DATA_DIR = "."   # current folder
CATEGORIES = ["Non Drowsy", "Drowsy"]
# =========================================

IMG_SIZE = 32    # a bit larger than 32 for more detail

X = []
y = []

for label_idx, label in enumerate(CATEGORIES):
    folder = os.path.join(DATA_DIR, label)
    print("Reading folder:", folder)
    if not os.path.isdir(folder):
        print("⚠️ Folder not found:", folder)
        continue

    for filename in os.listdir(folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        file_path = os.path.join(folder, filename)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Could not read:", file_path)
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        y.append(label_idx)

X = np.array(X, dtype="float32") / 255.0
X = np.expand_dims(X, axis=-1)   # (N, 48, 48, 1)
y = np.array(y)
y = to_categorical(y, num_classes=len(CATEGORIES))

print("✅ Total images loaded:", X.shape[0])

# ---- split ----
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train:", X_train.shape, "Val:", X_val.shape)

# ---- data augmentation (only on training set) ----
datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10,
    brightness_range=(0.7, 1.3),
    horizontal_flip=True
)
datagen.fit(X_train)

# ---- stronger CNN ----
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(CATEGORIES), activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---- train with augmented data ----
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=15,                       # little more epochs
    validation_data=(X_val, y_val)
)

model.save("eye_cnn.h5")
print("💾 Model saved as eye_cnn.h5")
