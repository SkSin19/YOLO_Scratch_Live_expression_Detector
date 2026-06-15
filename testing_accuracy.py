import glob
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import confusion_matrix

IMG_SIZE = 128
GRID_SIZE = 8

y_true = []
y_pred = []

EMOTIONS = [
    "angry",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
    "unknown"
]


def yolo_loss(y_true, y_pred):

    obj_mask = y_true[..., 4:5]

    xy_loss = tf.reduce_mean(
        obj_mask *
        tf.square(y_true[..., 0:2] - y_pred[..., 0:2])
    )

    wh_loss = tf.reduce_mean(
        obj_mask *
        tf.square(y_true[..., 2:4] - y_pred[..., 2:4])
    )

    obj_loss = tf.reduce_mean(
        obj_mask *
        tf.square(y_true[..., 4:5] - y_pred[..., 4:5])
    )

    noobj_loss = tf.reduce_mean(
        (1 - obj_mask) *
        tf.square(y_true[..., 4:5] - y_pred[..., 4:5])
    )

    class_loss = tf.reduce_mean(
        obj_mask *
        tf.square(y_true[..., 5:] - y_pred[..., 5:])
    )

    return (
        5.0 * xy_loss +
        5.0 * wh_loss +
        obj_loss +
        0.5 * noobj_loss +
        class_loss
    )


def create_target(label_path):

    target = np.zeros(
        (GRID_SIZE, GRID_SIZE, 14),
        dtype=np.float32
    )

    with open(label_path, "r") as f:

        for line in f.readlines():

            cls, xc, yc, w, h = map(
                float,
                line.strip().split()
            )

            cls = int(cls)

            gx = min(
                int(xc * GRID_SIZE),
                GRID_SIZE - 1
            )

            gy = min(
                int(yc * GRID_SIZE),
                GRID_SIZE - 1
            )

            target[gy, gx, 4] = 1.0
            target[gy, gx, 5 + cls] = 1.0

    return target


def get_true_class(target):

    cells = np.where(target[..., 4] == 1)

    if len(cells[0]) == 0:
        return None

    gy = cells[0][0]
    gx = cells[1][0]

    return np.argmax(
        target[gy, gx, 5:]
    )


def get_pred_class(pred):

    best_conf = -1
    best_class = None

    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):

            conf = pred[gy, gx, 4]

            if conf > best_conf:

                best_conf = conf

                best_class = np.argmax(
                    pred[gy, gx, 5:]
                )

    return best_class


print("Loading model...")

model = tf.keras.models.load_model(
    "best_model.keras",
    custom_objects={
        "yolo_loss": yolo_loss
    }
)

print("Model loaded.")

valid_images = glob.glob(
    "data/valid/images/*"
)

correct = 0
total = 0

pred_counts = np.zeros(9, dtype=int)
true_counts = np.zeros(9, dtype=int)

for i, img_path in enumerate(valid_images):

    if i % 100 == 0:
        print(
            f"Processed {i}/{len(valid_images)}"
        )

    img = cv2.imread(img_path)

    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = img.astype(np.float32)

    img = np.expand_dims(
        img,
        axis=0
    )

    pred = model.predict(
        img,
        verbose=0
    )[0]

    label_path = (
        Path(img_path).parent.parent
        / "labels"
        / (Path(img_path).stem + ".txt")
    )

    target = create_target(
        str(label_path)
    )

    true_class = get_true_class(target)
    pred_class = get_pred_class(pred)

    # store for confusion matrix
    y_true.append(true_class)
    y_pred.append(pred_class)

    true_counts[true_class] += 1
    pred_counts[pred_class] += 1

    if pred_class == true_class:
        correct += 1

    total += 1

accuracy = correct / total

print("\n====================")
print("Accuracy:", accuracy)
print("====================\n")

print("True distribution:")

for i, c in enumerate(true_counts):
    print(
        EMOTIONS[i],
        ":",
        c
    )

print("\nPredicted distribution:")

for i, c in enumerate(pred_counts):
    print(
        EMOTIONS[i],
        ":",
        c
    )

print("\n====================")
print("CONFUSION MATRIX")
print("====================")

cm = confusion_matrix(
    y_true,
    y_pred
)

print(cm)

print("\nRows = TRUE")
print("Columns = PREDICTED")

print("\nClass Mapping:")

for i, emotion in enumerate(EMOTIONS):
    print(f"{i}: {emotion}")