import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 128
GRID_SIZE = 8

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
        tf.square(
            y_true[..., 0:2] -
            y_pred[..., 0:2]
        )
    )

    wh_loss = tf.reduce_mean(
        obj_mask *
        tf.square(
            y_true[..., 2:4] -
            y_pred[..., 2:4]
        )
    )

    obj_loss = tf.reduce_mean(
        obj_mask *
        tf.square(
            y_true[..., 4:5] -
            y_pred[..., 4:5]
        )
    )

    noobj_loss = tf.reduce_mean(
        (1 - obj_mask) *
        tf.square(
            y_true[..., 4:5] -
            y_pred[..., 4:5]
        )
    )

    class_loss = tf.reduce_mean(
        obj_mask *
        tf.square(
            y_true[..., 5:] -
            y_pred[..., 5:]
        )
    )

    return (
        5.0 * xy_loss +
        5.0 * wh_loss +
        obj_loss +
        0.5 * noobj_loss +
        class_loss
    )


print("Loading model...")

model = tf.keras.models.load_model(
    "best_model.keras",
    custom_objects={
        "yolo_loss": yolo_loss
    }
)

print("Model loaded.")


def decode_prediction(pred):

    best_conf = 0
    best_box = None
    best_class = None

    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):

            cell = pred[gy, gx]

            conf = float(cell[4])

            if conf > best_conf:

                best_conf = conf

                xc = (gx + cell[0]) / GRID_SIZE
                yc = (gy + cell[1]) / GRID_SIZE

                w = float(cell[2]) ** 2
                h = float(cell[3]) ** 2

                best_box = (
                    float(xc),
                    float(yc),
                    w,
                    h
                )

                best_class = int(
                    np.argmax(cell[5:])
                )

    return best_box, best_class, best_conf


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

print("Press Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]

    img = cv2.resize(
        frame,
        (IMG_SIZE, IMG_SIZE)
    )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = img.astype(np.float32)

    inp = np.expand_dims(
        img,
        axis=0
    )

    pred = model.predict(
        inp,
        verbose=0
    )[0]

    box, cls, conf = decode_prediction(pred)

    if box is not None and conf > 0.5:

        xc, yc, bw, bh = box

        xc *= frame_w
        yc *= frame_h

        bw *= frame_w
        bh *= frame_h

        x1 = int(xc - bw / 2)
        y1 = int(yc - bh / 2)

        x2 = int(xc + bw / 2)
        y2 = int(yc + bh / 2)

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(frame_w, x2)
        y2 = min(frame_h, y2)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = (
            f"{EMOTIONS[cls]} "
            f"{conf:.2f}"
        )

        cv2.putText(
            frame,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Expression Detection",
        frame
    )

    key = cv2.waitKey(1)

    if key & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()