import glob
import numpy as np
import tensorflow as tf
from pathlib import Path
import cv2

print("Starting dataset loading")

train_images = glob.glob("data/train/images/*")
valid_images = glob.glob("data/valid/images/*")

print("Train images:", len(train_images))
print("Valid images:", len(valid_images))

IMG_SIZE = 128
GRID_SIZE = 8
BATCH_SIZE = 32

def create_target(label_path):
    NUM_CLASSES = 9

    target = np.zeros(
        (GRID_SIZE, GRID_SIZE, 5 + NUM_CLASSES),
        dtype=np.float32
    )

    with open(label_path, "r") as f:

        for line in f.readlines():

            cls, xc, yc, w, h = map(
                float,
                line.strip().split()
            )

            cls = int(cls)

            grid_x = int(xc * GRID_SIZE)
            grid_y = int(yc * GRID_SIZE)

            grid_x = min(grid_x, GRID_SIZE - 1)
            grid_y = min(grid_y, GRID_SIZE - 1)

            cell_x = xc * GRID_SIZE - grid_x
            cell_y = yc * GRID_SIZE - grid_y

            target[grid_y, grid_x, 0] = cell_x
            target[grid_y, grid_x, 1] = cell_y
            target[grid_y, grid_x, 2] = np.sqrt(w)
            target[grid_y, grid_x, 3] = np.sqrt(h)

            target[grid_y, grid_x, 4] = 1.0
        
            target[grid_y, grid_x, 5 + cls] = 1.0

    return target


def load_sample(img_path):

    img_path = img_path.numpy().decode()

    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)

    label_path = (
        Path(img_path).parent.parent
        / "labels"
        / (Path(img_path).stem + ".txt")
    )

    target = create_target(str(label_path))

    return img, target


def tf_load_sample(img_path):

    image, target = tf.py_function(
        load_sample,
        [img_path],
        [tf.float32, tf.float32]
    )

    image.set_shape((IMG_SIZE, IMG_SIZE, 3))
    target.set_shape((GRID_SIZE, GRID_SIZE, 14))

    return image, target

train_images = glob.glob("data/train/images/*")
np.random.shuffle(train_images)
train_images = train_images[:3000]

valid_images = glob.glob("data/valid/images/*")
valid_images = valid_images[:700]



train_data = (
    tf.data.Dataset
    .from_tensor_slices(train_images)
    .shuffle(len(train_images))
    .map(
        tf_load_sample,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

validation_data = (
    tf.data.Dataset
    .from_tensor_slices(valid_images)
    .map(
        tf_load_sample,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(128,128,3)),
    tf.keras.layers.Rescaling(1./255),

    # Block 1
    tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same'
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(negative_slope=0.1),
    tf.keras.layers.MaxPool2D(
        pool_size=(2,2),
        strides=(2,2)
    ),

    # Block 2
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same'
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(negative_slope=0.1),
    tf.keras.layers.MaxPool2D(
        pool_size=(2,2),
        strides=(2,2)
    ),

    # Block 3
    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same'
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(negative_slope=0.1),

    # Block 4
    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same'
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(negative_slope=0.1),
    tf.keras.layers.MaxPool2D(
        pool_size=(2,2),
        strides=(2,2)
    ),

    # Block 5
    tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same'
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(negative_slope=0.1),

    # Block 6
    tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same'
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(negative_slope=0.1),
    tf.keras.layers.MaxPool2D(
        pool_size=(2,2),
        strides=(2,2)
    ),

    # Block 7
    tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same'
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(negative_slope=0.1),

    # Block 8
    tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same'
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(negative_slope=0.1),

    # Detection Head
    tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same'
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(negative_slope=0.1),

    # Prediction Layer
    tf.keras.layers.Conv2D(
        filters=14,
        kernel_size=(1,1),
        strides=(1,1),
        padding='same'
    )
])

model.add(tf.keras.layers.Activation("sigmoid"))

def yolo_loss(y_true, y_pred):

    obj_mask = y_true[..., 4:5]

    # box loss
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

    # objectness loss
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

    # classification loss
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


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),
    loss=yolo_loss
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_model.keras",
    save_best_only=True,
    monitor="val_loss"
)

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=20,
    callbacks=[checkpoint]
)