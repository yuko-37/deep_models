import tensorflow as tf
import tensorflow.keras.layers as layers
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel(logging.ERROR)


def data_augmenter():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ])


def create_model():
    input_shape = (150, 150, 3)
    pretrained_model = tf.keras.applications.InceptionV3(
        include_top=False,
        input_shape= input_shape
    )
    pretrained_model.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmenter()(inputs)
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    x = pretrained_model(x, training=False)
    x = layers.GlobalAvgPool2D()(x)
    x = layers.Dropout(rate=0.2)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model(inputs, output)


def create_datasets():
    return tf.keras.preprocessing.image_dataset_from_directory(
        "datasets",
        labels="inferred",
        batch_size=100,
        image_size=(150, 150),
        shuffle=True,
        seed=37,
        validation_split=0.1,
        subset="both"
    )


def visualize(history):
    plt.plot(history.history['loss'], label="Training Loss")
    plt.plot(history.history['val_loss'], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def train():
    training_ds, validation_ds = create_datasets()
    training_ds = training_ds.ignore_errors(log_warning=False)
    validation_ds = validation_ds.ignore_errors(log_warning=False)
    model = create_model()
    model.summary()
    base_learning_rate = 0.001
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    initial_epochs = 10
    history = model.fit(
        training_ds,
        validation_data=validation_ds,
        epochs=initial_epochs
    )
    model_name = "model_parameters/inception_v3_cats_dogs_classifier.keras"
    model.save(model_name)
    print(f"Model is saved to {model_name}")
    visualize(history)


def get_image_tensor(img_path):
    img_array = None
    try:
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((150, 150))
        img_array = np.array(img, dtype=np.float32)
    except Exception as e:
        print(f"Failed to process image {img_path}", e)

    return img_array

def images_to_tensor(image_paths):
    img_arrays = [get_image_tensor(path) for path in image_paths]
    img_arrays = [arr for arr in img_arrays if arr is not None]
    batch_array = np.stack(img_arrays, axis=0)
    return tf.convert_to_tensor(batch_array, dtype=tf.float32)


def predict_batch(image_paths):
    model_path = "model_parameters/inception_v3_cats_dogs_classifier.keras"
    if not os.path.exists(model_path):
        print(f"No model found")
        return
    model = tf.keras.models.load_model(model_path)
    inputs = images_to_tensor(image_paths)
    prediction = model.predict(inputs)
    prediction = prediction.flatten()
    for i, img_path in enumerate(image_paths):
        print(f"'{img_path}' is detected as a {'CAT' if prediction[i] < 0.5 else 'DOG'} " 
            f"with y = {prediction[i]:.4f}")


if __name__ == '__main__':
    # train()
    img_paths = [f"images/{f}" for f in os.listdir("images")]
    predict_batch(img_paths)
