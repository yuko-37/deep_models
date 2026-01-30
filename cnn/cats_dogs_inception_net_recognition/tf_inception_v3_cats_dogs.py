import tensorflow as tf
import tensorflow.keras.layers as layers
import logging
import os


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
    initial_epochs = 1
    history = model.fit(
        training_ds,
        validation_data=validation_ds,
        epochs=initial_epochs
    )
    model_name = "model_parameters/inception_v3_cats_dogs_classifier.keras"
    model.save(model_name)
    print(f"Model is saved to {model_name}")


if __name__ == '__main__':
    train()
