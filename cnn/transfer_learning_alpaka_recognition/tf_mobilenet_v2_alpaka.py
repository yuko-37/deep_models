import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as tfl
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing import image


BATCH_SIZE = 32
IMG_SIZE = (160, 160)


def load_datasets(autotune=True):
    directory = "datasets/"
    train_dataset, validation_dataset = image_dataset_from_directory(
        directory=directory,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        seed=37,
        validation_split=0.2,
        subset="both"
    )
    if autotune:
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, validation_dataset


def load_dataset():
    tds, validation_dataset = load_datasets()

    print((type(validation_dataset)))
    print(len(validation_dataset))
    class_names = validation_dataset.class_names
    print(class_names)
    plt.figure(figsize = (10, 10))
    for images, labels in validation_dataset.take(1):
        print(f"images batch size is {len(images)}")
        for i in range(9):
            ax = plt.subplot(3, 3, i+1)
            print(f"images numpy shape is {images[i].numpy().shape}, label is {labels[i]}={class_names[labels[i]]}")
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
        plt.show()

    return tds, validation_dataset


def data_augmenter():
    data_augmentation = tf.keras.Sequential()
    data_augmentation.add(tfl.RandomFlip('horizontal'))
    data_augmentation.add(tfl.RandomRotation(0.2))
    return data_augmentation


def create_model(image_shape=IMG_SIZE, data_augmentation=data_augmenter()):
    input_shape = image_shape + (3,)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    # base_model.summary()

    base_model.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tfl.GlobalAvgPool2D()(x)
    x = tfl.Dropout(rate=0.2)(x)
    outputs = tfl.Dense(units=1)(x)

    model = tf.keras.Model(inputs, outputs)
    return model


def visualize(history):
    acc = [0.] + history.history['accuracy']
    val_acc = [0.] + history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


def train_model():
    train_dataset, validation_dataset = load_dataset()
    model = create_model()
    model.summary()
    base_learning_rate = 0.001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    initial_epochs = 20
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=initial_epochs)
    model_name = "model_parameters/mobilenet_v2_alpaka_recognizer"
    model.save(f"{model_name}.keras")
    print(f"Model is saved: {model_name}.keras")
    visualize(history)


def recognize_alpaka(image_file):
    img_path = "images/" + image_file
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    model = tf.keras.models.load_model(
        "model_parameters/mobilenet_v2_alpaka_recognizer.keras")
    # model.summary()

    prediction = model.predict(x)
    probability = tf.keras.activations.sigmoid(prediction)
    probability = np.squeeze(probability)
    print(probability)
    print(f"y = {probability}, model predicts {image_file} as '{'an alpaka' if probability < 0.5 else 'a non-alpaka'}' picture.")

    # plt.imshow(img)
    # plt.show()


def explore_dataset():
    _, vd = load_datasets(autotune=False)
    # x = vd.map(lambda xs, ys: tf.keras.applications.mobilenet_v2.preprocess_input(xs))
    # x = vd.map(lambda xs, ys: xs)
    for x, y in vd.take(1):
        print(x)
        print(y)
    model = tf.keras.models.load_model(
        "model_parameters/mobilenet_v2_alpaka_recognizer.keras")
    preds = model.predict(x)
    probability = tf.keras.activations.sigmoid(preds)
    probability = np.squeeze(probability)

    for label, prob in zip(y, probability):
        print(f"{label}: {prob}")
    # print(probability)


if __name__ == '__main__':
    # train_model()

    for i in range(1, 6):
        recognize_alpaka(f"alpaka_{i}.jpeg")
