import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow_hub as tf_hub
import numpy as np
import matplotlib.pyplot as plt
BATCH_SIZE = 4
IMG_HEIGHT = 256
IMG_WIDTH = 256


def augment(image, label):
    """
    image augmentation, imputing additions channels to grayscale image
    :param image:
    :param label:
    :return:
    """
    image = tf.image.grayscale_to_rgb(image)
    return image, label


def visualize_image(train_ds, batch_size, class_names):
    """
    visualize the images from the tensorflow iterator
    :param train_ds:
    :param batch_size:
    :param class_names:
    :return:
    """
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(batch_size):
            ax = plt.subplot(6, 6, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[np.argmax(labels[i])])
            plt.axis("off")

    # Plotting the images
    plt.show()


# Load images from local local directory to a tensorflow iterator --> feed it to model
ds_train = tf.keras.utils.image_dataset_from_directory(directory='archive/brain_tumor_dataset/', labels="inferred",
                                                       label_mode="binary", color_mode='grayscale',
                                                       batch_size=BATCH_SIZE, image_size=(IMG_HEIGHT, IMG_WIDTH),
                                                       shuffle=True, seed=12, validation_split=0.1, subset="training")


ds_validation = tf.keras.utils.image_dataset_from_directory(directory='archive/brain_tumor_dataset/', labels="inferred",
                                                            label_mode="binary", color_mode='grayscale',
                                                            batch_size=BATCH_SIZE, image_size=(IMG_HEIGHT, IMG_WIDTH),
                                                            shuffle=True, seed=12, validation_split=0.1, subset="validation")

class_names = ds_train.class_names

# add channel to image so as to work with the model
ds_train = ds_train.map(augment)
ds_validation = ds_validation.map(augment)

# slicing the second last layer of rest net weights
url = "https://tfhub.dev/google/imagenet/resnet_v1_101/feature_vector/5"
base_model = tf_hub.KerasLayer(url, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base_model.trainable = False

# Add last layers for tumor classification
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Parameters for model training
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy', 'Precision', 'Recall'])

# Train and evaluate the results
model.fit(ds_train, batch_size=BATCH_SIZE, epochs=10)
model.evaluate(ds_validation, batch_size=BATCH_SIZE)