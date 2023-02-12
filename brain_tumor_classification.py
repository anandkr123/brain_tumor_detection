import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow_hub as tf_hub
import numpy as np
import matplotlib.pyplot as plt
BATCH_SIZE = 16
IMG_HEIGHT = 256
IMG_WIDTH = 256
seed = 123


def augment(image, label):
    """
    image augmentation, imputing additions channels to grayscale image
    :param image:
    :param label:
    :return:
    """
    image = tf.cast(image, dtype=tf.float32) / tf.constant(255, dtype=tf.float32)
    image = tf.image.grayscale_to_rgb(image)

    return image, label


def visualize_image(ds, class_names):
    """
    visualize the images from the tensorflow iterator
    :param train_ds:
    :param batch_size:
    :param class_names:
    :return:
    """
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(5, 5))

    for images, labels in ds.take(1):
        for i in range(3):
            for j in range(3):
                ax[i][j].imshow(images[i * 3 + j].numpy().astype("uint8"))
                class_index = labels[i * 3 + j].numpy().squeeze().astype("uint8")
                ax[i][j].set_title(class_names[class_index])
    plt.show()


# Load images from local local directory to a tensorflow iterator --> feed it to model
ds_train = tf.keras.utils.image_dataset_from_directory(directory='archive/brain_tumor_dataset/', labels="inferred",
                                                       label_mode="binary", color_mode='grayscale',
                                                       batch_size=BATCH_SIZE, image_size=(IMG_HEIGHT, IMG_WIDTH),
                                                       shuffle=True, seed=seed, validation_split=0.1, subset="training")


ds_validation = tf.keras.utils.image_dataset_from_directory(directory='archive/brain_tumor_dataset/', labels="inferred",
                                                            label_mode="binary", color_mode='grayscale',
                                                            batch_size=BATCH_SIZE, image_size=(IMG_HEIGHT, IMG_WIDTH),
                                                            shuffle=True, seed=seed, validation_split=0.1, subset="validation")

class_names = ds_train.class_names


# add channel to image so as to work with the model
ds_train = ds_train.map(augment)
ds_validation = ds_validation.map(augment)

# Take one batch from dataset and display the images
# visualize_image(ds_validation, class_names)

# slicing the second last layer of rest net weights
url = "https://tfhub.dev/google/imagenet/resnet_v1_101/feature_vector/5"
base_model = tf_hub.KerasLayer(url, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base_model.trainable = False

# Add last layers for tumor classification
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dense(1000, activation='relu', trainable=True),
    tf.keras.layers.Dense(1)
])

# Parameters for model training
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy', 'Precision', 'Recall'])
#
# # Train and evaluate the results
model.fit(ds_train, batch_size=BATCH_SIZE, epochs=10)
model.save('brain_tumor_model/1/')

# model.evaluate(ds_validation, batch_size=BATCH_SIZE)