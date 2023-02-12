import json
import os
import sys
import random
import math
import requests

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 16
IMG_HEIGHT = 256
IMG_WIDTH = 256
seed = 123


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def show_image(image):
    """
    Display a sample image
    :param image:
    :return:
    """
    plt.figure(figsize=(3, 3))
    plt.imshow(tf.squeeze(image))
    plt.gray()
    plt.show()


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


# Read validation images with the same seed value when preparing training dataset
ds_validation = tf.keras.utils.image_dataset_from_directory(directory='archive/brain_tumor_dataset/', labels="inferred",
                                                            label_mode="binary", color_mode='grayscale',
                                                            batch_size=BATCH_SIZE, image_size=(IMG_HEIGHT, IMG_WIDTH),
                                                            shuffle=True, seed=seed, validation_split=0.1,
                                                            subset="validation")
# Class names - no & yes
class_names = ds_validation.class_names

# Array to store sample test image from test batch
image = []
label = []

# add channel to image so as to work with the model
ds_validation = ds_validation.map(augment)

# random test image
test_image_index = random.randint(0, BATCH_SIZE-1)

# Read a test image
for images, labels in ds_validation.take(1):
    image = images[test_image_index].numpy()
    label = labels[test_image_index].numpy().squeeze()
    break

# Prepare image for tensorflow model serving format
image = np.expand_dims(image, axis=0)
data = json.dumps({
    "instances": image.tolist()
})
headers = {"content-type": "application/json"}

# Model prediction using REST
response = requests.post('http://localhost:8605/v1/models/brain_tumor_classification:predict', data=data,
                         headers=headers)

# Get the response JSON
result = (sigmoid((response.json()['predictions'][0][0])))

# Display the result
print(f'Predicted label {result}')
print(f'True label {label}')

show_image(image)