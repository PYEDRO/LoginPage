from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow import keras
import os
from PIL import Image
import numpy as np


import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def adapt_efficient_net() -> Model:
   
   
    inputs = layers.Input(
        shape=(224, 224, 3)
    )  
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="efficientnetb0_notop.h5")
    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)
    top_dropout_rate = 0.4
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(1, name="pred")(x)

    # Compile
    model = keras.Model(inputs, outputs, name="EfficientNet")

    return model


def open_images(inference_folder: str) -> np.ndarray:
    
    images = []
    for img in os.listdir(inference_folder):
        img_location = os.path.join(inference_folder, img)  

        with Image.open(img_location) as img:  

            img = np.array(img)
            img = img[:, :, :3]
            img = np.expand_dims(img, axis=0) 

        images.append(img)
    images_array = np.vstack(images)  
    return images_array


model = adapt_efficient_net()
model.load_weights("./data/models/eff_net.h5")
images = open_images("./inference_samples")

predictions = model.predict(images)

images_names = os.listdir("./inference_samples")
for image_name, prediction in zip(images_names, predictions):
    print(image_name, prediction)



model = adapt_efficient_net()
model.load_weights("./data/models/eff_net.h5")
images = open_images("./inference_samples")

predictions = model.predict(images)

images_names = os.listdir("./inference_samples")
for image_name, prediction in zip(images_names, predictions):
    print(image_name, prediction)