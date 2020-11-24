from keras.models import load_model
import numpy as np
from keras_preprocessing.image import array_to_img
import os

numberOfImages = 150

path = "/home/andreas/Documents/Seminar/appliedml_using_gan/Model/Results/9000Generator64Dense.h5"

save_dir = "/home/andreas/Desktop/res/"

if os.path.isfile(path) and numberOfImages > 0:

    model = load_model(path)

    model.summary()

    latent_dim = model.input_shape[1]

    noise = np.random.normal(0, 1, (numberOfImages, latent_dim))

    generated_images = model.predict(noise)

    im = array_to_img(generated_images[0])

    index = 0
    for im in generated_images:
        im = array_to_img(im)
        im.save(save_dir+str(index)+".png")
        index = index + 1



