# imports
import numpy as np
from keras.layers import MaxPooling2D, Input, Dense, Reshape, Flatten, Dropout, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model, load_model


def build_generator(latent_dim: int,shapeOfImages):
    """
    Build discriminator network
    :param latent_dim: latent vector size
    """
    model = Sequential([
        Dense(256, input_dim=64),
        LeakyReLU(alpha=0.2),
        Reshape((16, 16, 1)),
        Conv2D(32, (7, 7), padding='same'),
        LeakyReLU(alpha=0.2),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), padding='same'),
        LeakyReLU(alpha=0.2),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), padding='same'),
        LeakyReLU(alpha=0.2),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), padding='same'),
        LeakyReLU(alpha=0.2),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(np.prod(shapeOfImages), activation='tanh'),
        Reshape(shapeOfImages)
    ])

    model.summary()

    # the latent input vector z
    z = Input(shape=(latent_dim,))
    generated = model(z)

    # build model from the input and output
    return Model(z, generated)