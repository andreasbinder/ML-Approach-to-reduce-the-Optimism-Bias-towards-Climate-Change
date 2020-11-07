# imports
import numpy as np
from keras.layers import MaxPooling2D, Input, Dense, Reshape, Flatten, Dropout, Conv2D, BatchNormalization, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model, load_model
from keras import regularizers


def build_generator(latent_dim: int,shapeOfImages):
    """
    Build discriminator network
    :param latent_dim: latent vector size
    """
    model = Sequential([
        Dense(64 * 16 * 16, input_dim=latent_dim, kernel_regularizer=regularizers.l2(0.01)),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),

        Reshape((16, 16, 64)),
        Conv2D(64, 5, padding='same'),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),

        Conv2DTranspose(64, 5, strides=2, padding='same'),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),

        Conv2DTranspose(64, 5, strides=2, padding='same'),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),

        Conv2DTranspose(64, 5, strides=2, padding='same'),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Conv2D(3, 7, activation='tanh', padding='same')
    ])

    model.summary()

    # the latent input vector z
    z = Input(shape=(latent_dim,))
    generated = model(z)

    # build model from the input and output
    return Model(z, generated)