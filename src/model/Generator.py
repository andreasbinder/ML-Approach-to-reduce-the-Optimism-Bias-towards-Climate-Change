# imports
import numpy as np
from keras.layers import BatchNormalization, Input, Dense, Reshape, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model, load_model
from keras import regularizers

def build_generator(latent_dim: int, shapeOfImages):
    """
    Build discriminator network
    :param latent_dim: latent vector size
    """

    model = Sequential([
        Dense(64, input_dim=latent_dim, kernel_regularizer=regularizers.l2(0.01)),
        BatchNormalization(momentum=0.8),
        LeakyReLU(alpha=0.2),

        Dense(128, kernel_regularizer=regularizers.l2(0.01)),
        BatchNormalization(momentum=0.8),
        LeakyReLU(alpha=0.2),

        Dense(256, kernel_regularizer=regularizers.l2(0.01)),
        BatchNormalization(momentum=0.8),
        LeakyReLU(alpha=0.2),

        Dense(np.prod(shapeOfImages), activation='tanh'),
        Reshape(shapeOfImages)
    ])

    model.summary()

    # the latent input vector z
    z = Input(shape=(latent_dim,))
    generated = model(z)

    # build model from the input and output
    return Model(z, generated)