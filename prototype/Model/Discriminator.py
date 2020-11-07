from keras.layers import BatchNormalization, Input, Dense, Reshape, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam


# build Discriminator
def build_discriminator(shapeOfImages):
    """
    Build discriminator network
    """

    model = Sequential([
        # 28, 28, 1
        Flatten(input_shape=shapeOfImages),
        Dense(128),
        LeakyReLU(alpha=0.3),
        Dense(64),
        Dropout(0.3),
        LeakyReLU(alpha=0.3),
        Dropout(0.3),
        Dense(1, activation='sigmoid'),
    ], name='discriminator')

    model.summary()

    image = Input(shape=shapeOfImages)
    output = model(image)

    return Model(image, output)
