from keras.layers import BatchNormalization, Input, Dense, Reshape, Flatten, Dropout, Conv2DTranspose, Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam


# build Discriminator
def build_discriminator(shapeOfImages):
    """
    Build discriminator network
    """
    # strides instead of MaxPooling
    model = Sequential([
        #Flatten(input_shape=shapeOfImages),
        Conv2D(32,(3,3),padding='same',input_shape=shapeOfImages),
        LeakyReLU(alpha=0.2),
        MaxPooling2D((2,2)),
        Dropout(0.3),
        #Dense(256),
        Conv2DTranspose(32, (3, 3), padding='same'),
        LeakyReLU(alpha=0.2),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        #Dense(128),
        #LeakyReLU(alpha=0.2),
        Flatten(),
        Dropout(0.4),
        Dense(1, activation='sigmoid'),
    ], name='discriminator')

    model.summary()

    image = Input(shape=shapeOfImages)
    output = model(image)

    return Model(image, output)