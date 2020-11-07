# imports
import numpy as np
from keras.layers import BatchNormalization, Input, Dense, Reshape, Flatten, Dropout
from keras.models import Model, load_model
from keras.optimizers import Adam
import os
from keras_preprocessing import image
from Scripts.ImageToNumpy import imageToNumpy
from Model.Discriminator import build_discriminator
from Model.Generator import build_generator
from Scripts.JudgePerformance import judgePerformance
from Scripts.SaveDuringTraining import saveDuringTraining

# global definitions
firstdim, seconddim, thirddim = 64, 64, 3

shapeOfImages = firstdim, seconddim, thirddim

source = "../Data/Train/" + str(firstdim) + str(seconddim) + "/"

save_dir = "Results/"

steps = 15000

saved_generator = None

saved_discriminator = None

#number for testing with Inception Score
numberOfImages = None

def train(generator, discriminator, combined, steps, batchsize):
    '''
    Train the GAN system
    :param generator: generator
    :param discriminator: discriminator
    :param combined: stacked generator and discriminator
    combined network used for training
    :param steps: number of alternating steps for training
    :param batch_size: size of the minibatch
    '''
    # load dataset

    x_train = imageToNumpy(source, shapeOfImages)

    np.random.shuffle(x_train)

    # rescale to [-1,1] interval for faster approaching  to global minima at error surface
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5

    latent_dim = generator.input_shape[1]

    for step in range(steps):
        # Train discriminator

        # create labels for discriminator
        real = np.ones((batchsize, 1))
        fake = np.zeros((batchsize, 1))

        # add noise for prevent
        #real += 0.05 * np.random.random(real.shape)
        #fake += 0.05 * np.random.random(fake.shape)

        # Select a random batch of images
        real_images = x_train[np.random.randint(0, x_train.shape[0], batchsize)]

        # Random batch of noise
        noise = np.random.normal(0, 1, (batchsize, latent_dim))

        # generate a batch of new images
        generated_images = generator.predict(noise)

        # train the discriminator
        discriminator_real_loss = discriminator.train_on_batch(real_images, real)
        discriminator_fake_loss = discriminator.train_on_batch(generated_images, fake)
        discriminator_loss = 0.5 * np.add(discriminator_fake_loss, discriminator_real_loss)

        # train the generator
        # random latent vector z
        noise = np.random.normal(0, 1, (batchsize, latent_dim))

        # train the generator
        generator_loss = combined.train_on_batch(noise, real)


        # display progress
        print("%d [Discriminator loss: %.4f%%, acc.: %.2f%%] [Generator loss: %.4f%%]" % (
            step, discriminator_loss[0], 100 * discriminator_loss[1], generator_loss))

        # save some pics during process
        if step % 1000 == 0 and step >= 7000 and step != 0:

            #save generator, discriminator and one random image
            saveDuringTraining(batchsize=batchsize, generated_images=generated_images,
                               generator=generator, discriminator=discriminator,
                               save_dir=save_dir, step=step, firstdim=firstdim)


    # save model
    generator.save(os.path.join(save_dir, "model.h5"))

if __name__ == "__main__":
    latent_dim = 64

    # build and compile the discriminator

    if saved_discriminator is None:
        discriminator = build_discriminator(shapeOfImages)
    else:
        discriminator = load_model(saved_discriminator)

    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

    # build the generator
    # use pretrained model
    if saved_generator is None:
        generator = build_generator(latent_dim,shapeOfImages)
    else:
        generator = load_model(saved_generator)

    # generator input z
    z = Input(shape=(latent_dim,))
    generated_image = generator(z)

    # only train the generator for the combined model
    discriminator.trainable = False

    # the discriminator takes generated images as input and determines validity
    real_or_fake = discriminator(generated_image)

    # stack the generator and discriminator in a combined model
    # trains the generator to deceive the discriminator
    combined = Model(z, real_or_fake)
    combined.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

    # train the GAN system,batchsize=128
    train(generator=generator, discriminator=discriminator, combined=combined, steps=steps, batchsize=128)

    print("Finished Training, waiting for Inception Score")

    judgePerformance(generator, numberOfImages=100)
