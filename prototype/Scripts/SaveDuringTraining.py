import numpy as np
from keras_preprocessing import image
import os

def saveDuringTraining(batchsize, generated_images, generator, discriminator, save_dir, step, firstdim):

    randIndex = np.random.randint(1, batchsize)

    saved_image = generated_images[randIndex]

    img = image.array_to_img(saved_image * 255., scale=False)

    img.save(os.path.join(save_dir, str(step) + '.png'))

    generator.save(os.path.join(save_dir, str(step) + "Generator" + str(firstdim) + "Dense.h5"))

    discriminator.save(os.path.join(save_dir, str(step) + "Discriminator" + str(firstdim) + "Dense.h5"))
