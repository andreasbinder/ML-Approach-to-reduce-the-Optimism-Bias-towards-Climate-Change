from Scripts.InceptionScore import calculate_inception_score
import numpy as np


def judgePerformance(generator, numberOfImages):
    latent_dim = generator.input_shape[1]

    noise = np.random.normal(0, 1, (numberOfImages, latent_dim))

    generated_images = generator.predict(noise)

    score, _ = calculate_inception_score(generated_images)

    print("Inception Score: " + str(score))
