import numpy as np
import os
from PIL import Image


def imageToNumpy(source,shape):
    temp = os.listdir(source)
    result = []
    for p in temp:
        image = Image.open(source + p)
        i = np.array(image)
        # 500, 500, 3
        if i.shape == shape:
            result.append(i)
    return np.stack(result)
