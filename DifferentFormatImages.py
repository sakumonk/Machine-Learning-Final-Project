import numpy as np
from PIL import Image, ImageOps
import os
from imageprocessing_helper_functions import flatten
from sklearn.model_selection import train_test_split

#images convert to grayscale and resized
images = [f for f in os.listdir('obama') if f != ".DS_Store"]
X = [Image.open("./obama/" + image) for image in images]
for i in range(0, len(X)):
    X[i] = ImageOps.grayscale(X[i])
    X[i] = X[i].resize((243,320))


for image in X:
    print(image)
    image.show()

