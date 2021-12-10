import numpy as np
from PIL import Image, ImageOps
import os
from imageprocessing_helper_functions import flatten
from sklearn.model_selection import train_test_split


def FiveMoreImages():

    #images convert to grayscale and resized
    images = [f for f in os.listdir('Augmented_With_5_More') if f != ".DS_Store"]
    X = [Image.open("./Augmented_With_5_More/" + image) for image in images]
    for i in range(0, len(X)):
        X[i] = ImageOps.grayscale(X[i])
        X[i] = X[i].resize((243,320))



    label = []
    for image in images:
        if "putin" in image: #subject 16 is Putin
            label.append(16)
        elif "obama" in image: #subject 17 is Obama
            label.append(17)
        elif "prayuth" in image: #subject 18 is General Prayuth
            label.append(18)
        elif "Depp" in image: #subject 19 is Johnny Depp
            label.append(19)
        elif "kim" in image: #subject 20 is Kim Jong Un
            label.append(20)

    label = np.array(label)
    features = flatten(X)
    return features, label

FiveMoreImages()