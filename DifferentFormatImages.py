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

def TenMoreImages():

    #images convert to grayscale and resized
    images = [f for f in os.listdir('Augmented_With_10_More') if f != ".DS_Store"]
    X = [Image.open("./Augmented_With_10_More/" + image) for image in images]
    for i in range(0, len(X)):
        X[i] = ImageOps.grayscale(X[i])
        X[i] = X[i].resize((243,320))



    label = []
    for image in images:
        if "berbatov" in image: #subject 21 is berbatov
            label.append(21)
        elif "churchill" in image: #subject 22 is churchill
            label.append(22)
        elif "cruise" in image: #subject 23 is tom cruise
            label.append(23)
        elif "dimaria" in image: #subject 24 is dimaria
            label.append(24)
        elif "drogba" in image: #subject 25 is drogba
            label.append(25)
        elif "klopp" in image: #subject 26 is klopp
            label.append(26)
        elif "mj" in image: #subject 27 is michael jackson
            label.append(27)
        elif "ronaldinho" in image: #subject 28 is ronaldinho
            label.append(28)
        elif "rooney" in image: #subject 29 is rooney
            label.append(29)
        elif "xi" in image: #subject 30 is xi
            label.append(30)

    label = np.array(label)
    features = flatten(X)
    return features, label

