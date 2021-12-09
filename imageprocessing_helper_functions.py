import numpy as np
from PIL import Image, ImageFilter, ImageOps
from sklearn.model_selection import train_test_split
import numpy as np

#X is list of image object
#label is list of integer label s
#return blurred image list X_blurred, and its labels label_blurred
def blurred(X, label):
    X_blurred = []
    label_blurred = []
    for i in range(0, len(X)):
        X_blurred.append(X[i].filter(ImageFilter.BLUR))
        label_blurred.append(label[i])
    return X_blurred, label_blurred

#X is list of image object
#label is list of integer label s
#return mirror image list X_mirrored, and its labels label_mirrored
def mirrored(X, label):
    X_mirrored = []
    label_mirrored = []
    for i in range(0, len(X)):
        X_mirrored.append(ImageOps.mirror(X[i]))
        label_mirrored.append(label[i])
    return X_mirrored, label_mirrored

#X is list of image object
#return flatten images numpy array of dimension (lenght of X, 77760)
def flatten(X):
    features = np.zeros((len(X),77760))
    i = 0
    for currentImage in X:
        data1 = np.asarray(currentImage)
        data1 = data1.reshape((1, 243*320)) 
        features[i,:] = data1
        i += 1
    return features