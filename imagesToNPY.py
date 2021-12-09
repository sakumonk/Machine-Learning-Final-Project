import numpy as np
from PIL import Image
import os
from imageprocessing_helper_functions import flatten
from sklearn.model_selection import train_test_split

#yale images
images = [f for f in os.listdir('yalefaces') if f != ".DS_Store"]
X = [Image.open("./yalefaces/" + image) for image in images]

#divide into test,dev, and train (test(22 images), train(120 images), dev(23 images))
label = []
for image in images:
    if "subject11" in image:
        label.append(11)
    elif "subject02" in image:
        label.append(2)
    elif "subject03" in image:
        label.append(3)
    elif "subject04" in image:
        label.append(4)
    elif "subject05" in image:
        label.append(5)
    elif "subject06" in image:
        label.append(6)
    elif "subject07" in image:
        label.append(7)
    elif "subject08" in image:
        label.append(8)
    elif "subject09" in image:
        label.append(9)
    elif "subject10" in image:
        label.append(10)
    elif "subject15" in image:
        label.append(15)
    elif "subject12" in image:
        label.append(12)
    elif "subject13" in image:
        label.append(13)
    elif "subject14" in image:
        label.append(14)
    elif "subject01" in image:
        label.append(1)

label = np.array(label)
features = flatten(X)


train_features, dev_features, train_labels, dev_labels = train_test_split(features, label, test_size=0.4, random_state=42)
dev_features, test_features, dev_labels, test_labels = train_test_split(dev_features, dev_labels, test_size=0.1, random_state=42)



np.save(os.path.join("yalefaces_npy","dev.feats"), dev_features)
np.save(os.path.join("yalefaces_npy","test.feats"), test_features)
np.save(os.path.join("yalefaces_npy","train.feats"), train_features)
np.save(os.path.join("yalefaces_npy","dev.labels"), dev_labels)
np.save(os.path.join("yalefaces_npy","test.labels"), test_labels)
np.save(os.path.join("yalefaces_npy","train.labels"), train_labels)