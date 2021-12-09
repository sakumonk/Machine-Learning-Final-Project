import numpy as np
from PIL import Image, ImageFilter, ImageOps
from sklearn.model_selection import train_test_split
import os
import numpy as np
from imageprocessing_helper_functions import blurred, mirrored, flatten

#yale images
images = [f for f in os.listdir('yalefaces') if f != ".DS_Store"]
X = [Image.open("./yalefaces/" + image) for image in images]

#divide into test,dev, and train 
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

#blur images
X_blurred, label_blurred = blurred(X, label)
label_blurred = np.array(label_blurred)
features_blurred = flatten(X_blurred)

#mirror images
X_mirrored, label_mirrored = mirrored(X, label)
label_mirrored = np.array(label_mirrored)
features_mirrored = flatten(X_mirrored)

#combined
features_combined = np.concatenate((features, features_blurred))
labels_combined = np.concatenate((label, label_blurred))
features_combined = np.concatenate((features_combined, features_mirrored))
labels_combined = np.concatenate((labels_combined, label_mirrored))




train_features, dev_features, train_labels, dev_labels = train_test_split(features_combined, labels_combined, test_size=0.25, random_state=42)
dev_features, test_features, dev_labels, test_labels = train_test_split(features_combined, labels_combined, test_size=0.5, random_state=42)

np.save("dev.feats", dev_features)
np.save("test.feats", test_features)
np.save("train.feats", train_features)
np.save("dev.labels", dev_labels)
np.save("test.labels", test_labels)
np.save("train.labels", train_labels)

