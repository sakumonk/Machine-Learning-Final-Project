import numpy as np
from PIL import Image
import os
import random

#yale images
images = [f for f in os.listdir('yalefaces/yalefaces') if f != ".DS_Store"]
X = [Image.open("./yalefaces/yalefaces/" + image) for image in images]

#divide into test,dev, and train (test(22 images), train(120 images), dev(23 images))
label = []
for image in images:
    if "subject11" in image:
        label.append(10)
    elif "subject02" in image:
        label.append(1)
    elif "subject03" in image:
        label.append(2)
    elif "subject04" in image:
        label.append(3)
    elif "subject05" in image:
        label.append(4)
    elif "subject06" in image:
        label.append(5)
    elif "subject07" in image:
        label.append(6)
    elif "subject08" in image:
        label.append(7)
    elif "subject09" in image:
        label.append(8)
    elif "subject10" in image:
        label.append(9)
    elif "subject15" in image:
        label.append(14)
    elif "subject12" in image:
        label.append(11)
    elif "subject13" in image:
        label.append(12)
    elif "subject14" in image:
        label.append(13)
    elif "subject01" in image:
        label.append(0)



label = np.array(label)

dev_features = np.zeros((23,77760))
test_features = np.zeros((22,77760))
train_features = np.zeros((285,77760))

dev_labels = np.zeros((23,1))
test_labels = np.zeros((22,1))
train_labels = np.zeros((285,1))

a_list = list(range(0, 165))
random.shuffle(a_list)

i = 0
for currImage in X:
    currentImage = X[a_list[i]]
    data1 = np.asarray(currentImage)
    data1 = data1.reshape((1, 243*320)) 
    if (i < 120):
        train_features[i,:] = data1
        train_labels[i] = label[a_list[i]]
    elif (i >= 120 and i < 143):
        dev_features[i-120,:] = data1
        dev_labels[i-120] = label[a_list[i]]
    elif(i >= 143):
        test_features[i-143,:] = data1
        test_labels[i-143] = label[a_list[i]]
    i += 1

#yale images
imagesmod = [f for f in os.listdir('yalefacesmod') if f != ".DS_Store"]
Xmod = [Image.open("./yalefacesmod/" + image) for image in imagesmod]

#divide into test,dev, and train (test(22 images), train(120 images), dev(23 images))
label = np.ndarray.tolist(label)
for image in imagesmod:
    if "subject11" in image:
        label.append(10)
    elif "subject02" in image:
        label.append(1)
    elif "subject03" in image:
        label.append(2)
    elif "subject04" in image:
        label.append(3)
    elif "subject05" in image:
        label.append(4)
    elif "subject06" in image:
        label.append(5)
    elif "subject07" in image:
        label.append(6)
    elif "subject08" in image:
        label.append(7)
    elif "subject09" in image:
        label.append(8)
    elif "subject10" in image:
        label.append(9)
    elif "subject15" in image:
        label.append(14)
    elif "subject12" in image:
        label.append(11)
    elif "subject13" in image:
        label.append(12)
    elif "subject14" in image:
        label.append(13)
    elif "subject01" in image:
        label.append(0)

label = np.array(label)

i = 120
for currentImage in Xmod:
    data1 = np.asarray(currentImage)
    data1 = data1.reshape((1, 243*320)) 
    train_features[i,:] = data1
    train_labels[i] = label[i]
    i += 1
    


np.save("dev.feats", dev_features)
np.save("test.feats", test_features)
np.save("train.feats", train_features)
np.save("dev.labels", dev_labels)
np.save("test.labels", test_labels)
np.save("train.labels", train_labels)
