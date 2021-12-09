from sklearn import svm
import numpy as np
from data import load
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from PIL import Image as im
from imageprocessing_helper_functions import blurred, mirrored, flatten
from sklearn.metrics import confusion_matrix



train_data, train_labels = load("yalefaces_npy", split="train")
dev_data, dev_labels = load("yalefaces_npy", split="dev")

X = []

for i in range(0, np.shape(train_data)[0]):
    array = np.reshape(train_data[i,:], (243, 320))
    data = im.fromarray(array, mode = 'L')
    X.append(data)

label = train_labels
features = train_data

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

# Compute a PCA 
n_components = 100
pca = PCA(n_components=n_components, whiten=True).fit(features_combined)

# apply PCA transformation
X_train_pca = pca.transform(features_combined)
X_test_pca = pca.transform(dev_data)

train_labels = np.ravel(labels_combined)
dev_labels = np.ravel(dev_labels)


print("Fitting the classifier to the training set")
clf = svm.SVC(C=5., gamma=0.001).fit(X_train_pca, train_labels)

y_pred = clf.predict(X_test_pca)
print(classification_report(dev_labels, y_pred))
