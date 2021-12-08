import numpy as np
from data import load
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report


train_data, train_labels = load("yalefaces_npy", split="train")
dev_data, dev_labels = load("yalefaces_npy", split="dev")

# Compute a PCA 
n_components = 100
pca = PCA(n_components=n_components, whiten=True).fit(train_data)
# apply PCA transformation
X_train_pca = pca.transform(train_data)
X_test_pca = pca.transform(dev_data)

train_labels = np.ravel(train_labels)
dev_labels = np.ravel(dev_labels)


print("Fitting the classifier to the training set")
clf = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=100, verbose=True, early_stopping=True).fit(X_train_pca, train_labels)

y_pred = clf.predict(X_test_pca)
print(classification_report(dev_labels, y_pred))