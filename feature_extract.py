import json
import os

import cv2
import numpy as np

from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class Sift:
    def __init__(self, name='sift'):
        self.name = name

    def extract_sift(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints= sift.detect(gray, None)
        keypoints, descriptors = sift.compute(gray, keypoints)

        descriptors = descriptors.mean(0)
        return descriptors

    def extract(self, image):
        sift_features = self.extract_sift(image)

        return sift_features


class Hog:
    def __init__(self, name='hog'):
        self.name = name

    def extract_hog(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hog = cv2.HOGDescriptor()
        descriptors = hog.compute(gray)
        return descriptors

    def extract(self, image):
        hog_features = self.extract_hog(image)

        return hog_features


class Orb:
    def __init__(self, name='orb'):
        self.name = name

    def extract_orb(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        descriptors = descriptors.mean(0)
        return descriptors

    def extract(self, image):
        orb_features = self.extract_orb(image)

        return orb_features


def extract_features(images, data_dir, mode):
    if mode == 'sift':
        extractor = Sift()
    elif mode == 'hog':
        extractor = Hog()
    elif mode == 'orb':
        extractor = Orb()

    features = []
    for image in range(len(images)):
        # print(data_dir[image])
        feature = extractor.extract(images[image])
        features.append(feature)
    return np.array(features)


def load_data(base="./resources/dataset/", mode='train'):
    image_list = []
    label_list = []

    if mode == 'train' or mode == 'val':
        with open(os.path.join(base, mode + "_.json")) as f:
            data_dir = json.load(f)

        for dic in data_dir:
            img_path = os.path.join(base, dic["dir"].replace("\\", "/"))
            label = int(dic["label"])
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

            image_list.append(img)
            label_list.append(label)

        return image_list, label_list, data_dir

    else:
        path = os.path.join(base, 'test_set')
        image_paths = os.listdir(path)

        for name in image_paths:
            image_path = os.path.join(path, name)
            img = cv2.imread(image_path)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

            image_list.append(img)

        return image_list


if __name__ == "__main__":
    # ----------- Initialize model --------------
    svm_model = SVC(kernel='rbf')
    kmeans_model = KMeans(n_clusters=16)
    random_model = RandomForestClassifier()
    model = LogisticRegression()

    # ----------- Load Training Set --------------
    X_train_images, y_train, data_dir = load_data(mode='train')

    X_train = extract_features(X_train_images, data_dir, mode='hog')

    # ----------- Load Validation Set --------------
    X_test_images, y_test, data_dir = load_data(mode='val')

    X_test = extract_features(X_test_images, data_dir, mode='hog')

    # ----------- Train model --------------
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print("Logistic Regression Accuracy:：", accuracy)

    # svm_model.fit(X_train, y_train)
    # svm_y_pred = svm_model.predict(X_test)
    # svm_accuracy = accuracy_score(y_test, svm_y_pred)
    # print("SVM Accuracy:", svm_accuracy)

    # kmeans_model.fit(X_train, y_train)
    # kmeans_y_pred = kmeans_model.predict(X_test)
    # kmeans_accuracy = accuracy_score(y_test, kmeans_y_pred)
    # print("K-Means Accuracy:", kmeans_accuracy)
    #
    random_model.fit(X_train, y_train)
    random_y_pred = random_model.predict(X_test)
    random_accuracy = accuracy_score(y_test, random_y_pred)
    print("Random Model Accuracy:", random_accuracy)


