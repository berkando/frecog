import pickle
from operator import itemgetter

import cv2
import numpy as np
import openface
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from constants import predictor_model, network_model

image_dim = 96
net = openface.TorchNeuralNet(network_model, imgDim=image_dim, cuda=False)


def train(working_directory):
    print("Loading embeddings.")
    fname = "{}/labels.csv".format(working_directory)
    labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
    labels = map(itemgetter(1),
                 map(os.path.split,
                     map(os.path.dirname, labels)))  # Get the directory.
    fname = "{}/reps.csv".format(working_directory)
    embeddings = pd.read_csv(fname, header=None).as_matrix()
    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)

    svm = SVC(C=1, kernel='linear', probability=True).fit(embeddings, labelsNum)
    with open("{}/classifier.pkl".format(working_directory), 'w') as f:
        pickle.dump((le, svm), f)


def load_classifier_model(path_to_classifier_model, loading_func=pickle.load):
    with open(path_to_classifier_model, 'r') as f:
        (le, svm) = loading_func(f)
    return (le, svm)


def infer(labels, predictor, aligned_face):
    rep = net.forward(aligned_face).reshape(1, -1)

    predictions = predictor.predict_proba(rep)[0]
    maxI = np.argmax(predictions)
    person = labels.inverse_transform(maxI)
    confidence = predictions[maxI]

    return dict(name=person, confidence=confidence)


def read_image(image_path):
    bgr_img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return rgb_img


def align_face(rgbImg):
    face_aligner = openface.AlignDlib(predictor_model)
    bb = face_aligner.getLargestFaceBoundingBox(rgbImg)
    aligned_face = face_aligner.align(image_dim, rgbImg, bb,
                                     landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    return aligned_face
