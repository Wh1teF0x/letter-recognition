import os
import pickle
import warnings

import cv2
import numpy as np

import image_helper
from config import *


class Model:
    syn0 = None
    syn1 = None
    data = None
    labels = None

    def __init__(self):
        np.random.seed(1)
        self.syn0 = 2 * np.random.random((INPUT_SIZE, HIDDEN_SIZE)) - 1
        self.syn1 = 2 * np.random.random((HIDDEN_SIZE, OUTPUT_SIZE)) - 1

    def f(self, x, deriv=False):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if deriv:
                return self.f(x) * (1 - self.f(x))
            return 1 / (1 + np.exp(-x))

    def save_model(self):
        with open(MODEL_FILE, 'wb') as file:
            model = {'syn0': self.syn0, 'syn1': self.syn1}
            pickle.dump(model, file)

    def load_model(self, path_to_model=MODEL_FILE):
        with open(path_to_model, 'rb') as file:
            model = pickle.load(file)
            self.syn0 = model['syn0']
            self.syn1 = model['syn1']

    def load_train_data(self, path_to_data):
        self.data = []
        self.labels = []
        letters = os.listdir(path_to_data)
        for letter in letters:
            items = os.listdir(f'{DATASET_FOLDER}/{letter}')
            for item in items:
                data_item = (cv2.imread(f'{DATASET_FOLDER}/{letter}/{item}', cv2.IMREAD_GRAYSCALE) / 255).flatten()
                data_target = np.zeros(OUTPUT_SIZE)
                data_target[OUTPUTS_MAP[letter]] = 1
                self.data.append(data_item)
                self.labels.append(data_target)
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def train(self):
        X = self.data
        y = self.labels
        for i in range(ITER_COUNT):
            l0 = X
            l1 = self.f(np.dot(l0, self.syn0))
            l2 = self.f(np.dot(l1, self.syn1))
            l2_error = y - l2
            if (i % (ITER_COUNT // 10)) == 0:
                print(f'Iteration {i} out of {ITER_COUNT}')
                print(f'Error: {str(np.mean(np.abs(l2_error)))}')
            l2_delta = l2_error * self.f(l2, deriv=True)
            l1_error = l2_delta.dot(self.syn1.T)
            l1_delta = l1_error * self.f(l1, deriv=True)
            self.syn0 += l0.T.dot(l1_delta)
            self.syn1 += l1.T.dot(l2_delta)
        print('Finish')

    def predict_y(self, X, prepare=True):
        if prepare:
            X = cv2.bitwise_not(X)
            X = X / 255
            X = X.flatten()
        layer_0 = X
        layer_1 = self.f(np.dot(layer_0, self.syn0))
        layer_2 = self.f(np.dot(layer_1, self.syn1))
        return np.argmax(layer_2)


def test_train():
    model = Model()
    model.load_train_data(DATASET_FOLDER)
    model.train()
    model.save_model()


def test_predict():
    letters = image_helper.letters_extract(TEST_FILE)
    out = image_helper.contour_letters(TEST_FILE)
    model = Model()
    model.load_model(MODEL_FILE)
    for letter in letters:
        cv2.imshow(f'{letter[2][1]}', letter[2])
        prediction = model.predict_y(letter[2].flatten())
        print(prediction)
        value = TARGET_ARRAY[prediction[0]]
        cv2.putText(out, f'{value}', (letter[0], letter[1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.25, 0, 1)
    out = cv2.resize(out, (out.shape[1] * 2, out.shape[0] * 2))
    cv2.imshow('out', out)
    cv2.waitKey(0)
