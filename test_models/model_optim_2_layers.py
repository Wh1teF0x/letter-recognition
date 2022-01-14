import os
import pickle
import warnings

import cv2
import numpy as np

import image_helper
from config import *


# https://habr.com/ru/post/271563/


class Model:
    syn0 = None
    syn1 = None
    syn2 = None
    data = None
    labels = None

    def __init__(self):
        np.random.seed(1)
        self.syn0 = 2 * np.random.random((INPUT_SIZE, HIDDEN_SIZE)) - 1
        self.syn1 = 2 * np.random.random((HIDDEN_SIZE, HIDDEN_SIZE_2)) - 1
        self.syn2 = 2 * np.random.random((HIDDEN_SIZE_2, OUTPUT_SIZE)) - 1

    def f(self, x, deriv=False):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if deriv:
                return self.f(x) * (1 - self.f(x))
            return 1 / (1 + np.exp(-x))

    def save_model(self):
        with open(MODEL_FILE, 'wb') as file:
            model = {'syn0': self.syn0, 'syn1': self.syn1, 'syn2': self.syn2}
            pickle.dump(model, file)

    def load_model(self, path_to_model=MODEL_FILE):
        with open(path_to_model, 'rb') as file:
            model = pickle.load(file)
            self.syn0 = model['syn0']
            self.syn1 = model['syn1']
            self.syn2 = model['syn2']

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
            input = X
            l1 = self.f(np.dot(input, self.syn0))
            l2 = self.f(np.dot(l1, self.syn1))
            out = self.f(np.dot(l2, self.syn2))
            out_error = y - out
            if (i % (ITER_COUNT // 10)) == 0:
                print(f'Iteration {i} out of {ITER_COUNT}')
                print(f'Error: {str(np.mean(np.abs(out_error)))}')
            out_delta = out_error * self.f(out, deriv=True)
            l2_error = out_delta.dot(self.syn2.T)
            l2_delta = l2_error * self.f(l2, deriv=True)
            l1_error = l2_delta.dot(self.syn1.T)
            l1_delta = l1_error * self.f(l1, deriv=True)
            self.syn0 += input.T.dot(l1_delta)
            self.syn1 += l1.T.dot(l2_delta)
            self.syn2 += l2.T.dot(out_delta)
        print('Finish')

    def predict_y(self, X, prepare=True):
        if prepare:
            X = cv2.bitwise_not(X)
            X = X / 255
            X = X.flatten()
        input = X
        l1 = self.f(np.dot(input, self.syn0))
        l2 = self.f(np.dot(l1, self.syn1))
        out = self.f(np.dot(l2, self.syn2))
        return np.argmax(out), np.max(out)


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
        x, y, letter = letter
        while np.max(letter[0]) == np.min(letter[0]):
            letter = np.delete(letter, 0, 0)
        letter = np.insert(letter, 0, 0, axis=0)
        letter = np.insert(letter, letter.shape[0], 0, axis=0)
        letter = cv2.resize(letter, SIZE, interpolation=cv2.INTER_AREA)
        prediction = model.predict_y(letter.flatten())
        print(prediction)
        value = TARGET_ARRAY[prediction[0]]
        cv2.putText(out, f'{value}', (x, y + 170), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 1)
    out = cv2.resize(out, (out.shape[1], out.shape[0]))
    cv2.imshow('out', out)
    cv2.waitKey(0)
