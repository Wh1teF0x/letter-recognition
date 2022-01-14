import os
import pickle

import cv2
import numpy as np
import numpy.random as r
from PIL import Image
from sklearn.preprocessing import StandardScaler

from config import *

# https://habr.com/ru/post/271563/
# https://habr.com/ru/post/466565/
# https://proglib.io/p/neural-nets-guide
# https://github.com/adventuresinML/adventures-in-ml-code/blob/master/neural_network_tutorial.py


class Model:
    W = {}
    b = {}
    alpha = 1
    nn_structure = []
    train_data = None
    train_labels = None

    def __init__(self, nn_structure=STRUCTURE, training_alpha=ALPHA):
        self.alpha = training_alpha
        self.W = {}
        self.b = {}
        self.nn_structure = nn_structure

        for i in range(1, len(nn_structure)):
            self.W[i] = r.random_sample((nn_structure[i], nn_structure[i - 1]))
            self.b[i] = r.random_sample((nn_structure[i],))

    def save_model(self):
        with open(MODEL_FILE, 'wb') as file:
            model = {'W': self.W, 'b': self.b, 'nn_structure': self.nn_structure}
            pickle.dump(model, file)

    def load_model(self, path_to_model=MODEL_FILE):
        with open(path_to_model, 'rb') as file:
            model = pickle.load(file)
            self.W = model['W']
            self.b = model['b']
            self.nn_structure = model['nn_structure']

    def f(self, x):
        return 1 / (1 + np.exp(-x))

    def f_deriv(self, x):
        return self.f(x) * (1 - self.f(x))

    def calculate_out_layer_delta(self, y, h_out, z_out):
        # delta^(nl) = -(y_i - h_i^(nl)) * f'(z_i^(nl))
        return -(y - h_out) * self.f_deriv(z_out)

    def calculate_hidden_delta(self, delta_plus_1, w_l, z_l):
        # delta^(l) = (transpose(W^(l)) * delta^(l+1)) * f'(z^(l))
        return np.dot(np.transpose(w_l), delta_plus_1) * self.f_deriv(z_l)

    def feed_forward(self, x):
        h = {1: x}
        z = {}
        for i in range(1, len(self.W) + 1):
            if i == 1:
                node_in = x
            else:
                node_in = h[i]
            # z^(l+1) = W^(l)*h^(l) + b^(l)
            z[i + 1] = self.W[i].dot(node_in) + self.b[i]
            # h^(l) = f(z^(l))
            h[i + 1] = self.f(z[i + 1])
        return h, z

    def init_tri_values(self):
        tri_W = {}
        tri_b = {}
        for l in range(1, len(self.nn_structure)):
            tri_W[l] = np.zeros((self.nn_structure[l], self.nn_structure[l - 1]))
            tri_b[l] = np.zeros((self.nn_structure[l],))
        return tri_W, tri_b

    def train(self, iter_count=ITER_COUNT):
        W = self.W
        b = self.b
        cnt = 0
        m = len(self.train_labels)
        print(f'Starting gradient descent for {iter_count} iterations')
        while cnt < iter_count:
            if cnt % 100 == 0:
                print(f'Iteration {cnt} of {iter_count}')
            tri_W, tri_b = self.init_tri_values()
            for i in range(len(self.train_labels)):
                delta = {}
                h, z = self.feed_forward(self.train_data[i, :])
                for j in range(len(self.nn_structure), 0, -1):
                    if j == len(self.nn_structure):
                        delta[j] = self.calculate_out_layer_delta(self.train_labels[i, :], h[j], z[j])
                    else:
                        if j > 1:
                            delta[j] = self.calculate_hidden_delta(delta[j + 1], W[j], z[j])
                        # triW^(l) = triW^(l) + delta^(l+1) * transpose(h^(l))
                        tri_W[j] += np.dot(delta[j + 1][:, np.newaxis], np.transpose(h[j][:, np.newaxis]))
                        # trib^(l) = trib^(l) + delta^(l+1)
                        tri_b[j] += delta[j + 1]
            for i in range(len(self.nn_structure) - 1, 0, -1):
                W[i] += -self.alpha * (1.0 / m * tri_W[i])
                b[i] += -self.alpha * (1.0 / m * tri_b[i])
            cnt += 1
        self.W = W
        self.b = b

    def load_data(self, folder):
        self.train_data = []
        self.train_labels = []
        scale = StandardScaler()
        letters = os.listdir(folder)
        for letter in letters:
            items = os.listdir(f'{DATASET_FOLDER}/{letter}')
            for item in items:
                data_item = np.array(Image.open(f'{DATASET_FOLDER}/{letter}/{item}'))
                data_target = np.zeros(STRUCTURE[-1])
                data_target[RESULTS_MAP[letter]] = 1
                self.train_data.append(data_item.flatten())
                self.train_labels.append(data_target)
        self.train_data = scale.fit_transform(np.array(self.train_data))
        self.train_labels = np.array(self.train_labels)

    def predict_y(self, X, n_layers=OUTPUT_SIZE, prepare=True):
        if prepare:
            X = cv2.bitwise_not(X)
            X = StandardScaler().fit_transform(X)
            X = X.flatten()
        h, z = self.feed_forward(X)
        return np.argmax(h[n_layers]), h[n_layers]


if __name__ == '__main__':
    model = Model()
    model.load_data(DATASET_FOLDER)
    model.train()
    model.save_model()
