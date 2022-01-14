import cv2
import matplotlib.pyplot as plt

import image_helper
from config import *
from model import Model

if __name__ == '__main__':
    for file in TEST_FILES:
        letters = image_helper.letters_extract(file)
        out = image_helper.contour_letters(file)
        model = Model()
        model.load_model(MODEL_FILE)
        for letter in letters:
            prediction = model.predict_y(letter[2])
            value = TARGET_ARRAY[prediction].replace('М', 'M')
            cv2.putText(out, f'{value}', (letter[0], letter[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
        out = cv2.resize(out, (out.shape[1], out.shape[0]))
        plt.figure()
        plt.imshow(out)
        plt.show()
