import cv2

import image_helper
from config import *
from model import Model

if __name__ == '__main__':
    letters = image_helper.letters_extract(TEST_FILE)
    out = image_helper.contour_letters(TEST_FILE)
    model = Model()
    model.load_model(MODEL_FILE)
    for letter in letters:
        print(letter)
        prediction = model.predict_y(letter[2])
        value = TARGET_ARRAY[prediction]
        cv2.putText(out, f'{value}', (letter[0], letter[1] + 200), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 1)
    out = cv2.resize(out, (out.shape[1], out.shape[0]))
    cv2.imshow('out', out)
    cv2.waitKey(0)
