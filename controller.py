import cv2
import keras
import numpy as np
import math
import os
import pyautogui

def region_of_interest(frame):
    return frame[200:400, 200:400]

def mask_extractor(region_of_interest):

    hsv = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2HSV)
    #define range of skin color in HSV
    lower_skin = np.array([0,20,70], dtype=np.uint8)
    upper_skin = np.array([20,255,255], dtype=np.uint8)

    #extract skin colur imagw
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    #extrapolate the hand to fill dark spots within
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = 4)

    #blur the image
    mask = cv2.GaussianBlur(mask,(5,5),100)

    return mask

# CNN
def scale_to_range(image):
    return image/255

def matrix_to_vector(image):
    return image.flatten()

def prepare_for_predict(mask):
    scale = scale_to_range(mask)
    vector = matrix_to_vector(scale)
    return np.reshape(vector, (1, 40000))

def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]


#State machine
def convert_to_key(last_key, current_key):
    print(current_key)
    dic = {
        'q': "left",
        'e': "right"
    }
    is_last_key_none = last_key == None
    is_current_key_none = current_key == None

    if (is_last_key_none and is_current_key_none):
        return

    last_key_touple = dic.get(last_key) != None

    if(is_current_key_none):
        if not is_last_key_none:
            if last_key_touple:
                pyautogui.keyUp(dic.get(last_key)[0])
                pyautogui.keyUp("ctrl")
            else:
                pyautogui.keyUp(last_key)
        return

    if current_key == last_key:
        return

    current_key_touple = dic.get(current_key) != None

    if not current_key_touple:
        if last_key_touple: pyautogui.keyUp("ctrl")
        pyautogui.keyDown(current_key)
    else:
        if not last_key_touple: pyautogui.keyDown("ctrl")
        pyautogui.keyDown(dic.get(current_key))


if __name__ == '__main__':
    last_key = None
    alphabet = ["space", "left", None, "right", "ctrl", "q", "e"]

    model = keras.models.load_model('0.9980ac.h5')

    cap = cv2.VideoCapture(0)
    while (1):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (200, 200), (400, 400), (0, 255, 0), 0)

        roi = region_of_interest(frame)
        mask = mask_extractor(roi)

        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)

        cnn_input = prepare_for_predict(mask)
        result = model.predict(np.array(cnn_input[0:1], np.float32))
        win = winner(result[0])

        current_key = alphabet[win]
        convert_to_key(last_key, current_key)
        last_key = current_key

        key = cv2.waitKey(250)
        if key == ord('0'):
            break

    cv2.destroyAllWindows()
    cap.release()