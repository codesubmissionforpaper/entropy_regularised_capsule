# import the necessary packages
from imutils import face_utils
import os
import random
import shutil
import numpy as np
import imutils
import dlib
import cv2

def get_face_part(image,shape,part):
    if part == 'right_eye' or part == 'right_eyebrow':
       i, j = face_utils.FACIAL_LANDMARKS_IDXS['right_eyebrow']
       a, b = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
       (x, y, w, h) = cv2.boundingRect(np.concatenate((np.asarray(shape[i:j]),np.asarray(shape[a:b]))))
       face_part = (x,y,w,h)
    else:
        if part == 'left_eye' or part == 'left_eyebrow':
           i, j = face_utils.FACIAL_LANDMARKS_IDXS['left_eyebrow']
           a, b = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
           (x, y, w, h) = cv2.boundingRect(np.concatenate((np.asarray(shape[i:j]),np.asarray(shape[a:b]))))
           face_part = (x,y,w,h)
        else:
             i, j = face_utils.FACIAL_LANDMARKS_IDXS[part]
             (x, y, w, h) = cv2.boundingRect(shape[i:j])
             face_part = (x,y,w,h)
    return face_part

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
file_list = os.listdir('test/0/')
for files in file_list:
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread('test/0/'+files)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # choose each face part and copy to a different image
        for part in list(set(face_utils.FACIAL_LANDMARKS_IDXS.items())-set([('jaw',(0, 17)),('inner_mouth', (60, 68))])):
            face_part = get_face_part(image, shape, part[0])
            if 'eye_brow' in part[0]:
               continue 
            (x,y,w,h) = face_part
            if 'eye' in part[0]:
                scale = 1.5
            if part[0] == 'nose':
               scale = 1
            if part[0] == 'mouth':
               scale = 2
            roi = image[y:y + int(scale*h), x:x + int(scale*w)]
            blank_image = np.zeros((image.shape[0],image.shape[1],3),np.uint8)
            blank_image[y:y + int(scale*h), x:x + int(scale*w)] = roi
            cv2.imwrite('test_3/'+part[0]+'/'+files,blank_image)
