# import the necessary packages
from imutils import face_utils
import os
import random
import shutil
import numpy as np
import imutils
import dlib
import cv2

def get_random_face_part(image, shape):
    fa1, fa2 = random.sample(list(set(face_utils.FACIAL_LANDMARKS_IDXS.items())-set([('jaw',(0, 17)),('inner_mouth', (60, 68))])),2)
    fa3, fa4 = random.sample(list(set(face_utils.FACIAL_LANDMARKS_IDXS.items())-set([('jaw',(0, 17)),('inner_mouth', (60, 68))])),2)
    while fa3 in list([fa1,fa2]) or fa4 in list([fa1,fa2]):
          fa3, fa4 = random.sample(list(set(face_utils.FACIAL_LANDMARKS_IDXS.items())-set([('jaw', (0, 17)),('inner_mouth', (60, 68))])),2)
    fa5, fa6 = random.sample(list(set(face_utils.FACIAL_LANDMARKS_IDXS.items())-set([('jaw', (0, 17)),('inner_mouth', (60, 68))])),2)
    while fa5 in list([fa1,fa2,fa3,fa4]) or fa6 in list([fa1,fa2,fa3,fa4]):
          fa5, fa6 = random.sample(list(set(face_utils.FACIAL_LANDMARKS_IDXS.items())-set([('jaw', (0, 17)),('inner_mouth', (60, 68))])),2)
    i, j = face_utils.FACIAL_LANDMARKS_IDXS[fa1[0]]
    (x, y, w, h) = cv2.boundingRect(shape[i:j])
    face_part1 = (x,y,w,h)
    i, j = face_utils.FACIAL_LANDMARKS_IDXS[fa2[0]]
    (x, y, w, h) = cv2.boundingRect(shape[i:j])
    face_part2 = (x,y,w,h)             
    i, j = face_utils.FACIAL_LANDMARKS_IDXS[fa3[0]]
    (x, y, w, h) = cv2.boundingRect(shape[i:j])
    face_part3 = (x,y,w,h)
    i, j = face_utils.FACIAL_LANDMARKS_IDXS[fa4[0]]
    (x, y, w, h) = cv2.boundingRect(shape[i:j])
    face_part4 = (x,y,w,h)
    i, j = face_utils.FACIAL_LANDMARKS_IDXS[fa5[0]]
    (x, y, w, h) = cv2.boundingRect(shape[i:j])
    face_part5 = (x,y,w,h)
    i, j = face_utils.FACIAL_LANDMARKS_IDXS[fa6[0]]
    (x, y, w, h) = cv2.boundingRect(shape[i:j])
    face_part6 = (x,y,w,h)
    return face_part1, face_part2, face_part3, face_part4, face_part5, face_part6      

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

        # chose two face parts randomly and return their rectangles
        face_part1, face_part2, face_part3, face_part4, face_part5, face_part6 = get_random_face_part(image, shape)
        (x,y,w,h) = face_part1
        roi1 = image[y:y + h, x:x + w]
        (x,y,w,h) = face_part2
        roi2 = image[y:y + h, x:x + w]
        (x,y,w,h) = face_part3
        roi3 = image[y:y + h, x:x + w]
        (x,y,w,h) = face_part4
        roi4 = image[y:y + h, x:x + w]
        (x,y,w,h) = face_part5
        roi5 = image[y:y + h, x:x + w]
        (x,y,w,h) = face_part6
        roi6 = image[y:y + h, x:x + w]

        roi1_shape = roi1.shape
        roi2_shape = roi2.shape
        roi3_shape = roi3.shape
        roi4_shape = roi4.shape
        roi5_shape = roi5.shape
        roi6_shape = roi6.shape
        if roi1_shape[0]*roi1_shape[1] <=0 or roi2_shape[1]*roi2_shape[0]<=0 or roi3_shape[0]*roi3_shape[1]<=0 or roi4_shape[0]*roi4_shape[1]<=0 or roi5_shape[0]*roi5_shape[1]<=0 or roi6_shape[0]*roi6_shape[1]<=0:
           continue        
        roi2 = cv2.resize(roi2,(roi1_shape[1],roi1_shape[0]))
        roi1 = cv2.resize(roi1,(roi2_shape[1],roi2_shape[0]))
        roi3 = cv2.resize(roi3,(roi4_shape[1],roi4_shape[0]))
        roi4 = cv2.resize(roi4,(roi3_shape[1],roi3_shape[0]))
        roi5 = cv2.resize(roi5,(roi6_shape[1],roi6_shape[0]))
        roi6 = cv2.resize(roi6,(roi5_shape[1],roi5_shape[0]))
        (x,y,w,h) = face_part1
        image[y:y + h, x:x + w] = roi2
        (x,y,w,h) = face_part2
        image[y:y + h, x:x + w] = roi1
        (x,y,w,h) = face_part4
        image[y:y + h, x:x + w] = roi3
        (x,y,w,h) = face_part3
        image[y:y + h, x:x + w] = roi4
        (x,y,w,h) = face_part6
        image[y:y + h, x:x + w] = roi5
        (x,y,w,h) = face_part5
        image[y:y + h, x:x + w] = roi6
        cv2.imwrite('test_2/0/'+files,image)
