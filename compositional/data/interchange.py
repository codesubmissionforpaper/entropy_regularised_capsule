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
    fa1, fa2 = random.sample(face_utils.FACIAL_LANDMARKS_IDXS.items(),2)
    while 'eye' in fa1[0] and 'eye' in fa2[0] or 'jaw' in fa1[0] or 'jaw' in fa2[0] or 'inner_mouth' in fa1[0] or 'inner_mouth' in fa2[0]:
          fa1, fa2 = random.sample(face_utils.FACIAL_LANDMARKS_IDXS.items(),2)
    if fa1[0] == 'right_eye' or fa1[0] == 'right_eyebrow':
       i, j = face_utils.FACIAL_LANDMARKS_IDXS['right_eyebrow']
       a, b = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
       (x, y, w, h) = cv2.boundingRect(np.concatenate((np.asarray(shape[i:j]),np.asarray(shape[a:b]))))
       face_part1 = (x,y,w,h)
    else:
        if fa1[0] == 'left_eye' or fa1[0] == 'left_eyebrow':
           i, j = face_utils.FACIAL_LANDMARKS_IDXS['left_eyebrow']
           a, b = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
           (x, y, w, h) = cv2.boundingRect(np.concatenate((np.asarray(shape[i:j]),np.asarray(shape[a:b]))))
           face_part1 = (x,y,w,h)
        else:
             i, j = face_utils.FACIAL_LANDMARKS_IDXS[fa1[0]]
             (x, y, w, h) = cv2.boundingRect(shape[i:j])
             face_part1 = (x,y,w,h)             
    if fa2[0] == 'right_eye' or fa2[0] == 'right_eyebrow':
       i, j = face_utils.FACIAL_LANDMARKS_IDXS['right_eyebrow']
       a, b = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
       (x, y, w, h) = cv2.boundingRect(np.concatenate((np.asarray(shape[i:j]),np.asarray(shape[a:b]))))
       face_part2 = (x,y,w,h)
    else:
         if fa2[0] == 'left_eye' or fa2[0] == 'left_eyebrow':
            i, j = face_utils.FACIAL_LANDMARKS_IDXS['left_eyebrow']
            a, b = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
            (x, y, w, h) = cv2.boundingRect(np.concatenate((np.asarray(shape[i:j]),np.asarray(shape[a:b]))))
            face_part2 = (x,y,w,h)
         else:
              i, j = face_utils.FACIAL_LANDMARKS_IDXS[fa2[0]]
              (x, y, w, h) = cv2.boundingRect(shape[i:j])
              face_part2 = (x,y,w,h)
    return face_part1, face_part2      

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
        face_part1, face_part2 = get_random_face_part(image, shape)
        (x,y,w,h) = face_part1
        roi1 = image[y:y + h, x:x + w]
        (x,y,w,h) = face_part2
        roi2 = image[y:y + h, x:x + w]
        

        roi1_shape = roi1.shape
        roi2_shape = roi2.shape
        if roi1_shape[0]*roi1_shape[1] <=0 or roi2_shape[1]*roi2_shape[0]<=0:
           continue        
        roi2 = cv2.resize(roi2,(roi1_shape[1],roi1_shape[0]))
        roi1 = cv2.resize(roi1,(roi2_shape[1],roi2_shape[0]))
        (x,y,w,h) = face_part1
        image[y:y + h, x:x + w] = roi2
        (x,y,w,h) = face_part2
        image[y:y + h, x:x + w] = roi1
        cv2.imwrite('test_1/0/'+files,image)
