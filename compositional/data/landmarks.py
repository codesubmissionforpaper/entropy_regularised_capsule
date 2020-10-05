from imutils import face_utils
import cv2
import dlib

p = "../shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
print(dlib.__version__)

image = cv2.imread('MorphProcessed/353156_00F16.JPG')
# Converting the image to gray scale
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#cv2.imshow("",image)
#k = cv2.waitKey(30000)
# Get faces into webcam's image
rect = detector(image, 1)
# Make the prediction and transfom it to numpy array
shape = predictor(image, rect[0])
shape = face_utils.shape_to_np(shape)
(x, y, w, h) = face_utils.rect_to_bb(rect)
cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
for (x, y) in shape:
    cv2.circle(image, (x, y), 1, (0, 255, 0), 3)
