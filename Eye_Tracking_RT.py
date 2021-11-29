import cv2 as cv
import numpy as np
import dlib

"""Convert shapes to numpy array"""
def shape_to_np(shape,dtype='int'):
    # initialize x-y coordinates
    coords = np.zeros((68,2),dtype=dtype)
    # loop 68 facial markers and convert to 2 tuple x-y coords
    for i in range (0,68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    
    return coords

"""Create eye mask"""
def eye_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv.fillConvexPoly(mask, points, 255)

    return mask

"""Contour detection"""
def find_contours(thresh, mid, img, right=False):
    # Find contours
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Select largest contour area and find centroid
    try: 
        c = max(contours, key=cv.contourArea)
        # image moments to find centroid
        moment = cv.moments(c)
        cx = int(moment['m10']/['m00'])
        cy = int(moment['m01']/['m00'])

        if right:
            cx += mid
        # draw circle at contour centroid location
        cv.circle(img, (cx,cy),4,(0,0,255),2)
    except:
        pass

"""Initialize face detector"""
detector = dlib.get_frontal_face_detector()
# load dlib shape predictor model 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

left_eye = [36,37,38,39,40,41]
right_eye = [42,43,44,45,46,47]

cap = cv.VideoCapture(0)
# while True:
#     success, img = cap.read()
#     if img is None:
#         break
#     thresh = img.copy()
ret, img = cap.read()
thresh = img.copy()

cv.namedWindow('image')
kernel = np.ones((9,9), np.uint8)

# pass if nothing there
def no_data(x):
    pass

cv.createTrackbar('threshold','image',0,255,no_data)

"""Main loop"""
while(True):
    ret, img = cap.read()
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    rects = detector(gray,1)

    for r in rects:
        shape = predictor(gray,r)
        shape = shape_to_np(shape)

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = eye_mask(mask,left_eye)
        mask = eye_mask(mask, right_eye)
        mask = cv.dilate(mask,kernel,5)

        eyes = cv.bitwise_and(img,img, mask=mask)
        mask = (eyes == [0,0,0]).all(axis=2)
        eyes[mask] = [255,255,255]

        mid = (shape[42][0] + shape[39][0])//2
        eyes_gray = cv.cvtColor(eyes,cv.COLOR_BGR2GRAY)
        
        threshold = cv.getTrackbarPos('threshold', 'image')
        _, thresh = cv.threshold(eyes_gray,threshold, 255,cv.THRESH_BINARY)
        thresh = cv.erode(thresh, None, iterations=2)
        thresh = cv.dilate(thresh, None, iterations=4)
        thresh = cv.medianBlur(thresh,3)
        thresh = cv.bitwise_not(thresh)

        find_contours(thresh[:,0:mid], mid, img)
        find_contours(thresh[:mid:],mid, img, True)
        
    # show images with detected features and markers
    cv.imshow('eyes',img)
    cv.imshow('image',thresh)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

