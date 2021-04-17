import numpy as np
import cv2

# From the docs: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html
def face_and_eye_detection(img):
    cv2.putText(img,'Face & Eye Detection',(10,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

    face_cascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('xml/haarcascade_eye.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    return img

### Setup Camera Calibration Data ###
# Termination criteria specifies the max # of iterations and the desired accuracy at which the alg stops
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(4,6,0)
# checkerboard of size (6 x 9) is used
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

num_images_calibrated = 0
gray = None

# Adapted from the docs: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
def calibrate_camera(img, take_image):
    global num_images_calibrated

    # Convert our image to grayscale
    global gray
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    success, corners = cv2.findChessboardCorners(gray, (6,9), None)

    # If found, add object points, image points (after refining them)
    if success:

        # Increase the found corners' accuracy with cornerSubPix to fid more exact corner positions
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (6,9), corners2, success)

        if take_image:
            objpoints.append(objp)
            imgpoints.append(corners2)
            num_images_calibrated += 1
    
    cv2.putText(img,'Num Images = ' + str(num_images_calibrated),(10,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    
    # We still return the image even when we're unable to find the chessboard corners
    return img

# Camera matrix, distortion coefficients, rotation vectors, translation vectors
# Necessary for projecting 3D points w.r.t. the world reference plane onto the image plane
# For more info, see Pinhole Camera Model: https://en.wikipedia.org/wiki/Pinhole_camera_model
mtx, dist, rvecs, tvecs = (np.array([]),)*4
def calibrate_saved_images():
    if num_images_calibrated > 0:
        global mtx, dist, rvecs, tvecs
        success, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        # Print the re-projection error (RMS), usually b/t 0.1 and 1.0 pixels in a good calibration
        print(success)

# Adapted from the docs: https://docs.opencv.org/master/d5/dae/tutorial_aruco_detection.html
def detect_aruco_markers(img):
    if mtx.size == 0:
        raise Exception("calibrate saved images before detecting aruco markers")
    
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters_create()

    (corners, ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict, parameters=parameters)

    # check if the ids list is not empty
    if np.all(ids != None):

        # Estimate pose of each marker and return the values
        rvec, tvec ,_ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)

        for i in range(0, ids.size):
            cv2.aruco.drawAxis(img, mtx, dist, rvec[i], tvec[i], 0.1)

        # Draw a square around the markers
        cv2.aruco.drawDetectedMarkers(img, corners)

        # Show ids of the marker found
        str_ids = ''
        for i in range(0, ids.size):
            str_ids += str(ids[i][0])+', '

        cv2.putText(img, "Id: " + str_ids, (0,64), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)
    else:
        # Show 'No Ids' when no markers are found
        cv2.putText(img, "No Ids", (0,64), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)

    return img