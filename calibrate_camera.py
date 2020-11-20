# initial camera calibration
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

import sys
import os
import numpy as np
import cv2
import glob

SQUARE_SIZE = 0.03 #meters
CHESS_SIZE = (7,5)
OUTFILE = "camera_calibrations.npz"

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((CHESS_SIZE[0]*CHESS_SIZE[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHESS_SIZE[0],0:CHESS_SIZE[1]].T.reshape(-1,2)*SQUARE_SIZE

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('imgs/*.JPG')

for fname in images:
    img = cv2.imread(fname)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHESS_SIZE,None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHESS_SIZE, corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

print("camera matrix: ", mtx)
print("refined camera matrix: ", newcameramtx)
print("distortion coefficients: ", dist)

print("saving matrices to ", OUTFILE)
np.savez(OUTFILE, mtx, newcameramtx, dist)
