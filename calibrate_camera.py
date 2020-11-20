# initial camera calibration
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

import sys
import os
import numpy as np
import cv2
import glob

MODE = "fisheye" #"fisheye" or "normal"

SQUARE_SIZE = 0.03 #meters
OUTFILE = "camera_calibrations"
chessBoardHeight = 5
chessBoardWidth = 7

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessBoardHeight*7,3), np.float32)
objp[:,:2] = np.mgrid[0:chessBoardWidth,0:chessBoardHeight].T.reshape(-1,2)*0.03

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('imgs/*.JPG')

for fname in images:
    img = cv2.imread(fname)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (chessBoardWidth,chessBoardHeight),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        # img = cv2.drawChessboardCorners(img, (chessBoardWidth,chessBoardHeight), corners2,ret)
        # cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

if MODE == "fisheye":
    ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(objpoints, imgpoints, gray.shape[::-1],None,None)   
else:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

print("camera matrix: ", mtx)

if MODE == "normal":
    print("refined camera matrix: ", newcameramtx)

print("distortion coefficients: ", dist)

print("saving matrices to ", OUTFILE+"_"+MODE+".npz")
np.savez(OUTFILE, mtx, newcameramtx, dist)

mean_error = 0
for i in range(len(objpoints)):
    if MODE == "fisheye":
        imgpoints2, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    elif MODE == "normal":
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )
