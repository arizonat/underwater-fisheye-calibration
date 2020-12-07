# initial camera calibration
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

import sys
import os
import numpy as np
import cv2
import glob

MODE = "fisheye" #"fisheye" or "normal"

IMAGES_LOC = "uw_fisheye_right_imgs/"
IMAGES_TYPE = "*.png"
CALIB_OUTFILE = "camera_calibrations"
UNDISTORT_OUTFOLDER = "output/"

SQUARE_SIZE = 0.03 #meters
BOARD_HEIGHT = 5
BOARD_WIDTH = 7

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((1,BOARD_HEIGHT*BOARD_WIDTH,3), np.float32)
objp[0,:,:2]  = np.mgrid[0:BOARD_WIDTH,0:BOARD_HEIGHT].T.reshape(-1,2)*0.03

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
img_shape = None

# Parse chessboards from images
images = glob.glob(IMAGES_LOC+IMAGES_TYPE)

print("Parsing chessboards")

for fname in images:
    img = cv2.imread(fname)
    img_shape = img.shape[:2]
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (BOARD_WIDTH,BOARD_HEIGHT),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        # img = cv2.drawChessboardCorners(img, (BOARD_WIDTH,BOARD_HEIGHT), corners,ret)
        # cv2.imshow('img',img)
        # cv2.waitKey(500)

cv2.destroyAllWindows()

# Calibrate (find K, D)

print("Calibrating...")

N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
DIM = img_shape

if MODE == "fisheye":
    ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(objpoints, imgpoints, gray.shape[::-1],K,D,flags=calibration_flags)

else:
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None,flags=calibration_flags)
    K_new, roi=cv2.getOptimalNewCameraMatrix(K,D,(w,h),1,(w,h))

print("Found " + str(N_OK) + " valid images for calibration")
print("DIM = " + str(img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")
print("rotation vectors: " + str(rvecs))
print(len(rvecs))
print("translation vectors: " + str(tvecs))
print(len(tvecs))

# Save the calibrations
if MODE == "normal":
    print("refined camera matrix: ", K_new)

if MODE == "fisheye":
    print("saving matrix to ", CALIB_OUTFILE+"_"+MODE+".npz")
    np.savez(CALIB_OUTFILE, K, D)
else:
    print("saving matrices to ", CALIB_OUTFILE+"_"+MODE+".npz")
    np.savez(CALIB_OUTFILE, K, K_new, D)

# Show the undistorted images
print("Saving undistorted images...")
try:
    os.mkdir(UNDISTORT_OUTFOLDER + IMAGES_LOC)
except OSError as oe:
    print("Warning: output folder already exists, may be overwriting images.")

for fname in images:
    img = cv2.imread(fname)
    h, w = img.shape[:2]

    if MODE == "normal":
        dst = cv2.undistort(img, K, D, None, K_new)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        # vis = np.concatenate((img, dst), axis=0)
        
        h1, w1 = img.shape[:2]
        h2, w2 = dst.shape[:2]
        
        #create empty matrix
        vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)
        
        #combine 2 images
        vis[:h1, :w1,:3] = img
        vis[:h2, w1:w1+w2,:3] = dst
        prefix = fname[:-4]
        cv2.imwrite(UNDISTORT_OUTFOLDER + prefix + 'undistorted_normal.png', vis)

    else:
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
        dst = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        h1, w1 = img.shape[:2]
        h2, w2 = dst.shape[:2]
        
        #create empty matrix
        vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)
        
        #combine 2 images
        vis[:h1, :w1,:3] = img
        vis[:h2, w1:w1+w2,:3] = dst
        prefix = fname[:-4]
        cv2.imwrite(UNDISTORT_OUTFOLDER + prefix + 'undistorted_fisheye.png', vis)

# Check the errors
mean_error = 0
for i in range(len(objpoints)):
    if MODE == "fisheye":
        imgpoints2, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        imgpoints2 = imgpoints2.transpose(1,0,2)

    elif MODE == "normal":
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )
