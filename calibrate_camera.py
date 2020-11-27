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
BOARD_HEIGHT = 5
BOARD_WIDTH = 7

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((1,BOARD_HEIGHT*BOARD_WIDTH,3), np.float32)
objp[0,:,:2]  = np.mgrid[0:BOARD_WIDTH,0:BOARD_HEIGHT].T.reshape(-1,2)*0.03

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
img_shape = None

images = glob.glob('uw_fisheye_right_imgs/*.png')

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
        img = cv2.drawChessboardCorners(img, (BOARD_WIDTH,BOARD_HEIGHT), corners,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

if MODE == "fisheye":
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    DIM = img_shape

    ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(objpoints, imgpoints, gray.shape[::-1],K,D,flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC)
    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM = " + str(img_shape[::-1]))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")
    print("rotation vectors: " + str(rvecs))
    print(len(rvecs))
    print("translation vectors: " + str(tvecs))
    print(len(tvecs))

# UNDISTORT
    def undistort(img_path):
        img = cv2.imread(img_path)
        h,w = img.shape[:2]
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        prefix = fname[:-4]
        cv2.imshow("undistorted" + prefix, undistorted_img)
        cv2.imwrite(prefix + 'calibresult.png', undistorted_img)
    
    for fname in images:
        undistort(fname)
    #     img = cv2.imread(fname)
    #     h,w = img.shape[:2]
    #     map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    #     undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    #     # cv2.imshow("undistorted", undistorted_img)
    #     prefix = fname[:-4]
    #     cv2.imwrite(prefix + 'calibresult.png', undistorted_img)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # for fname in images:

        # dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort        
        # scaled_K = K # The values of K is to scale with image dimension.
        # scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
        # new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim1, np.eye(3), balance=1)
        # map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim1, cv2.CV_16SC2)
        # undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)   
        # prefix = fname[:-4]
        # cv2.imwrite(prefix + 'calibresult.png', undistorted_img)
else:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    for fname in images:
        img = cv2.imread(fname)
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        # print("new camera matrix", newcameramtx)
        dst = cv2.fisheye.undistort(img, mtx, dist, None, newcameramtx)
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
        cv2.imwrite(prefix + 'calibresult.png', vis)

# print("camera matrix", mtx)
# print("distortion coefficients", dist)



if MODE == "normal":
    print("refined camera matrix: ", newcameramtx)

if MODE == "fisheye":
    print("saving matrix to ", OUTFILE+"_"+MODE+".npz")
    np.savez(OUTFILE, mtx, dist)
else:
    print("saving matrices to ", OUTFILE+"_"+MODE+".npz")
    np.savez(OUTFILE, mtx, newcameramtx, dist)

mean_error = 0
for i in range(len(objpoints)):
    if MODE == "fisheye":
        imgpoints2, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        imgpoints2 = imgpoints2.transpose(1,0,2)

    elif MODE == "normal":
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )
