import sys
import os
import numpy as np
import cv2
from scipy.optimize import least_squares
import glob

from functools import partial

# MODE = "fisheye"

# SQUARE_SIZE = 0.03 #meters
# OUTFILE = "camera_calibrations"
# BOARD_HEIGHT = 5
# BOARD_WIDTH = 7

# # Termination criteria
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

# # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((1,BOARD_HEIGHT*BOARD_WIDTH,3), np.float32)
# objp[0,:,:2]  = np.mgrid[0:BOARD_WIDTH,0:BOARD_HEIGHT].T.reshape(-1,2)*0.03

# # Arrays to store object points and image points from all the images.
# objpoints = [] # 3d point in real world space
# imgpoints = [] # 2d points in image plane.
# img_shape = None

# images = glob.glob('uw_fisheye_right_imgs/*.png')

# # Get image points
# for fname in images:
#     img = cv2.imread(fname)
#     img_shape = img.shape[:2]
#     h, w = img.shape[:2]
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#     # Find the chess board corners
#     ret, corners = cv2.findChessboardCorners(gray, (BOARD_WIDTH,BOARD_HEIGHT),None)

#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         objpoints.append(objp)

#         cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
#         imgpoints.append(corners)

#         # Draw and display the corners
#         img = cv2.drawChessboardCorners(img, (BOARD_WIDTH,BOARD_HEIGHT), corners,ret)
#         cv2.imshow('img',img)
#         cv2.waitKey(500)

# cv2.destroyAllWindows()

# # Load in-air calibrations
# K, D = np.load("camera_calibrations.npz").values()

# _K = np.zeros((3, 3))
# _D = np.zeros((4, 1))
# # Get guesstimate of rvecs and tvecs (TODO: need a better initial conditions)
# _, _, _, rvecs, tvecs = cv2.fisheye.calibrate(objpoints, imgpoints, gray.shape[::-1],_K,_D,flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC)
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
    scale = 1 #1/1.33 # percent of original size
    dim = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    img = cv2.resize(img, dim)
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

if MODE == "fisheye":
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    DIM = img_shape

    ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(objpoints, imgpoints, gray.shape[::-1],K,D,flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC)
    # print("Found " + str(N_OK) + " valid images for calibration")
    # print("DIM = " + str(img_shape[::-1]))
    # print("K=np.array(" + str(K.tolist()) + ")")
    # print("D=np.array(" + str(D.tolist()) + ")")
    # print("rotation vectors: " + str(rvecs))
    # print(len(rvecs)) #38. one array for each img.
    # print("translation vectors: " + str(tvecs))
    # print(len(tvecs))
    # print("objpoints: ", objpoints)
    # print(len(objpoints))
    # for arr in objpoints:
    #     print(len(arr))
    # print("imgpoints: ", imgpoints)
    # print(len(imgpoints))
    # for arr in imgpoints:
    #     print(len(arr))

newObjPoints = []
newImgPoints = []
newRvecs = []
newTVecs = []

for i in range(len(rvecs)):
    rvec = rvecs[i]
    tvec = tvecs[i]
    for j in range(len(objpoints)):
        obj = np.squeeze(objpoints[j])
        img = np.squeeze(imgpoints[j])
        print(obj.shape)
        print(img.shape)
        for k in range(len(objpoints[j])):
            o = obj[k]
            i = img[k]
            print(o.shape)
            print(i.shape)
        # print("obj: ", obj)
        # print("img: ", img)
            newRvecs.append(rvec)
            newTVecs.append(tvec)
            newObjPoints.append(o)
            print(o)
            newImgPoints.append(i)
            print(i)

K, D = np.load("camera_calibrations.npz").values()

# Compute the vector of residuals
def uw_ray_trace_residuals(objPoints, imgPoints, K, D, rvecs, tvecs, params):
    """
    Function to compute the residuals for img coordinates
    """
    # Unpack and gather parameters
    dh, th, nx, ny, nz, ra, rh, rw = params

    n = np.array([nx, ny, nz]).T
    n = n / np.linalg.norm(n)
    
    # Undistort points
    print("imgPoints: ", imgPoints)
    undistortedPoints = cv2.fisheye.undistortPoints(imgPoints, K, D)

    # Prepare residual output
    num_points = len(undistortedPoints)
    res = np.zeros((num_points,1))

    # TODO: vectorize all this, write most things in linear form
    for i, undistortedPoint in enumerate(undistortedPoints):

        # Project initial ray
        fx = K[0,0]
        fy = K[1,1]
        f = np.mean((fx,fy)) # this is wrong, but what is correct? m/px
        #xra = np.append(undistorted,np.ones((num_points,1)) * f,1)
        xra = np.append(undistortedPoint, f, 0)
        xra_n = xra/np.linalg.norm(xra)
        
        # Compute angles due to Snell's law
        theta_a = np.acos(np.dot(n,xra_n))
        theta_h = np.asin((ra/rh) * np.sin(theta_a))
        theta_w = np.asin((rh/rw) * np.sin(theta_h))

        # Up to housing barrier
        lambda_h = dh/np.cos(theta_a)
        xri = lambda_h * xra_n

        # Up to water barrier
        lambda_w = (dh+th-np.dot(n,xri)) / np.cos(theta_h)
        xrh_axis = np.cross(n,xra_n) / sin(theta_a)
        xrh_axis = xrh_axis / np.linalg.norm(xrh_axis)
        xrh = rot_axis_angle(xra_n, xrh_axis, theta_h - theta_a)
        xrh_n = xrh / np.linalg.norm(xrh)
        xro = xri + lambda_w * xrh_n

        # Through water
        xrw_axis = np.cross(n,xrh_n) / np.sin(theta_h)
        xrw_axis = xrw_axis / np.linalg.norm(xrw_axis)
        xrw = rot_axis_angle(xrh_n, xrw_axis, theta_w - theta_h)
        xrw_n = xrw / np.linalg.norm(xrw)
        
        # Compute xro', backwards from object point
        # Transform into camera frame
        objPoint = objPoints[i]
        rvec = rvecs[i]
        M_ex = np.zeros((3,4))
        R = cv2.Rodrigues(rvecs)
        M_ex[:,:3] = R
        M_ex[:,3] = tvecs[i]
        P_c = np.matmul(M_ex, objPoint)

        xro_p = P_c + np.dot(((th+dh)*n - P_c), n)/(np.dot(-xrw_n,n))
        
        res[i] = xro - xro_p

def aa_to_quat(axis, theta):
    return np.concatenate(([np.cos(theta/2.), axis * np.sin(theta/2.)]))

def quat_mul(q1, q2):
    # TODO: use matrix version to speed up
    q1_w = q1[0]
    q1_v = q1[1:]
    q2_w = q2[0]
    q2_v = q2[1:]
    q_w = q1_w * q2_w - np.dot(q1_v,q2_v)
    q_v = q1_w * q2_v + q2_w * q1_v + np.cross(q1_v, q2_v)
    return np.concatenate(([q_w],q_v))
    
def quat_rot(vec, q):
    # verify that this is correct
    vec_q = np.concatenate(([0],vec))
    dst_q = quat_mul(quat_mul(q, vec_q), quat_conj(q))
    dst_v = dst_q[1:]
    return dst_v
    
def quat_conj(q):
    qj = q.copy()
    qj[1:] = -qj[1:]
    return qj
    
def rot_axis_angle(vec, axis, theta):
    """
    Rotate a 1x3 vector around an axis by theta (uses quaternion rotations)
    """
    q = aa_to_quat(axis, theta)
    return quat_rot(vec, q)
    
def uw_calibrate(objPoints, imgPoints, K, D, rvecs, tvecs, flags=0, criteria=None):

    #uw_ray_trace_residuals(objPoints, imgPoints, K, D, rvecs, tvecs, params):
    uw_res = partial(uw_ray_trace_residuals, objPoints, imgPoints, K, D, rvecs, tvecs)

    params0 = [1, 1, 0, 0, 1, 1, 1.33, 1.495]
    
    params = least_squares(uw_res, params0)
    return params

params = uw_calibrate(newObjPoints, newImgPoints, K, D, newRvecs, newTVecs)

# see what our final errors and parameters look like
print(params)
print(np.linalg.norm(uw_ray_trace_residuals(objpoints, imgpoints, K, D, rvecs, tvecs, params)))