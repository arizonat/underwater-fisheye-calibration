import sys
import os
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import glob
import math

from functools import partial

MODE = "fisheye"

SQUARE_SIZE = 0.03 #meters
OUTFILE = "camera_calibrations"
BOARD_HEIGHT = 5
BOARD_WIDTH = 7

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((1,BOARD_HEIGHT*BOARD_WIDTH,3), np.float32)
objp[0,:,:2]  = np.mgrid[0:BOARD_WIDTH,0:BOARD_HEIGHT].T.reshape(-1,2)*SQUARE_SIZE

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
img_shape = None

images = glob.glob('uw_fisheye_right_imgs/*.png')

print("Loading chessboard points")

# Get image points
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
        #cv2.imshow('img',img)
        #cv2.waitKey(500)

cv2.destroyAllWindows()

print("Computing rvecs and tvecs")

# Load in-air calibrations
K, D = np.load("camera_calibrations.npz").values()

ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(objpoints, imgpoints, gray.shape[::-1],K,D,flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC)

# TODO: I don't know why I need to concat twice
objPoints = np.concatenate(np.concatenate(objpoints,axis=0), axis=0)
imgPoints = np.concatenate(imgpoints)
imgPoints = np.squeeze(imgPoints,axis=1)

rvecs = np.repeat(rvecs,BOARD_HEIGHT*BOARD_WIDTH,axis=0)
tvecs = np.repeat(tvecs,BOARD_HEIGHT*BOARD_WIDTH,axis=0)

# Just use the in-air calibrations
K, D = np.load("camera_calibrations.npz").values()

resLength = 0

# Compute the vector of residuals
def uw_ray_trace_residuals(objPoints, imgPoints, K, D, rvecs, tvecs, params):
    """
    Function to compute the residuals for img coordinates
    Projects onto the chessboard itself rather than the outside face of the housing
    """
    global resLength

    # Unpack and gather parameters
    nx, ny, nz = 0, 0, 1
    th = 0
    rh = 1.33

    dh, ra, rw = params

    n = np.array([nx, ny, nz]).T
    n = n / np.linalg.norm(n)
    
    # Undistort points
    # TODO: understand these weird dimensions
    undistortedPoints = cv2.fisheye.undistortPoints(np.expand_dims(imgPoints,axis=0), K, D, None, K)
    undistortedPoints = np.squeeze(undistortedPoints)

    # Prepare residual output
    num_points = undistortedPoints.shape[0]
    res = np.zeros((num_points,1))

    # TODO: vectorize all this, write most things in linear form
    for i, undistortedPoint in enumerate(undistortedPoints):
        # Project initial ray
        fx = K[0,0]
        fy = K[1,1]
        f = np.mean((fx,fy)) # this is wrong, but what is correct? m/px
        #xra = np.append(undistorted,np.ones((num_points,1)) * f,1)
        xra = np.append(undistortedPoint, f)
        xra_n = xra/np.linalg.norm(xra)
        
        # Compute angles due to Snell's law
        theta_a = np.arccos(np.dot(n,xra_n))
        theta_h = np.arcsin((ra/rh) * np.sin(theta_a))
        theta_w = np.arcsin((rh/rw) * np.sin(theta_h))

        # Up to housing barrier
        lambda_a = dh/np.cos(theta_a)
        xri = lambda_a * xra_n

        # Up to water barrier
        lambda_h = (dh+th-np.dot(n,xri)) / np.cos(theta_h)
        xrh_axis = np.cross(n,xra_n) / np.sin(theta_a)
        xrh_axis = xrh_axis / np.linalg.norm(xrh_axis)
        xrh = rot_axis_angle(xra_n, xrh_axis, theta_h - theta_a)
        xrh_n = xrh / np.linalg.norm(xrh)
        xro = xri + lambda_h * xrh_n

        # Up to calibration target plane
        xrw_axis = np.cross(n,xrh_n) / np.sin(theta_h)
        xrw_axis = xrw_axis / np.linalg.norm(xrw_axis)
        xrw = rot_axis_angle(xrh_n, xrw_axis, theta_w - theta_h)
        xrw_n = xrw / np.linalg.norm(xrw)

        # Transform calibration target to camera frame
        objPoint = np.insert(objPoints[i],3,1.)
        rvec = rvecs[i].flatten()
        M_ex = np.zeros((3,4))
        R,J = cv2.Rodrigues(rvec)
        M_ex[:,:3] = R
        M_ex[:,3] = tvecs[i].flatten()
        P_c = np.matmul(M_ex, objPoint)

        target_pd_wf = np.array([0.,0.,1.,1.]) # z-direction + homogenous coordinate
        target_pd_cf = np.matmul(M_ex, target_pd_wf)
        target_pd_cf = target_pd_cf / np.linalg.norm(target_pd_cf)

        # Compare p_c with p_c' to get error
        lambda_w, P_c_p = intersect_ray_plane(xro, xrw_n, P_c, target_pd_cf)

        # is this the correct error?
        res[i] = np.linalg.norm(P_c - P_c_p)
    print(params)
    print(np.linalg.norm(res))
    resLength = len(res)
    return np.squeeze(res)

def intersect_ray_plane(r0, rd, p0, pd):
    """
    Returns the distance and point of intersection of a ray and a plane
    r0 - point on ray, rd - (unit) direction of ray, p0 - point on plane, pd - (unit) normal of plane
    """
    rd = rd / np.linalg.norm(rd)
    pd = pd / np.linalg.norm(pd)
    lambda_r = np.dot((p0 - r0), pd) / np.dot(rd, pd)
    return (lambda_r, r0 + lambda_r * rd)
        
def aa_to_quat(axis, theta):
    return np.insert(axis * np.sin(theta/2.), 0, np.cos(theta/2.))

# TODO: w x y z
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
    # print("axis: ", axis)
    # print("theta: ", theta)
    # q = aa_to_quat(axis, theta)
    # res = quat_rot(vec, q)
    # print("result: ", res)

    rotation_vector = theta * axis
    rotation = Rotation.from_rotvec(rotation_vector)
    rotated_vec = rotation.apply(vec)

    return rotated_vec

    
def uw_calibrate(objPoints, imgPoints, K, D, rvecs, tvecs, flags=0, criteria=None):

    #uw_ray_trace_residuals(objPoints, imgPoints, K, D, rvecs, tvecs, params):
    uw_res = partial(uw_ray_trace_residuals, objPoints, imgPoints, K, D, rvecs, tvecs)

    # params0 = [0.001, 0.00635, 0, 0, 1, 1, 1.33, 1.495]
    params0 = [0.001, 1, 1.495]

    # lwrbnd = [0., 0., -1., -1., 0., 0., 0., 0.]
    # uprbnd = [10., 10., 1., 1., 1., 38., 38., 38.]

    lwrbnd = [0., 0., 0.]
    uprbnd = [10., 38., 38.]
    
    #TODO: use mask on least_squares
    params = least_squares(uw_res, params0, bounds=(lwrbnd,uprbnd))
    cost = params.cost 
    cost /= resLength
    cost = math.sqrt(cost)
    print("cost: ", cost)
    return params

params = uw_calibrate(objPoints, imgPoints, K, D, rvecs, tvecs)

# see what our final errors and parameters look like
print(params)
#print(np.linalg.norm(uw_ray_trace_residuals(objpoints, imgpoints, K, D, rvecs, tvecs, params)))
