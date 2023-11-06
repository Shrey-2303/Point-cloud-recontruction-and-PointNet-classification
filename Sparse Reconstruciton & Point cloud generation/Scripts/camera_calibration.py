import cv2
import os
import sys
import numpy as np

def calculate_projection(pts2d, pts3d):
    """
    Compute a 3x4 projection matrix M using a set of 2D-3D point correspondences.

    Given a set of N 2D image points (pts2d) and their corresponding 3D world coordinates
    (pts3d), this function calculates the projection matrix M using the Direct Linear
    Transform (DLT) method. The projection matrix M relates the 3D world coordinates to
    their 2D image projections in homogeneous coordinates.

    Parameters:
    pts2d (numpy.ndarray): An Nx2 array containing the 2D image points.
    pts3d (numpy.ndarray): An Nx3 array containing the corresponding 3D world coordinates.

    Returns:
    M (numpy.ndarray): A 3x4 projection matrix M that relates 3D world coordinates to 2D
                   image points in homogeneous coordinates.
    """

    M = None
    num_points = pts2d.shape[0]
    A = np.zeros((num_points * 2, 12))
    for i in range(A.shape[0]):
        if i % 2 == 0:
            row = np.array([pts3d[i//2, 0], pts3d[i//2, 1], pts3d[i//2, 2], 1, 0, 0, 0, 0,
                            -pts3d[i//2, 0] * pts2d[i//2, 0],
                            -pts3d[i//2, 1] * pts2d[i//2, 0],
                            -pts3d[i//2, 2] * pts2d[i//2, 0],
                            -pts2d[i//2, 0]])
        else:
            row = np.array([0, 0, 0, 0, pts3d[i//2, 0], pts3d[i//2, 1], pts3d[i//2, 2], 1,
                            -pts3d[i//2, 0] * pts2d[i//2, 1],
                            -pts3d[i//2, 1] * pts2d[i//2, 1],
                            -pts3d[i//2, 2] * pts2d[i//2, 1],
                            -pts2d[i//2, 1]])

        A[i, :] = row

    U, S, V = np.linalg.svd(A)
    M = V[-1,:]
    M = M.reshape((3, 4), order='F')

    ####################################
    ##########YOUR CODE HERE############
    ####################################

    ####################################
    return M


def calculate_reprojection_error(pts2d,pts3d):
    """
    Calculate the reprojection error for a set of 2D-3D point correspondences.

    Given a set of N 2D image points (pts2d) and their corresponding 3D world coordinates
    (pts3d), this function calculates the reprojection error. The reprojection error is a
    measure of how accurately the 3D points project onto the 2D image plane when using a
    projection matrix.

    Parameters:
    pts2d (numpy.ndarray): An Nx2 array containing the 2D image points.
    pts3d (numpy.ndarray): An Nx3 array containing the corresponding 3D world coordinates.

    Returns:
    float: The reprojection error, which quantifies the accuracy of the 3D points'
           projection onto the 2D image plane.
    """
    M = calculate_projection(pts2d, pts3d)
    error = None
    

    ####################################
    ##########YOUR CODE HERE############
    ####################################
    
    ####################################
 
    num_points = len(pts3d)
    if num_points != len(pts2d):
        raise ValueError("Number of 3D and 2D points must match.")

    total_error = 0.0

    for i in range(num_points):
        X_i = np.append(pts3d[i], 1)
        projected_2d = M @ X_i  
        projected_2d /= projected_2d[2]  

        error = np.linalg.norm(projected_2d[:2] - pts2d[i], 2) 
        total_error += error

    reproj_error = (1 / num_points) * total_error
    print(M,"\n\n\n\n")
    print(reproj_error)
    return reproj_error

if __name__ == '__main__':
    data = np.load("data/camera_calib_data.npz")
    pts2d = data['pts2d']
    pts3d = data['pts3d']
    print(pts2d)
    print("\n\n",pts3d)
    P = calculate_projection(pts2d,pts3d)
    print(P)
    reprojection_error = calculate_reprojection_error(pts2d, pts3d)

    print(f"Projection matrix: {P}")    
    print()
    print(f"Reprojection Error: {reprojection_error}")