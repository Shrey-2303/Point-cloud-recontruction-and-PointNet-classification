import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import helper_functions as _helper



def compute_fundamental_matrix(pts1, pts2, scale):
    """
    Compute the Fundamental matrix from corresponding 2D points in two images.

    Given two sets of corresponding 2D image points from Image 1 (pts1) and Image 2 (pts2),
    as well as a scaling factor (scale) representing the maximum dimension of the images, 
    this function calculates the Fundamental matrix.

    Parameters:
    pts1 (numpy.ndarray): An Nx2 array containing 2D points from Image 1.
    pts2 (numpy.ndarray): An Nx2 array containing 2D points from Image 2, corresponding to pts1.
    scale (float): The maximum dimension of the images, used for scaling the Fundamental matrix.

    Returns:
    F (numpy.ndarray): A 3x3 Fundamental matrix 
    """
    F = None
    num_points = pts1.shape[0]
    A = np.zeros((num_points, 9))

    pts1 = pts1 / scale
    pts2 = pts2 / scale

    for i in range(num_points):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        A[i] = [x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1]

    U, S, V = np.linalg.svd(A)
    f = V[-1]
    F = f.reshape((3, 3))

    U, S, V = np.linalg.svd(F)
    S_matrix = np.zeros((3, 3))
    S_matrix[:2, :2] = np.diag(S[:2])
    F_matrix = np.dot(U, np.dot(np.diag(S), V))

    T1 = np.array([[1/scale, 0, 0], [0, 1/scale, 0], [0, 0, 1]])
    T2 = np.array([[1/scale, 0, 0], [0, 1/scale, 0], [0, 0, 1]])
    F_denormalized = np.dot(T2.T, np.dot(F_matrix, T1))

    return F_denormalized

    ####################################
    ##########YOUR CODE HERE############
    ####################################

    ####################################
    return F 


def compute_epipolar_correspondences(img1, img2, pts1, F,threshold = 500,window_size = 5):
    """
    Compute epipolar correspondences in Image 2 for a set of points in Image 1 using the Fundamental matrix.

    Given two images (img1 and img2), a set of 2D points (pts1) in Image 1, and the Fundamental matrix (F)
    that relates the two images, this function calculates the corresponding 2D points (pts2) in Image 2.
    The computed pts2 are the epipolar correspondences for the input pts1.

    Parameters:
    img1 (numpy.ndarray): The first image containing the points in pts1.
    img2 (numpy.ndarray): The second image for which epipolar correspondences will be computed.
    pts1 (numpy.ndarray): An Nx2 array of 2D points in Image 1.
    F (numpy.ndarray): The 3x3 Fundamental matrix that relates img1 and img2.

    Returns:
    pts2_ep (numpy.ndarray): An Nx2 array of corresponding 2D points (pts2) in Image 2, serving as epipolar correspondences
                   to the points in Image 1 (pts1).
    """

    sy, sx, sd = img2.shape

    pts2 = []
    x = pts1[0,0]
    y = pts1[0,1]
    v = np.array([x, y, 1])

    l = F @ v
    s = np.sqrt(l[0]**2+l[1]**2)

    if s==0:
        error('Zero line vector in displayEpipolar')

    l = l / s
    if l[1] != 0:
        xs = 0
        xe = sx - 1
        ys = -(l[0] * xs + l[2]) / l[1]
        ye = -(l[0] * xe + l[2]) / l[1]
    else:
        ys = 0
        ye = sy - 1
        xs = -(l[1] * ys + l[2]) / l[0]
        xe = -(l[1] * ye + l[2]) / l[0]


    lowest = 1000000000
    point = [0,0]
    #candidates = []
    #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    for i in range(min(xs, xe), max(xs, xe) + 1):

        j = int((-l[0] * i - l[2]) / l[1])

        if (
            i >= window_size // 2
            and i < img2.shape[1] - window_size // 2
            and j >= window_size // 2
            and j < img2.shape[0] - window_size // 2):
            
            '''half_size = window_size // 2
            x_start = max(0, i - half_size)
            x_end = min(img1.shape[1], i + half_size + 1)
            y_start = max(0, j - half_size)
            y_end = min(img1.shape[0], j + half_size + 1)
            window2 = img2[y_start:y_end, x_start:x_end]
            window1 = img1[y_start:y_end, x_start:x_end]

            '''

            window2 = img2[j - window_size // 2 : j + window_size // 2 + 1, i - window_size // 2 : i + window_size // 2 + 1]
            window1 = img1[y - window_size // 2 : y + window_size // 2 + 1, x - window_size // 2 : x + window_size // 2 + 1]

            #print(window1)
            #print(window2)
            #print(img1[i-1:i+1,j-1:j+1])
            #print(img1[i,j])
            similarity = np.sum(np.abs(window1 - window2))
            if similarity < lowest: 
                point = [i,j]
                lowest = similarity
            else: continue

    pts2 = np.array(point)

            #if similarity < threshold:
            #    candidates.append([i, j])


   #pts2_ep = np.array(candidates)

    ####################################
    ##########YOUR CODE HERE############
    ####################################
   
    ####################################
    return pts2


def compute_essential_matrix(K1, K2, F):
    """
    Compute the Essential matrix from the intrinsic matrices and the Fundamental matrix.

    Given the intrinsic matrices of two cameras (K1 and K2) and the 3x3 Fundamental matrix (F) that relates
    the two camera views, this function calculates the Essential matrix (E).

    Parameters:
    K1 (numpy.ndarray): The 3x3 intrinsic matrix for Camera 1.
    K2 (numpy.ndarray): The 3x3 intrinsic matrix for Camera 2.
    F (numpy.ndarray): The 3x3 Fundamental matrix that relates Camera 1 and Camera 2.

    Returns:
    E (numpy.ndarray): The 3x3 Essential matrix (E) that encodes the essential geometric relationship between
                   the two cameras.

    """

    K2_transpose = np.transpose(K2)
    E = K2_transpose @ F.T @ K1   
    ####################################
    ##########YOUR CODE HERE############
    ####################################
    
    ####################################
    return E 

def triangulate_points(E, pts1_ep, pts2_ep, K1, K2):
    """
    Triangulate 3D points from the Essential matrix and corresponding 2D points in two images.

    Given the Essential matrix (E) that encodes the essential geometric relationship between two cameras,
    a set of 2D points (pts1_ep) in Image 1, and their corresponding epipolar correspondences in Image 2
    (pts2_ep), this function calculates the 3D coordinates of the corresponding 3D points using triangulation.

    Extrinsic matrix for camera1 is assumed to be Identity. 
    Extrinsic matrix for camera2 can be found by cv2.decomposeEssentialMat(). Note that it returns 2 Rotation and 
    one Translation matrix that can form 4 extrinsic matrices. Choose the one with the most number of points in front of 
    the camera.

    Parameters:
    E (numpy.ndarray): The 3x3 Essential matrix that relates two camera views.
    pts1_ep (numpy.ndarray): An Nx2 array of 2D points in Image 1.
    pts2_ep (numpy.ndarray): An Nx2 array of 2D points in Image 2, corresponding to pts1_ep.

    Returns:
    point_cloud (numpy.ndarray): An Nx3 array representing the 3D point cloud, where each row contains the 3D coordinates
                   of a triangulated point.
    point_cloud_cv (numpy.ndarray): An Nx3 array representing the 3D point cloud, where each row contains the 3D coordinates
                   of a triangulated point calculated using cv2.triangulate
    """
    
    extrinsic_matrix_camera1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    R1, R2, T, = cv2.decomposeEssentialMat(E)
    # Extrinsic matrix for Camera 2
    extrinsic_matrix_camera2 = np.array([np.hstack([R1, T]), np.hstack([R1, -T]), np.hstack([R2, T]), np.hstack([R2, -T])])
    #print(extrinsic_matrix_camera2)
    #print(np.shape(K1))

    P1 = K1 @ extrinsic_matrix_camera1
    P2 = np.array([K1 @ x for x in extrinsic_matrix_camera2])

    

    #print(P1)
    #print(P2)



    point_cloud = None
    point_cloud_cv = None

    points_3d = []

    for pt1, pt2 in zip(pts1_ep, pts2_ep):

        # forming equation foe Ax = 0
        A = np.vstack([
            pt1[0] * P1[2, :] - P1[0, :],
            pt1[1] * P1[2, :] - P1[1, :],
            pt2[0] * P2[0][2, :] - P2[0][0, :],
            pt2[1] * P2[0][2, :] - P2[0][1, :]
        ])

        _, _, V = np.linalg.svd(A)

        point_3d = V[-1, :3] / V[-1, 3]

        points_3d.append(point_3d)


    #points_4d = cv2.triangulatePoints(P1, P2[0], pts1_ep.T, pts2_ep.T)
    # Normalize the 4D points to obtain 3D points
    #points_3d_cv = points_4d[:3] / points_4d[3]
    #print(points_3d_cv)

    points_3d = np.array(points_3d)
    
    #sprojection error of camera 2
    total_error_2 = 0.0
    num_points = pts2_ep.shape[0]
    for i in range(num_points):
        X_i = np.append(points_3d[i], 1)  # Homogeneous coordinates of 3D point
        projected_2d_camera2 = P2[0] @ X_i  # Projected 2D point
        projected_2d_camera2 /= projected_2d_camera2[2] 

        error = np.linalg.norm(projected_2d_camera2[:2] - pts2_ep[i], 2)  
        total_error_2 += error

    reproj_error_2 = (1 / num_points) * total_error_2
    print(reproj_error_2)

    total_error_1 = 0.0

    # For projection error of camera 1
    for i in range(num_points):
        projected_2d_camera1 = P1 @ X_i  # Projected 2D point
        projected_2d_camera1 /= projected_2d_camera1[2] 

        error = np.linalg.norm(projected_2d_camera1[:2] - pts1_ep[i], 2)  
        total_error_1 += error

    reproj_error_1 = (1 / num_points) * total_error_1
    print(reproj_error_1)

    return points_3d#, points_3d_cv


    ####################################
    ##########YOUR CODE HERE############
    ####################################
    
    ####################################
    #return point_cloud, point_cloud_cv


def visualize(point_cloud):
    """
    Function to visualize 3D point clouds
    Parameters:
    point_cloud (numpy.ndarray): An Nx3 array representing the 3D point cloud,where each row contains the 3D coordinates
                   of a triangulated point.
    """
    ####################################
    ##########YOUR CODE HERE############
    ####################################
    
    ####################################

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]
    

    ax.scatter(x, y, z, c='b', marker='o',s=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud Visualization')
    
    plt.show()



if __name__ == "__main__":
    data_for_fundamental_matrix = np.load("data/corresp_subset.npz")
    pts1_for_fundamental_matrix = data_for_fundamental_matrix['pts1']
    pts2_for_fundamental_matrix = data_for_fundamental_matrix['pts2']

    img1 = cv2.imread('data/im1.png')
    img2 = cv2.imread('data/im2.png')
    scale = max(img1.shape)
    

    data_for_temple = np.load("data/temple_coords.npz")
    pts1_epipolar = data_for_temple['pts1']

    data_for_intrinsics = np.load("data/intrinsics.npz")
    K1 = data_for_intrinsics['K1']
    K2 = data_for_intrinsics['K2']
    F = compute_fundamental_matrix(pts1_for_fundamental_matrix, pts2_for_fundamental_matrix, scale)
    

    pts2_ep = []
    #pts2_ep = compute_epipolar_correspondences(img1, img2, pts1_epipolar, F)
    #print(pts2_ep)
    for i in range(len(pts1_epipolar)):
        temp = compute_epipolar_correspondences(img1, img2, np.array([pts1_epipolar[i]]), F.T)
        pts2_ep.append(temp.tolist())
    pts2_ep = np.array(pts2_ep)
    

    #print(type(pts2_ep))
    #print(np.shape(pts2_ep))
    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(20, 15))
    #f.tight_layout()
    ax1.set_title('The original points')
    ax1.set_axis_off()
    for i in range(len(pts2_ep)): ax2.scatter(pts2_ep[i,0],pts2_ep[i,1], marker = 'x')
    ax2.set_title('The corresponding points')
    ax2.set_axis_off()
    for i in range(len(pts1_epipolar)): ax1.scatter(pts1_epipolar[i,0],pts1_epipolar[i,1], marker = 'x')
    #for i in range(len(pts2_ep)): plt.scatter(pts2_ep[i,0],pts2_ep[i,1], marker = 'x')
    #plt.show()
    #for i in range(len(pts1_epipolar)): plt.scatter(pts1_epipolar[i,0],pts1_epipolar[i,1], marker = 'x')
    plt.show()
    _helper.epipolar_lines_GUI_tool(img1, img2, F.T)
    _helper.epipolar_correspondences_GUI_tool(img1, img2, F.T)
    E = compute_essential_matrix(K1, K2, F.T)
    #print(E)
    cloud_points = triangulate_points(E, pts1_epipolar, pts2_ep, K1, K2)
    #print(cloud_points)
    visualize(cloud_points)
    #print(cv)
    ####################################
    ##########YOUR CODE HERE############
    ####################################

    ####################################