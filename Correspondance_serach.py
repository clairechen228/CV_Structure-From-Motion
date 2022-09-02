import os 
import cv2
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt


class SIFT:
    """
    kps is a list of keypoints: N
    descs is a numpy array of shape: (Number of Keypoints)x128.
    """
    def __init__(self, path):
        self.img = cv2.imread(path)
        self.color = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        # self.img = np.asarray(Image.open(path))
        self.kps = None 
        self.dscts = None
        self.C = None
        self.R = None
        self.find()
    
    def find(self):
        sift = cv2.SIFT_create()
        self.kps, self.dscts = sift.detectAndCompute(self.img, None)



def feature_matching(imgx, imgy, thr_match):
    """
    Input
    imgx, imgy are SIFT objects
    thr_match is the ratio of the 1st and 2nd matched distance: scalar
    -
    Output
    fx is the matched keypoint locations in imgx: np.array(N x 2)
    fy is the matched keypoint locations in imgy: np.array(N x 2)
    indx is a list of indices of the matched keypoints in imgx: N
    dmatch is a DMatch object: (Number of matched keypoints)
    """
    #DMatch.distance - Distance between descriptors. The lower, the better it is.
    #DMatch.trainIdx - Index of the descriptor in train descriptors
    #DMatch.queryIdx - Index of the descriptor in query descriptors
    #DMatch.imgIdx - Index of the train image.

    bf = cv2.BFMatcher()
    #options: normType = NORM_L1, crossCheck = False
    #to get k best matches
    matches = bf.knnMatch(imgx.dscts, imgy.dscts, k=2)
    
    good = []
    fx = []
    fy = []
    dmatch = []
    indx = []
    for m,n in matches:
        if m.distance < thr_match * n.distance:
            good.append([m])
            fx.append(imgx.kps[m.queryIdx].pt)
            fy.append(imgy.kps[m.trainIdx].pt)
            dmatch.append(m)
            indx.append(m.queryIdx)

    return fx, fy, indx, dmatch



def estimaetE(random8):
    random8 = np.asarray(random8)
    A = []
    for i in range(8):
        x1, y1, x2, y2 = random8[i][0], random8[i][1],  random8[i][2],  random8[i][3]
        a = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]
        A.append(a)
    A = np.stack(A)

    U, S, Vt = np.linalg.svd(A)
    E = Vt.T[:,-1].reshape(3,3)

    U, S, V = np.linalg.svd(E)
    D = np.eye(3)
    D[2,2] = 0
    E = U @ D @ V

    return E



def RANSAC(iter, thr, f1, f2):
    """
    Input
    iter is the number of RANSAC iterations
    thr is the error threshold
    f1 is the set of correspondences in the first image: ndarray of shape (n, 2)
    f2 is the set of correspondences in the second image: ndarray of shape (n, 2)
    -
    Output
    E is the essential matrix: 3 x 3
    inlier_mask is the a boolean list of the correspondances Mab: n
    """
    xa = np.insert(f1, 2, 1, axis=1)
    xb = np.insert(f2, 2, 1, axis=1)
    E = np.zeros((3,3))
    inlier_idx = None
    max_n_inlier = 0
    for i in range(iter):
        #N x 4
        corrs = np.hstack((f1 , f2)).tolist()
        random8 = random.sample(corrs, 8)
        e = estimaetE(random8)
        # print(e)
        #n x n 
        dis = np.abs(np.diag(xb @ e @ xa.T))
        mask = dis < thr
        n_inlier = np.sum(mask)
        # print(dis)
        if n_inlier > max_n_inlier:
            max_n_inlier = n_inlier
            E = e
            inlier_mask = mask
    
    return E, inlier_mask



def TwoDx(K, imgs, thr_match, iter, thr_ransac):
    """
    Input
    K is the intrinsic matrix: 3 x 3
    imgs is a list of SIFT objects
    thr_match is the ratio for 'feature_matching'
    iter is the number of iterations for 'RANSAC'
    thr_ransac is the error threshold for 'RANSAC'
    -
    Output
    track: 6 x (at most the sum of features in all images) x 2
    E1 is the essential matrix of img0 and img1: 3 x 3
    """
    ## This track uses homogeneous coordinates
    n = len(imgs)

    track = np.array([]).reshape((n,0,2))
    colors = np.array([]).reshape((0,3))
    for i in range(n-1):
        f = len(imgs[i].kps)
        track_i = -1 * np.ones((n,f,2))
        colors_i = -1 * np.ones((f, 3))
        for j in range(i+1, n):
            print('---------Finding matches btw img%d and img%d---------'%(i, j))
            fx, fy, matched_indx, _ = feature_matching(imgs[i], imgs[j], thr_match)

            # normalize
            norm_fx = (np.insert(fx, 2, 1, axis=1) @ np.linalg.inv(K).T)[:,:2]
            norm_fy = (np.insert(fy, 2, 1, axis=1) @ np.linalg.inv(K).T)[:,:2]

            #run RANSAC
            E, inlier_indx = RANSAC(iter, thr_ransac, norm_fx, norm_fy) #(iter=1000, thr=0.005, fx, fy) 200/0.05
            #save E1
            if i == 0 and j == 1:
                E1 = E

            track_indx = np.array(matched_indx)[inlier_indx]
            track_i[i,track_indx,:] = np.array(norm_fx)[inlier_indx]
            track_i[j,track_indx,:] = np.array(norm_fy)[inlier_indx]
            
            colors_i[track_indx,:] = imgs[i].color[np.round(np.asarray(fx)[:,1]).astype('int32'), np.round(np.asarray(fx)[:,0]).astype('int32')][inlier_indx] 
            print('# of matched features before/after RANSAC is', len(fx), len(track_indx))
    
        matched_features_indx = np.sum(track_i[i], axis = 1) != -2
        track_i = track_i[:, matched_features_indx, :]
        colors_i = colors_i[matched_features_indx, :]

        track = np.concatenate([track, track_i], axis=1)
        colors = np.vstack((colors, colors_i))

    return track, E1, colors



def camera_pose(E):
    """
    Input 
    E is the essential matrix: 3 x 3
    -
    Output
    C is a list of 4 camera centers: 4 x 3
    R is a list of 4 rotation matrices: 4 x 3 x 3
    """
    U,D,Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])
    
    t1 = U[:,2]
    R1 = U @ W @ Vt
    if np.linalg.det(R1) < 0:
        t1 = -t1
        R1 = -R1
    C1 = -R1.T @ t1

    t2 = -U[:,2]
    R2 = U @ W @ Vt
    if np.linalg.det(R2) < 0:
        t2 = -t2
        R2 = -R2
    C2 = -R2.T @ t2

    t3 = U[:,2]
    R3 = U @ W.T @ Vt
    if np.linalg.det(R3) < 0:
        t3 = -t3
        R3 = -R3
    C3 = -R3.T @ t3

    t4 = -U[:,2]
    R4 = U @ W.T @ Vt
    if np.linalg.det(R4) < 0:
        t4 = -t4
        R4 = -R4
    C4 = -R4.T @ t4

    C = [C1, C2, C3, C4]
    R = [R1, R2, R3, R4]

    return C, R



def SkewMatrix(f):
    """
    Input
    f is the matched keypoint locations [x, y]: 1 x 2
    -
    Output
    skew is the Skew-symmetric matrix: 3 x 3
    """
    a, b, c = f[0], f[1], 1
    skew = np.array([[0, -c, b],
                     [c, 0, -a],
                     [-b, a, 0]])
    return skew



def Triangulation(Pa, Pb, f1, f2):
    """
    Input
    Pa is the camera projection matrix a: 3 x 4
    Pb is the camera projection matrix b: 3 x 4
    f1 is the set of correspondences in the first image: ndarray of shape (n, 2)
    f2 is the set of correspondences in the second image: ndarray of shape (n, 2)
    -
    Output
    X is the set of 3D points: (Number of inliers) x 3
    """
    n = len(f1)
    X = np.zeros((n,3))
    for i in range(n):
        a = SkewMatrix(f1[i])
        b = SkewMatrix(f2[i])
        Aa = a @ Pa
        Ab = b @ Pb
        A = np.vstack((Aa, Ab))
        U, S, Vt = np.linalg.svd(A)
        p = Vt[-1]   
        x = p[:3] / p[3]
        X[i] = x

    return X



def Cheirality(C, R, f1, f2, imga, imgb):
    """
    Input
    K is the camera intrinsic matrix
    C is a list of 4 camera centers: 4 x 3
    R is a list of 4 rotation matrices: 4 x 3 x 30
    f1 is the set of correspondences in the first image: ndarray of shape (n, 2)
    f2 is the set of correspondences in the second image: ndarray of shape (n, 2)
    imga, imgb are the SIFT objects
    -
    Output
    set P, C, R in imgb
    X is the set of 3D points according to 1 of the 4 camera poses: (Number of inliers) x 3
    
    """
    max_nValid = 0
    corrX = None

    Pa = imga.R @ (np.hstack((np.eye(3), -imga.C.reshape(3,1))))
    max_n_valid = 0
    for i in range(4):
        Pb = R[i] @ (np.hstack((np.eye(3),-C[i].reshape(3,1))))
        X = Triangulation(Pa, Pb, f1, f2)
        deta = (X - imga.C) @ (imga.R[-1]).T
        detb = (X - C[i])  @ (R[i][-1]).T
        mask = np.logical_and(deta > 0, detb > 0)
        nValid = np.sum(mask)
        print('nValid is ', nValid)

        if nValid > max_n_valid:
            max_n_valid = nValid
            corrX = X[mask]
            ch_mask = mask
            imgb.P = Pb
            imgb.C = C[i]
            imgb.R = R[i]

    return corrX, ch_mask
