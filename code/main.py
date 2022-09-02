import numpy as np
import os
import cv2
import csv
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import open3d as o3d

from scipy.linalg import block_diag
from scipy.optimize import least_squares

from Correspondance_serach import *
from PnP import *
from plots import *
from BA import RunBundleAdjustment
from BA_scipy import RunBundleAdjustment_scipy

def getP(img):
    P = img.R @ (np.hstack((np.eye(3), -img.C.reshape(3,1))))
    return P

def updatePose(img, c, r): 
    img.C = c
    img.R = r
   

def savePX(imgs, track3d, colors):
    headX = ['x', 'y', 'z']
    with open('./3dx.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(headX)
        writer.writerows(track3d)

    headC = ['r', 'g', 'b']
    with open('./color.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(headC)
        writer.writerows(colors)


    headP = ['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33', 'c1', 'c2','c3']
    with open('./pose.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(headP)
        for img in imgs:
            r = img.R.reshape(-1)
            c = img.C
            row = np.append(r,c)
            writer.writerow(row)


def main():
    ## -----Parameters----- 
    show = False
    folder_path = 'C:/Users/garba/Documents/Project/new_3Dreconstruction/hw_images'
    K = np.asarray([
        [350, 0, 480],
        [0, 350, 270],
        [0, 0, 1] ])
    ## ---------------------

    raw_imgs = list(sorted(os.listdir(folder_path)))
    n = len(raw_imgs)

    ##Feature extraction
    imgs = []
    for name in raw_imgs:
        path = os.path.join(folder_path, name)
        name = SIFT(path)
        imgs.append(name)
    print('***Finished feature extraction ')
    #imgs is a list of SIFT objects
    
    ######for debugging
    # drawkp(imgs[0],False)
    # drawmatch(K, imgs[0], imgs[1], False)
    # drawRANSAC(K, 3000, 0.7, 0.005, imgs[0], imgs[1], True)
    # checkTwoDx(K, imgs, 0.7, 1000, 0.003)
    # drawCameras(K, imgs, 0.7, 1000, 0.003, 1, False)
    # draw3Dpoints(K, imgs, 0.7, 1000, 0.003, False)
    # drawCheirality(K, imgs, 0.7, 1000, 0.003, False)


    ##Build feature tracks
    track2d, E1, colors = TwoDx(K, imgs, 0.7, 1000, 0.007)
    print('***Finished feature tracks builing ')
    track3d = -1*np.ones((len(track2d[0]),3))

    
    #Set the 1st camera pose to the world origin
    updatePose(imgs[0], np.zeros(3), np.eye(3))


    ##Estimate the 2nd camera pose and construct 3D points
    print('---------Estimating the 2nd camera pose---------')
    f12_mask = np.logical_and(np.sum(track2d[0], axis=1) != -2, np.sum(track2d[1], axis=1) != -2)
    f1, f2 = track2d[0][f12_mask], track2d[1][f12_mask]
    Cset, Rset = camera_pose(E1)
    l_X, l_ch_mask = Cheirality(Cset, Rset, f1, f2, imgs[0], imgs[1])
    f12_indx = (np.flatnonzero(f12_mask)[l_ch_mask])
    track3d[f12_indx] = l_X
		

    ##Refine and update 3D points
    x = np.hstack((f1[l_ch_mask],f2[l_ch_mask]))
    nl_X, nl_ch_mask = Triangulation_nl(imgs[0], imgs[1], l_X, x, 100, 0.003)
    f12_indx = (np.flatnonzero(f12_mask)[l_ch_mask])[nl_ch_mask]
    track3d[f12_indx] = nl_X
    # plotTriangulation(l_X[nl_ch_mask], nl_X, imgs[1], K, track2d[1,f12_indx]) #

##########
    for i in range(2,n):
        #Estimae new camera poses
        print('---------Estimating the camera pose of img %d---------'%(i))
        fi_mask = np.logical_and(np.sum(track2d[i], axis=1) != -2 , np.sum(track3d, axis=1) != -3)
        X = track3d[fi_mask]
        x = track2d[i, fi_mask]
        r, c, inlier_mask = PnP_RANSAC(X, x, 1000, 0.003, -1)
        updatePose(imgs[i], c, r)
      

        #Refine new camera pose
        r, c = PnP_nl(r, c, X, x, 300)
        updatePose(imgs[i], c, r)
        

        #Construct new 3d points
        for j in range(i-1, -1, -1):   
            fij_mask = np.logical_and(np.sum(track2d[i], axis=1) != -2, np.sum(track2d[j], axis=1) != -2)
            X_mask = np.sum(track3d, axis=1) == -3
            miss_mask = np.logical_and(fij_mask, X_mask)
            f1, f2 = track2d[i][miss_mask], track2d[j][miss_mask]
            x = np.hstack((f1,f2))

            l_Xnew = Triangulation(getP(imgs[i]), getP(imgs[j]), f1, f2)
            nl_Xnew, nl_ch_mask = Triangulation_nl(imgs[i], imgs[j], l_Xnew, x, 300, 0.001) #300
            # plotTriangulation(l_Xnew, nl_Xnew, imgs[i], K, track2d[i, miss_mask]) #

            print('---------Reconstructed %d matched points between img %d and img %d---------'%(len(nl_Xnew),i,j)) 
            fij_indx = np.flatnonzero(miss_mask)[nl_ch_mask]
            track3d[fij_indx] = nl_Xnew

    		#Choose one of the BA mthods below
        ### BA.py ###
        track3d = RunBundleAdjustment(track3d, track2d[:i+1], imgs[:i+1], 50, i)
        ### BA_scipy.py ###
        # track3d = RunBundleAdjustment_scipy(track3d, track2d[:i+1], imgs[:i+1])

        # show3dcameras(imgs[:i+1]) #
        # show3dpoints(track3d) #

    print('Total # of 3D points is %d'%len(track3d))
    savePX(imgs, track3d, colors)
    


if __name__ == '__main__':
    main()
