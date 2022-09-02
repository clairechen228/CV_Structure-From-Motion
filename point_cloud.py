import open3d as o3d
from PIL import Image
import numpy as np
import os
import cv2
import pandas
import copy

def readfile():
    X = pandas.read_csv('./3dx.csv', header = 0, dtype=np.float64)
    track3d = X.to_numpy()

    Col = pandas.read_csv('./color.csv', header = 0, dtype=np.float64)
    colors = Col.to_numpy() / 255.0

    P = pandas.read_csv('./pose.csv', header = 0, dtype=np.float64)
    Pose = P.to_numpy()
    R =[]
    C = []
    for row in Pose:
        R.append(row[:9].reshape(3,3))
        C.append(row[9:])
    R = np.asarray(np.stack(R))
    C = np.asarray(np.stack(C))

    return track3d, R, C, colors

    

def ShowPC(track3d, R, C, colors):
    All = []

    all_mesh = []
    for i in range(len(C)):
        all_mesh.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1).rotate(R[i], C[i]))
        all_mesh[i].scale(1, center=all_mesh[i].get_center())
    o3d.io.write_triangle_mesh("./cam_scipy.ply", all_mesh[0])
    All = all_mesh
    

    maska = np.sum(track3d, axis=1) != -3 
    maskb = np.logical_and( np.logical_and(abs(track3d[:,0]) < 30 ,  abs(track3d[:,1]) < 30), track3d[:,2] <30)
    mask = np.logical_and(maska, maskb)
    X = track3d[mask]
    

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(X)
    pcd.colors = o3d.utility.Vector3dVector(colors[mask])
    All.append(pcd)

    o3d.visualization.draw_geometries(All)
    o3d.io.write_point_cloud("./pc_scipy.pcd", pcd)
    






def main():
    K = np.asarray([
    [350, 0, 480],
    [0, 350, 270],
    [0, 0, 1] ])
    track3d, R, C, colors = readfile()


    ShowPC(track3d, R, C, colors)

if __name__ == '__main__':
    main()