import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import time

def SetupBundleAdjustment(track3d, track2d, imgs, p, x):
    x0 = np.array([])
    for pp in range(p):
        r = imgs[pp].R 
        q_from_r = R.from_matrix(r)
        q = q_from_r.as_quat()
        x0 = np.append(x0, q)
        x0 = np.append(x0, imgs[pp].C)
    x0 = np.append(x0, track3d.reshape(-1))

    visible = np.logical_and(track2d[:,:,0]!= -1, track2d[:,:,1]!= -1)
    construct =  np.tile((np.sum(track3d, axis=1) != -3) , (len(visible),1))
    mask = np.logical_and(visible, construct)
    camera_index,point_index = np.nonzero(mask)

    b = track2d[camera_index,point_index].reshape(-1)

    n_visible = len(camera_index)
    A = np.zeros((2*n_visible, 7*p+3*x))
 
    # b = np.array([])
    for i in range(n_visible):
        pp = camera_index[i]
        xx = point_index[i]
        # b = np.append(b, track2d[pp,xx,:])
        if pp != 0 and pp != 1:
            A[2*i:2*(i+1), 7*pp:7*(pp+1)] = 1
        A[2*i:2*(i+1), 7*p+3*xx:7*p+3*(xx+1)] = 1            

    return x0, b, camera_index, point_index, A



def fun(params, b, p, x, camera_index, point_index):
    qnc = params[:7*p].reshape((p,7))
    X = params[7*p:].reshape((x,3))
    x0 = []
    
    for (pp,xx) in zip(camera_index, point_index):
        q = qnc[pp,:4] / np.linalg.norm(qnc[pp,:4])
        r_from_q = R.from_quat(q)
        r = r_from_q.as_matrix()
        c = qnc[pp,4:]
        u,v,w =  (X[xx] - c) @ r.T
        x0.extend([u/w, v/w])
    error = np.abs(np.asarray(x0)-b)    

    return error

def Error(imgs, track3d, b, camera_index, point_index):
    err = 0
    i = 0
    for (pp,xx) in zip(camera_index, point_index):
        u,v,w =  (track3d[xx] - imgs[pp].C) @ imgs[pp].R.T
        err += np.linalg.norm(np.array([u/w, v/w]) - b[i:i+2])
        i += 2
    return err



def UpdatePosePoint(params, imgs, track3d, p, x):
    track3d = params[7*p:].reshape((x,3))

    qnc = params[:7*p].reshape((p,7))
    for i in range(p):
        q = qnc[i,:4] / np.linalg.norm(qnc[i,:4])
        r_from_q = R.from_quat(q)
        r = r_from_q.as_matrix()
        imgs[i].R = r

        c = qnc[i,4:]
        imgs[i].C = c

    return track3d


def RunBundleAdjustment_scipy(track3d, track2d, imgs):
    
    p = len(imgs)
    x = len(track3d)

    start = time.time()
    x0, b, camera_index, point_index, A = SetupBundleAdjustment(track3d, track2d, imgs, p, x)
    prev_err = Error(imgs, track3d, b, camera_index, point_index)

    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf', args=(b, p, x, camera_index, point_index))
    track3d = UpdatePosePoint(res.x, imgs, track3d, p, x)
    final_err = Error(imgs, track3d, b, camera_index, point_index)
    end = time.time()

    print('computing time', (end-start))
    print('Bundle Adjustment error', final_err-prev_err)

    return track3d