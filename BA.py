import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import time

def ComputePointJacobianall(img, X):
    r = img.R
    C = img.C
    q_from_r = R.from_matrix(r)
    q = q_from_r.as_quat() 
    qx, qy, qz, qw = q

    #fx
    u,v,w =  (X - C) @ r.T
    fx = np.array([u/w, v/w])

    #jp
    dr_dq = np.array([[    0,  -4*qy,  -4*qz,     0],
                    [ 2*qy,   2*qx,  -2*qw, -2*qz],
                    [ 2*qz,   2*qw,   2*qx,  2*qy],
                    [ 2*qy,   2*qx,   2*qw,  2*qz],
                    [-4*qx,      0,  -4*qz,     0],
                    [-2*qw,   2*qz,   2*qy,  2*qx],
                    [ 2*qz,  -2*qw,   2*qx, -2*qy],
                    [ 2*qw,   2*qz,   2*qy,  2*qx],
                    [-4*qx,   -4*qy,     0,     0]])

    duvw_dr = np.zeros((3,9))
    duvw_dr[0, :3] = X-C
    duvw_dr[1, 3:6] = X-C
    duvw_dr[2, 6:] = X-C

    duvw_dq = duvw_dr @ dr_dq
    du_dq, dv_dq, dw_dq = duvw_dq
    df_dq = np.array([(w*du_dq - u*dw_dq) / w**2 , (w*dv_dq - v*dw_dq) / w**2])

    du_dc = -1 * r[0]
    dv_dc = -1 * r[1]
    dw_dc = -1 * r[2]
    df_dc = np.array([(w*du_dc - u*dw_dc) / w**2, (w*dv_dc - v*dw_dc) / w**2])

    jp = np.hstack((df_dc, df_dq))


    #jx
    du_dX = r[0]
    dv_dX = r[1]
    dw_dX = r[2]
    df_dX = np.array([(w*du_dX - u*dw_dX) / w**2, (w*dv_dX - v*dw_dX) / w**2])
    
    jx = df_dX

    return jp, jx, fx


def UpdatePosePoint(deltaP, deltaX, imgs, track3d):
    #deltaP: (p,7), deltaX: (x,3)
    n = len(imgs)
    Rnew = np.zeros((n,3,3))
    Cnew = np.zeros((n,3))

    for pp in range(n):
        #update r
        r = imgs[pp].R 
        q_from_r = R.from_matrix(r)
        q = q_from_r.as_quat()
        q += deltaP[pp,3:]
        q = q / np.linalg.norm(q)
        r_from_q = R.from_quat(q)
        r = r_from_q.as_matrix()
        imgs[pp].R = r
        Rnew[pp] = r

        #update c
        imgs[pp].C += deltaP[pp,:3]
        Cnew[pp] = imgs[pp].C 

    Xnew = track3d + deltaX

    return Xnew, Rnew, Cnew


def Error(track3d, track2d, imgs):
    err = 0
    for xx in range(len(track3d)):
        for pp in range(len(imgs)):
            visible = np.logical_and(np.sum(track2d[pp,xx]) != -2, np.sum(track3d[xx]) != -3)
            if visible:
                u,v,w =  (track3d[xx] - imgs[pp].C) @ imgs[pp].R
                f = np.array([u/w, v/w])
                b =track2d[pp,xx]
                err += np.linalg.norm((f - b))
    return err


def set_min(imgs, track3d):
    min_X = track3d

    n = len(imgs)
    Rnew = np.zeros((n,3,3))
    Cnew = np.zeros((n,3))

    for i in range(n):
        Rnew[i] = imgs[i].R
        Cnew[i] = imgs[i].C

    return min_X, Rnew, Cnew
    

def UpdateFinalPosePoint(imgs, min_X, min_R, min_C):
    n = len(imgs)
    for i in range(n):
        imgs[i].R = min_R[i]
        imgs[i].C = min_C[i]

    return min_X


def RunBundleAdjustment(track3d, track2d, imgs, iter, j):
    lamda = 1

    min_error = Error(track3d, track2d, imgs)
    all_error = [min_error]
    min_X, min_R, min_C = set_min(imgs, track3d)

    x = len(track3d)
    p = len(imgs)
    for i in range(iter):
        #J=[Jp, Jx]: (7p+3x,2p)
        #Jp: (2p,7p), Jx: (2p,3x)
        Jp = np.array([]).reshape(0,7*p)
        Jx = np.array([]).reshape(0,3*x)
        b = np.array([])
        f = np.array([])
        Dinv = np.array([])

        start = time.time() #start----------------------------
        for xx in range(x):
            d = np.zeros((3,3))
            for pp in range(p):
                visible = np.logical_and(np.sum(track2d[pp,xx]) != -2, np.sum(track3d[xx]) != -3)
                if visible:
                    Jp0 = np.zeros((2,7*p))
                    Jx0 = np.zeros((2,3*x))
                    jp, jx, fx = ComputePointJacobianall(imgs[pp], track3d[xx])
                    Jp0[:,7*pp:7*(pp+1)] = jp
                    Jx0[:,3*xx:3*(xx+1)] = jx
                    Jp = np.vstack((Jp,Jp0))
                    Jx = np.vstack((Jx,Jx0))
                    d = d + jx.T @ jx

                    b = np.append(b,track2d[pp,xx])
                    f = np.append(f,fx)

            d = d + lamda*np.eye(3)
            if xx == 0:
                Dinv = np.linalg.inv(d)
            else:
                Dinv = block_diag(Dinv, np.linalg.inv(d))

        
        ep = Jp.T @ (b-f)
        ex = Jx.T @ (b-f)
        A = Jp.T @ Jp + lamda*np.eye(7*p)
        B = Jp.T @ Jx

        # deltaP:7c x 1, deltaX: 3m x 1
        deltaP = np.linalg.inv(A -(B @ Dinv @ B.T)) @ (ep -(B @ Dinv @ ex))
        deltaX = Dinv @ (ex - (B.T @ deltaP))
        deltaP = deltaP.reshape(p,7)
        deltaX = deltaX.reshape(x,3)

        track3d, Rnew, Cnew = UpdatePosePoint(deltaP, deltaX, imgs, track3d)

        curr_error = Error(track3d, track2d, imgs)
        all_error.append(curr_error)

        if curr_error < min_error:
            min_error = curr_error
            min_X = track3d
            min_R = Rnew
            min_C = Cnew

    track3d = UpdateFinalPosePoint(imgs, min_X, min_R, min_C)
    end = time.time()   #end------------------------------
    print('computing time', (end-start))

    print('Bundle Adjustment error', (min(all_error) - all_error[0]))
    ax = plt.subplot(111)
    ax.plot(list(range(iter+1)), all_error)
    txt = 'img' + str(j) + '.png'
    plt.savefig(txt)
    # plt.show()

        
    return track3d