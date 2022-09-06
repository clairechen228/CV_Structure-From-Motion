import jax.numpy as jnp
import random
from jax import jacfwd, jacrev
import pandas
import numpy as jnp
import csv
from scipy.spatial.transform import Rotation as R
import time
from scipy.linalg import block_diag
import matplotlib.pyplot as plt


def f(r,C,X):
    u, v, w = jnp.dot((X-C), r.T) 
    return jnp.array([u/w, v/w])


def ComputePointJacobian(img, X):
    r = img.R
    C = img.C
    q_from_r = R.from_matrix(r)
    q = q_from_r.as_quat() 
    qx, qy, qz, qw = q

    fx = f(r,C,X)

    Jr = jnp.asarray(jacrev(f,0)(r,C,X))
    Jc = jnp.asarray(jacrev(f,1)(r,C,X))
    Jx = jnp.asarray(jacrev(f,2)(r,C,X))
    Jq = jnp.array([[    0,  -4*qy,  -4*qz,     0],
                    [ 2*qy,   2*qx,  -2*qw, -2*qz],
                    [ 2*qz,   2*qw,   2*qx,  2*qy],
                    [ 2*qy,   2*qx,   2*qw,  2*qz],
                    [-4*qx,      0,  -4*qz,     0],
                    [-2*qw,   2*qz,   2*qy,  2*qx],
                    [ 2*qz,  -2*qw,   2*qx, -2*qy],
                    [ 2*qw,   2*qz,   2*qy,  2*qx],
                    [-4*qx,   -4*qy,     0,     0]])
    
    Jp = jnp.hstack((Jc, jnp.dot(Jr,Jq)))

    return Jp, Jx, fx

def UpdatePosePoint(deltaP, deltaX, imgs, track3d):
    #deltaP: (p,7), deltaX: (x,3)
    n = len(imgs)
    Rnew = jnp.zeros((n,3,3))
    Cnew = jnp.zeros((n,3))

    for pp in range(n):
        #update r
        r = imgs[pp].R 
        q_from_r = R.from_matrix(r)
        q = q_from_r.as_quat()
        q += deltaP[pp,3:]
        q = q / jnp.linalg.norm(q)
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
            visible = jnp.logical_and(jnp.sum(track2d[pp,xx]) != -2, jnp.sum(track3d[xx]) != -3)
            if visible:
                u,v,w =  jnp.dot((track3d[xx] - imgs[pp].C) , imgs[pp].R)
                f = jnp.array([u/w, v/w])
                b = track2d[pp,xx]
                err += jnp.linalg.norm((f - b))
    return err
    
def set_min(imgs, track3d):
    min_X = track3d

    n = len(imgs)
    Rnew = jnp.zeros((n,3,3))
    Cnew = jnp.zeros((n,3))

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

def RunBundleAdjustment_jax(track3d, track2d, imgs, iter, j):
    lamda = 1

    min_error = Error(track3d, track2d, imgs)
    all_error = [min_error]
    min_X, min_R, min_C = set_min(imgs, track3d)

    x = len(track3d)
    p = len(imgs)
    for i in range(iter):
        #J=[Jp, Jx]: (7p+3x,2p)
        #Jp: (2p,7p), Jx: (2p,3x)
        Jp = jnp.array([]).reshape(0,7*p)
        Jx = jnp.array([]).reshape(0,3*x)
        b = jnp.array([])
        f = jnp.array([])
        Dinv = jnp.array([])

        start = time.time() #start----------------------------
        for xx in range(x):
            d = jnp.zeros((3,3))
            for pp in range(p):
                visible = jnp.logical_and(jnp.sum(track2d[pp,xx]) != -2, jnp.sum(track3d[xx]) != -3)
                if visible:
                    Jp0 = jnp.zeros((2,7*p))
                    Jx0 = jnp.zeros((2,3*x))
                    jp, jx, fx = ComputePointJacobianall(imgs[pp], track3d[xx])
                    Jp0.at[:,7*pp:7*(pp+1)].set(jp)
                    Jx0.at[:,3*xx:3*(xx+1)].set(jx)
                    Jp = jnp.vstack((Jp,Jp0))
                    Jx = jnp.vstack((Jx,Jx0))
                    d = d + jnp.dot(jx.T , jx)

                    b = jnp.append(b,track2d[pp,xx])
                    f = jnp.append(f,fx)

            d = d + lamda*jnp.eye(3)
            if xx == 0:
                Dinv = jnp.linalg.inv(d)
            else:
                Dinv = block_diag(Dinv, jnp.linalg.inv(d))

        
        ep = jnp.dot(Jp.T , (b-f))
        ex = jnp.dot(Jx.T , (b-f))
        A = jnp.dot(Jp.T , Jp) + lamda*jnp.eye(7*p)
        B = jnp.dot(Jp.T , Jx)

        # deltaP:7c x 1, deltaX: 3m x 1
        deltaP = jnp.dot(jnp.linalg.inv(A -(jnp.dot(jnp.dot(B , Dinv) , B.T))) , (ep -(jnp.dot(jnp.dot(B , Dinv) , ex))))
        deltaX = jnp.dot(Dinv , (ex - (jnp.dot(B.T , deltaP))))
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