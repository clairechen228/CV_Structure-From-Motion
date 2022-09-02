import numpy as np
import cv2 
import random
import copy
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


def PnP(X, x):
    """
    Input
    K, X(n,3), x(n,2)
    -
    Output
    C, R
    """
    A = np.array([]).reshape((0,12))
    for i in range(len(X)):
        a = np.array([X[i,0], X[i,1], X[i,2], 1, 0, 0, 0, 0, -x[i,0]*X[i,0], -x[i,0]*X[i,1], -x[i,0]*X[i,2], -x[i,0]])
        A = np.vstack((A,a))
        b = np.array([0, 0, 0, 0, X[i,0], X[i,1], X[i,2], 1, -x[i,1]*X[i,0], -x[i,1]*X[i,1], -x[i,1]*X[i,2], -x[i,1]])
        A = np.vstack((A,b))

    U, S, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape((3,4))

    U, D, Vt = np.linalg.svd(P[:, :3])
    # scale = D[0,0]
    R = U @ Vt
    t = np.divide(P[:, 3] , D[0])
  
    if np.linalg.det(R) < 0:
        R = -R
        t = -t
    C = -R.T @ t

    return C, R

def ReprojectionError(c, r, X, b):
    """
    Input
    C, r, X(n,3), b(2n,)
    -
    Output
    error(scalar)
    """
    uvw = (X - c) @ r.T 
    f = uvw[:,:2] / uvw[:,-1:]  #shape(n,2)
    b = b.reshape(-1,2) #shape(n,2)

    error_sum = np.linalg.norm((f-b)) #scalar
    error_vec = np.square(np.linalg.norm((f - b), axis=1)) #shape(n,)

    return error_sum, error_vec

def PnP_RANSAC(X, x, iter, thr, max_n_inlier):
    """
    Input
    X(n,3), x(n,2), iter, thr
    -
    Output 
    R, C, inlier_idx(n,)
    """
    inlier_mask = np.full(len(x), False)
    corrC = None
    corrR = None
    max_error_sum = 100
    
    for i in range(iter):
        #choose 6 correspondance
        corrs = np.hstack((X,x))
        random6 = np.asarray(random.choices(corrs, k=6))

        #get a new estimation of C and R
        c, r = PnP(random6[:,:3], random6[:,3:])

        #prjoect 3D points back into 2D spaces using new C and R
        error_sum, error_vec = ReprojectionError(c, r, X, x)


        mask = error_vec < thr
        n_inlier = np.sum(mask)

        if n_inlier > max_n_inlier and error_sum < max_error_sum:
            max_error_sum = error_sum
            max_n_inlier = n_inlier
            corrC = c
            corrR = r
            inlier_mask = mask
    
    print('# of inlier before/after PnP_RANSAC is ', len(x), np.sum(max_n_inlier))
    if max_n_inlier == -1 :
        raise Exception("all n_inliers < max_n_inlier")

    return corrR, corrC, inlier_mask




def JacobianCR(C, r, q, X):
    """
    Input
    C(3,), r(rotation:3x3), q(quaternion:1x4), X(3,), K(3,3)
    -
    Output 
    Jp(df_dp:2,7), fx([u/w, v/w]:1,2)
    """ 
    qx, qy, qz, qw = q
    u, v, w = (X - C) @ r.T 
    # 1 x 2
    fx = np.array([u/w, v/w])

    # 9 x 4
    dr_dq = np.array([[    0,  -4*qy,  -4*qz,     0],
                      [ 2*qy,   2*qx,  -2*qw, -2*qz],
                      [ 2*qz,   2*qw,   2*qx,  2*qy],
                      [ 2*qy,   2*qx,   2*qw,  2*qz],
                      [-4*qx,      0,  -4*qz,     0],
                      [-2*qw,   2*qz,   2*qy,  2*qx],
                      [ 2*qz,  -2*qw,   2*qx, -2*qy],
                      [ 2*qw,   2*qz,   2*qy,  2*qx],
                      [-4*qx,   -4*qy,     0,     0]])
    # 3 x 9
    duvw_dr = np.zeros((3,9))
    duvw_dr[0, :3] = X-C
    duvw_dr[1, 3:6] = X-C
    duvw_dr[2, 6:] = X-C

    #2 x 7
    duvw_dq = duvw_dr @ dr_dq
    du_dq, dv_dq, dw_dq = duvw_dq
    df_dq = np.array([(w*du_dq - u*dw_dq) / w**2 , (w*dv_dq - v*dw_dq) / w**2])

    # 2 x 3
    du_dc = -1 * r[0]
    dv_dc = -1 * r[1]
    dw_dc = -1 * r[2]
    df_dc = np.array([(w*du_dc - u*dw_dc) / w**2, (w*dv_dc - v*dw_dc) / w**2])


    # Jp = [dfdq dfdc]: (2 x 7), Jx: (2 x 3)
    Jp = np.hstack((df_dc, df_dq))

    return Jp, fx

def PnP_nl(r, C, X, x, iter):
    """
    Input
    r(3,3), C(3,), X(n,3), x(n,2), iter, thr
    -
    Output
    r(3,3), C(3,)
    """
    lamda = 0.01 #adjust

    q_from_r = R.from_matrix(r)
    q = q_from_r.as_quat() # shape(4,)
    b = x.reshape(-1) #shpae(2n,)

    all_error = []
    prev_error, initial_error_vec = ReprojectionError(C, r, X, b) #scalar
    all_error.append(prev_error)

    for i in range(iter):
        J = np.array([]).reshape((0,7))
        f = []
        for j in range(len(X)):
            Jp, fx = JacobianCR(C, r, q, X[j]) #Jp(2,7), fx(2,)
            J = np.concatenate((J,Jp))
            f = np.append(f,fx)
        #J(2n,7),f(2n,1)

        # 7 x 1
        delta_p = (np.linalg.inv((J.T @ J) + (lamda * np.eye(7))) @ J.T) @ (b - f)
        # p = p + delta_p = [C,R]

        #update C, q, r
        C = C + delta_p[:3]
        q = q + delta_p[3:]
        q = q / np.linalg.norm(q)
        r_from_q = R.from_quat(q)
        r = r_from_q.as_matrix()

        curr_error, error_vec = ReprojectionError(C, r, X, b)
        all_error.append(curr_error)
        
    final_error_vec = error_vec
    print('PnP_nl error', (all_error[-1]-all_error[0])) 


    # plotErrors(all_error, initial_error_vec, final_error_vec, 'CameraPoseRefinement') #

    return r, C




def JacobianX(img1,img2, X):
    """
    Input
    P=[R C], X(3,), K(3,3)
    -
    Output 
    Jp(4,3), fx(4,)
    """
    
    Jx = np.array([]).reshape((0,3))
    fx = []
    for img in [img1,img2]:         
        u,v,w =  (X - img.C) @ img.R.T 
        f = np.array([u/w, v/w])

        # 2 x 3
        du_dX = img.R[0]
        dv_dX = img.R[1]
        dw_dX = img.R[2]
        df_dX = np.array([(w*du_dX - u*dw_dX) / w**2, (w*dv_dX - v*dw_dX) / w**2])

        Jx = np.concatenate((Jx, df_dX))
        fx = np.append(fx, f)

  
    return Jx, fx

def ReprojectionErrorX(img1, img2, X, b):
    f1 = (X - img1.C) @ img1.R.T
    f2 = (X - img2.C) @ img2.R.T
    f1 = f1[:,:2] / f1[:,-1:]
    f2 = f2[:,:2] / f2[:,-1:]
    f = np.hstack((f1,f2))
    b = b.reshape(-1,4)

    error_sum = np.linalg.norm((f-b))
    error_vec = np.square((np.linalg.norm((f - b), axis=1)))

    return error_sum, error_vec

def EvaluateCheirality(imga, imgb, X):
    deta = (X - imga.C) @ (imga.R[-1]).T
    detb = (X - imgb.C) @ (imgb.R[-1]).T
    mask = np.logical_and(deta > 0, detb > 0)
    nValid = np.sum(mask)

    print('# of points before/after filtering out by cheirality is: ', len(X), nValid)

    return mask

def Triangulation_nl(img1, img2, X, x, iter, thr):
    lamda = 5 #adjust
    
    newX = copy.deepcopy(X)
    b = x.reshape(-1) #shape(4n,)
    all_error = []
    prev_error, initial_error_vec = ReprojectionErrorX(img1, img2, newX, b) 
    all_error.append(prev_error)

    for i in range(iter):
        for j in range(len(newX)):
            J, f = JacobianX(img1, img2, newX[j])
            #J(4,3), f(4,)
        
            delta_x = (np.linalg.inv((J.T @ J) + (lamda * np.eye(3))) @ J.T) @ (b[4*j:4*(j+1)] - f)
            newX[j] += delta_x

        curr_error, error_vec = ReprojectionErrorX(img1, img2, newX, b)
        all_error.append(curr_error)

    mask_err = error_vec < thr
    final_error_vec = error_vec[mask_err]
    print('Tri_nl error', (all_error[-1]-all_error[0])) 

 
    # plotErrors(all_error, initial_error_vec, final_error_vec, 'PointRefinement') #

    mask_cheirality = EvaluateCheirality(img1, img2, newX)
    mask = np.logical_and(mask_err, mask_cheirality)
    validX = newX[mask]
    
    return validX, mask
  






def plotErrors(all_error, initial_error_vec, final_error_vec, txt):
    ax1 = plt.subplot(1,3,1)
    ax1.plot(list(range(len(all_error))), all_error)
    ax1.set_title('sum of geometric errors at each iteration', fontsize=10)

    t1 = 'error of each point BEFORE ' + txt
    ax2 = plt.subplot(1,3,2)
    ax2.scatter(list(range(len(initial_error_vec))), initial_error_vec)
    ax2.set_title(t1, fontsize=10)

    t2 = 'error of each point AFTER ' + txt
    ax3 = plt.subplot(1,3,3, sharey=ax2)
    ax3.scatter(list(range(len(final_error_vec))), final_error_vec)
    ax3.set_title(t2, fontsize=10)

    plt.show()
    plt.clf()