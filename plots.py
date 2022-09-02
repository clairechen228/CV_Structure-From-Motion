import cv2
from Correspondance_serach import *

save = False
def drawkp(img,save):
    #Check Class: SIFT
    RGB_img = cv2.cvtColor(img.img, cv2.COLOR_BGR2RGB)
    RGB_kp_img = cv2.drawKeypoints(RGB_img, img.kps, RGB_img)
    plt.imshow(RGB_kp_img)
    if save:
        plt.savefig('sift_keypoints.png', dpi=1000)
        print('sift_keypoints.png saved')
    plt.show()
    plt.clf()



def drawmatch(K, img1,img2, save):
    #Check function: feature_matching 
    RGB_img1 = cv2.cvtColor(img1.img, cv2.COLOR_BGR2RGB)
    RGB_img2 = cv2.cvtColor(img2.img, cv2.COLOR_BGR2RGB)
    fx, fy, matched_indx, dmatch = feature_matching(img1, img2, 0.7)
    RGB_allimg = cv2.drawMatches(RGB_img1, img1.kps, RGB_img2, img2.kps, dmatch, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(RGB_allimg)
    if save:
        plt.savefig('feature_matching.png', dpi=1000)
        print('feature_matching.png saved')
    plt.show()
    plt.clf()



def drawRANSAC(K, iter, thr_match, thr_ransac, img1, img2, save):
    ## Check functions: RANSAC, estimateE
    fx, fy, matched_indx, _ = feature_matching(img1, img2, thr_match)
    f1, f2 = np.asarray(fx), np.asarray(fy)
    # normalize
    norm_f1 = (np.insert(f1, 2, 1, axis=1) @ np.linalg.inv(K).T)[:,:2]
    norm_f2 = (np.insert(f2, 2, 1, axis=1) @ np.linalg.inv(K).T)[:,:2]
    
    ## essential matrix
    E, inlier_mask = RANSAC(iter, thr_ransac, norm_f1, norm_f2)
    print('# of matched points before/after RANSAC: %d, %d '%(len(norm_f1), len(norm_f1[inlier_mask])))
    F = np.linalg.inv(K.T) @ E @ np.linalg.inv(K)
    epilines(img1, img2, f1[inlier_mask], f2[inlier_mask], F, save, 'epipolar_lines.png' )
    print('my F \n', F)

    ## use cv2 function
    # F2, mask = cv2.findFundamentalMat(fx,fy,cv2.FM_8POINT)
    # epilines(img1, img2, fx[mask.ravel() == 1], fy[mask.ravel() == 1], F2)
    # print('cv2_F \n' ,F2)

    
def drawlines(img1, img2, lines, pts1, pts2):
    r, c, _ = img1.shape
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255,3).tolist())
            
        x0, y0 = map(int, [0, -r[2] / r[1] ])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1] ])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(np.int32(pt1)), 5, color, -1)
        img2 = cv2.circle(img2, tuple(np.int32(pt2)), 5, color, -1)

    return img1, img2


def epilines(imgLeft, imgRight, ptsLeft, ptsRight, F, save, name):
    # Find epilines corresponding to points
    # in right image (second image) and
    # drawing its lines on left image
    RGB_imgLeft = cv2.cvtColor(imgLeft.img, cv2.COLOR_BGR2RGB)
    RGB_imgRight = cv2.cvtColor(imgRight.img, cv2.COLOR_BGR2RGB)
    # RGB_imgLeft = imgLeft.img
    # RGB_imgRight = imgRight.img

    linesLeft = cv2.computeCorrespondEpilines(ptsRight.reshape(-1,1,2), 2, F)
    linesLeft = linesLeft.reshape(-1, 3)
    img5, img6 = drawlines(RGB_imgLeft, RGB_imgRight, linesLeft, ptsLeft, ptsRight)
    
    # Find epilines corresponding to 
    # points in left image (first image) and
    # drawing its lines on right image
    linesRight = cv2.computeCorrespondEpilines(ptsLeft.reshape(-1, 1, 2), 1, F)
    linesRight = linesRight.reshape(-1, 3)
    
    img3, img4 = drawlines(RGB_imgRight, RGB_imgLeft, linesRight, ptsRight, ptsLeft)
    
    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)

    if save:
        plt.savefig(name, dpi=1000)
        print(name+' saved')
    
    plt.show()
    plt.clf()



def checkTwoDx(K, imgs, thr_match, iter, thr_ransac):
    #Check function: TwoDx
    track2d, E1 = TwoDx(K, imgs, thr_match, iter, thr_ransac)
    print('shape of track is', track2d.shape)

    f12_mask = np.logical_and(np.sum(track2d[0], axis=1) != -2, np.sum(track2d[1], axis=1) != -2)
    f1, f2 = track2d[0][f12_mask], track2d[1][f12_mask]
    unnorm_f1 = (np.insert(f1, 2, 1, axis=1) @ K.T)[:,:2]
    unnorm_f2 = (np.insert(f2, 2, 1, axis=1) @ K.T)[:,:2]
    F1 = np.linalg.inv(K.T) @ E1 @ np.linalg.inv(K)
    epilines(imgs[0], imgs[1], unnorm_f1, unnorm_f2, F1, False, None )
    print(F1)



def drawCameras(K, imgs, thr_match, iter, thr_ransac, scale, save):
    ## Check function: camera_pose
    track2d, E1 = TwoDx(K, imgs, thr_match, iter, thr_ransac)
    Cset, Rset = camera_pose(E1)

    #set figures
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    fig_set(ax)

    #planes
    f = 2
    img_size = (3,5)
    grid_size = (8,8)
    xx, yy, Z = create_image_grid(f, img_size, grid_size)
    ax.plot_surface(xx, yy, Z, alpha=0.5, color='k')
    grid = convert_grid_to_homogeneous(xx, yy, Z, grid_size)
    
    #imgi
    for c, r in zip(Cset, Rset):
        # print(c, r)
        # print(np.linalg.det(r))
        cameras(ax, r, c*scale)
        planes(ax, r, c*scale, grid, grid_size)

    if save:
        plt.savefig('cameras.png', dpi=1000)
        print('cameras.png saved')

    plt.show()
    plt.clf()

    return track2d, E1, Rset, Cset, ax


def fig_set(ax):
    # ax.set(xlim=(1*scale,-1*scale), ylim=(1*scale,-1*scale), zlim=(1*scale,-1*scale))
    ax.set_xlabel('X-axis', fontweight ='bold')
    ax.set_ylabel('Y-axis', fontweight ='bold')
    ax.set_zlabel('Z-axis', fontweight ='bold')
    #set imgi-1 as the word origin
    ax.scatter3D(0, 0, 0, color='k', marker=',', s=30)
    ax.plot3D([0,1], [0,0], [0,0], 'r')
    ax.plot3D([0,0], [0,1], [0,0], 'g')
    ax.plot3D([0,0], [0,0], [0,1], 'b')

    
def create_image_grid(f, img_size, grid_size):
    h, w = img_size
    x = np.linspace(-(h//2), h//2, num=grid_size[0])
    y = np.linspace(-(w//2), w//2, num=grid_size[0])
    xx, yy = np.meshgrid(x, y)
    Z = np.ones(shape=grid_size) * f

    return xx, yy, Z


def convert_grid_to_homogeneous(xx, yy, Z, grid_size):
    '''
    Extract coordinates from a grid and convert them to homogeneous coordinates
    '''
    h, w = grid_size
    pi = np.ones(shape=(4, h*w))
    c = 0
    for i in range(h):
        for j in range(w):
            x = xx[i, j]
            y = yy[i, j]
            z = Z[i, j]
            point = np.array([x, y, z])
            pi[:3, c] = point
            c += 1
    return pi


def convert_homogeneous_to_grid(pts, grid_size):
    '''
    Convert a set of homogeneous points to a grid
    '''
    xxt = pts[0, :].reshape(grid_size)
    yyt = pts[1, :].reshape(grid_size)
    Zt = pts[2, :].reshape(grid_size)

    return xxt, yyt, Zt


def planes(ax, r, c, grid, grid_size):
    R = np.eye(4)
    R[:3,:3] = r
    t = np.eye(4)
    t[:3,3] = -r@c
    grid_transformed = R @ t @ grid
    xx, yy, Z = convert_homogeneous_to_grid(grid_transformed, grid_size)
    ax.plot_surface(xx, yy, Z, alpha=0.3)

    
def cameras(ax, r, c):
    color = ['r', 'g', 'b']
    #origin
    ax.scatter3D(c[0], c[1], c[2], color='r', s=30)
    #axis
    for i in range(3):
        unitV = r[:,i] / np.dot(r[:,i], r[:,i]) 
        ax.plot3D([c[0], c[0]+unitV[0]], [c[1], c[1]+unitV[1]], [c[2], c[2]+unitV[2]], color[i])


def draw3Dpoints(K, imgs, thr_match, iter, thr_ransac, save):
    ## Check functions: Triangulation, SkewMatrix, Cheirality
    Ra = np.eye(3)
    ta = np.zeros(3)
    Pa = np.hstack((Ra,ta.reshape(3,1)))
    imgs[0].P = Pa
    imgs[0].R = Ra
    imgs[0].C = ta

    scale = 1
    track2d, E1 = TwoDx(K, imgs, thr_match, iter, thr_ransac)
    Cset, Rset = camera_pose(E1)
    f12_mask = np.logical_and(np.sum(track2d[0], axis=1) != -2, np.sum(track2d[1], axis=1) != -2)
    f1, f2 = track2d[0][f12_mask], track2d[1][f12_mask]

    #fig1 seperate
    fig1 = plt.figure(figsize=plt.figaspect(0.5))
    index = 1
    for c, r in zip(Cset, Rset):
        #set figures
        ax = fig1.add_subplot(2, 2, index, projection='3d')
        fig_set(ax)

        #planes
        f = 2
        img_size = (3,5)
        grid_size = (8,8)
        xx, yy, Z = create_image_grid(f, img_size, grid_size)
        ax.plot_surface(xx, yy, Z, alpha=0.5, color='k')
        grid = convert_grid_to_homogeneous(xx, yy, Z, grid_size)

        cameras(ax, r, c*scale)
        planes(ax, r, c*scale, grid, grid_size)

        Pb = np.hstack((r, (-r @ c.reshape(3,1))))
        X = Triangulation(Pa, Pb, f1, f2)
        ax.scatter3D(X[:,0], X[:,1], X[:,2], s=10)

        index += 1
    
    if save:
        plt.savefig('draw3Dpoints.png', dpi=1000)
        print('draw3Dpoints.png saved')
    plt.show()
    plt.clf()

def drawCheirality(K, imgs, thr_match, iter, thr_ransac, save):
    ## Check function: Cheirality
    Ra = np.eye(3)
    ta = np.zeros(3)
    Pa = np.hstack((Ra,ta.reshape(3,1)))
    imgs[0].P = Pa
    imgs[0].R = Ra
    imgs[0].C = ta

    scale = 1
    track2d, E1 = TwoDx(K, imgs, thr_match, iter, thr_ransac)
    Cset, Rset = camera_pose(E1)
    f12_mask = np.logical_and(np.sum(track2d[0], axis=1) != -2, np.sum(track2d[1], axis=1) != -2)
    f1, f2 = track2d[0][f12_mask], track2d[1][f12_mask]
    X, mask = Cheirality(Cset, Rset, f1, f2, imgs[0], imgs[1])

    #set figures
    fig2 = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig2.add_subplot(1, 1, 1, projection='3d')
    # ax.set(xlim=(1*scale,-1*scale), ylim=(1*scale,-1*scale), zlim=(1*scale,-1*scale))
    ax.set_xlabel('X-axis', fontweight ='bold')
    ax.set_ylabel('Y-axis', fontweight ='bold')
    ax.set_zlabel('Z-axis', fontweight ='bold')
    #set imgi-1 as the word origin
    ax.scatter3D(0, 0, 0, color='k', marker=',', s=30)
    ax.plot3D([0,1], [0,0], [0,0], 'r')
    ax.plot3D([0,0], [0,1], [0,0], 'g')
    ax.plot3D([0,0], [0,0], [0,1], 'b')

    #planes
    f = 2
    img_size = (3,5)
    grid_size = (8,8)
    xx, yy, Z = create_image_grid(f, img_size, grid_size)
    ax.plot_surface(xx, yy, Z, alpha=0.5, color='k')
    grid = convert_grid_to_homogeneous(xx, yy, Z, grid_size)
    
    #imgi
    for c, r in zip(Cset, Rset):
        # print(c, r)
        # print(np.linalg.det(r))
        cameras(ax, r, c*scale)
        planes(ax, r, c*scale, grid, grid_size)
    ax.scatter3D(X[:,0], X[:,1], X[:,2], color='k', s=10)

    if save:
        plt.savefig('drawCheirality.png', dpi=1000)
        print('drawCheirality.png saved')
    plt.show()
    plt.clf()



def plotTriangulation(l_X, nl_X, img1, K, track2d):
    rgb_img = cv2.cvtColor(img1.img, cv2.COLOR_BGR2RGB)
    ax = plt.subplot(111)
    plt.imshow(rgb_img)

    uvw = (l_X - img1.C) @ img1.R.T 
    f = uvw[:,:2] / uvw[:,-1:]
    # l_x = (np.insert(f, 2, 1, axis=1) @ K.T)[:,:2]
    l_x = (np.insert(f, 2, 1, axis=1) @ K.T)
    l_x = l_x[:,:2] / l_x[:,-1:]

    uvw = (nl_X - img1.C) @ img1.R.T 
    f = uvw[:,:2] / uvw[:,-1:]
    # nl_x = (np.insert(f, 2, 1, axis=1) @ K.T)[:,:2]
    nl_x = (np.insert(f, 2, 1, axis=1) @ K.T)
    nl_x = nl_x[:,:2] / nl_x[:,-1:]

    # x = (np.insert(track2d, 2, 1, axis=1) @ K.T)[:,:2]
    x = (np.insert(track2d, 2, 1, axis=1) @ K.T)
    x = x[:,:2] / x[:,-1:]
    
    ax.scatter(x[:,0], x[:,1], marker='o', color='y')
    ax.scatter(l_x[:,0], l_x[:,1], marker='x', color='k')
    ax.scatter(nl_x[:,0], nl_x[:,1], marker='.', color='r')

    plt.show()

#---------------------------------------------------------------------------#
def show3dcameras(imgs):
    ax = plt.subplot(1, 1, 1, projection='3d')
    fig_set(ax)
    ax.set(xlim=(-5,5), ylim=(-5,5), zlim=(-5,5))

    for i in range(len(imgs)):
        label = "cam" + str(i)
        cameras(ax, imgs[i].R, imgs[i].C)
        ax.text(imgs[i].C[0], imgs[i].C[1], imgs[i].C[2], label)
   
    plt.show()
    plt.clf()

def show3dpoints(track3d):
    mask = np.sum(track3d, axis=1) != -3
    X = track3d[mask]
    print(len(X), ' of 3d points')
    # inlier = np.logical_and(np.logical_and(X[:,1] > -2000, X[:,0] > -4000), X[:,2] <5000)
    # X = X[inlier]


    ax = plt.subplot(1, 1, 1, projection='3d')
    ax.set_xlabel('X-axis', fontweight ='bold')
    ax.set_ylabel('Y-axis', fontweight ='bold')
    ax.set_zlabel('Z-axis', fontweight ='bold')
    ax.scatter3D(X[:,0], X[:,1], X[:,2], s=10)
    plt.show()

