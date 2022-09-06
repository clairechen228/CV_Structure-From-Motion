# Structure from Motion (SfM)
- Reconstruct the 3-dimensional structure of a scene and estimate the camera poses from a set of 2-dimensional images.
- A comparison of different bundle adjustment methods.

### Input Data
<p align="center" width="100%">
  <img width=16% src="/raw_imgs/image0000001.jpg ">
  <img width=16% src="/raw_imgs/image0000002.jpg ">
  <img width=16% src="/raw_imgs/image0000003.jpg ">
  <img width=16% src="/raw_imgs/image0000004.jpg ">
  <img width=16% src="/raw_imgs/image0000005.jpg ">
  <img width=16% src="/raw_imgs/image0000006.jpg ">
  <br>(The dataset is from UMN Spring 2021 CSCI-5563 course material)
</p>

### Methods
#### Correspondance_serach.py
  1. Feature Matching: Matched SIFT keypoints and descriptors and rejected outliers using RANSAC
  <p align="center" width="100%">
    <img width=45% src="/results/figures/sift_keypoints.png">
    <img width=54% src="/results/figures/feature_matching.png">
  </p>
  
  2. Essential Matrix: Estimated essential matrix between first two images
  <p align="center" width="100%">
     <img align="center" width="70%" src="/results/figures/epipolar_lines.png">
  </p>
  
  3. Camera Pose: Estimated four camera configulations and verified by checking cheirality conditions
  <p align="center" width="100%">
     <img align="center" width="30%" src="/results/figures/camera_configurations.png">
     <img align="center" width="69%" src="/results/figures/triangulations.png">
  </p>

#### PnP.py
   4. Perspective-n-Point: Estimated camera poses using linera PnP and refined the poses using nonlinear PnP
   5. 3D Points: Constructed missing 3D points using Triangulation and refined the points using nonlinear Triangulation 
   <p align="center" width="100%">
     <img align="center" width="49%" src="/results/figures/CameraPoseRefinement.png">
     <img align="center" width="49%" src="/results/figures/PointRefinement.png">
   </p>
   
#### BA_scipy.py
   6-1. Bundle adjustment(scipy): Minimizing the reprojection error using scipy.optimize.least_squares function
#### BA.py
   6-2. Bundle adjustment(numpy): Minimizing the reprojection error using Levenberg–Marquardt algorithm
#### BA_jax.py
   6-3. Bundle adjustment(JAX): Same structure as Bundle adjustment(numpy) but automatically differentiate Jacobian matrix using Jax.
  
  
### Results
#### Visualization
<p align="center" width="100%">
  <img width=29% src="/results/figures/cameras.png">
  <img width=70% src="/results/scipy/isometric_view.PNG"><br>
  camera poses(left) / isometric view(right)
</p>
<p align="center" width="100%">
  <img width=49% src="/results/scipy/top_view.PNG" >
  <img width=49% src="/results/scipy/front_view.PNG"><br>
  top viw(left) / front view(right)
</p>

#### Runtime Comparision

|               |     BA_scipy  |    BA         |     BA_jax    |
| ------------- | ------------- | ------------- | ------------- |
|    runtime*   |     2.2s      |     270s      |     300s      |
|error reduction|     -2.4      |     -4.5      |     -4.2      |

*Runtime of 1 iteration with approximate 2000 points
