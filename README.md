# Camera Calibration using Zhang's Method

This project implements Zhang's method for camera calibration. Camera calibration is essential in computer vision to obtain the intrinsic and extrinsic parameters of a camera, which are used to correct image distortions and understand the 3D geometry of the scene.

## Theory

### Zhang's Camera Calibration Method

Zhang's method is a well-known technique for camera calibration using a planar calibration object, such as a chessboard pattern. The method involves capturing multiple images of the calibration object at different orientations and positions. The main steps are:

1. **Image and World Points Extraction**:
   - Detect and extract 2D image points (corners of the chessboard pattern) from the calibration images.
   - Generate corresponding 3D world points based on the known dimensions of the chessboard.

2. **Homography Computation**:
   - Compute homographies between the world points and the image points for each calibration image.

3. **Intrinsic Parameters Calculation**:
   - Construct a system of linear equations using the homographies to solve for the intrinsic parameters of the camera.

4. **Extrinsic Parameters Calculation**:
   - Using the intrinsic parameters, compute the extrinsic parameters (rotation and translation) for each image.

5. **Distortion Coefficients**:
   - Estimate distortion coefficients to account for lens distortion.

6. **Optimization**:
   - Optimize the intrinsic, extrinsic parameters, and distortion coefficients to minimize the reprojection error.

## Prerequisites

To run the code, you need to have the following libraries installed:

- `numpy`
- `opencv-python`
- `matplotlib`
- `scipy`

You can install these libraries using pip:

```sh
pip install numpy opencv-python matplotlib scipy
```

## To run the code (calibrate your camera with custom dataset)         
Put all of your images in the "Calibration_Imgs" folder then run the Wrapper file.     

```sh
python3 Wrapper.py
```    

** Results **    
![Result 1](https://github.com/devsonni/CameraCalibration/blob/main/Results/result0.png)
![Result 2](https://github.com/devsonni/CameraCalibration/blob/main/Results/result1.png)
![Result 3](https://github.com/devsonni/CameraCalibration/blob/main/Results/result2.png)
![Result 4](https://github.com/devsonni/CameraCalibration/blob/main/Results/result3.png)
