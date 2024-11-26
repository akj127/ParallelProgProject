# CUDA Accelerated Canny Edge Detection

## Overview
This repository implements a CUDA-accelerated version of the Canny Edge Detection algorithm for real-time edge detection on images. The project leverages the power of GPUs for parallel processing and uses OpenCV for image input/output and preprocessing.

---

## Features
- **Gaussian Blur**: Smooths the image to reduce noise.
- **Gradient Computation**: Calculates image gradients using Sobel filters.
- **Non-Maximum Suppression**: Thins the edges to a single pixel width.
- **Double Thresholding**: Distinguishes between strong, weak, and irrelevant edges.
- **Edge Tracking by Hysteresis**: Finalizes edges by connecting weak edges to strong edges.

---

## How It Works
The implementation is divided into the following stages, each executed on the GPU:

1. **Gaussian Blur**
   - Smooths the input image to reduce noise.
   - Applies a Gaussian filter using a convolution operation.

2. **Gradient Computation**
   - Computes the intensity gradients in the x and y directions using Sobel filters.
   - Calculates the gradient magnitude and direction.

3. **Non-Maximum Suppression**
   - Suppresses non-maximum gradient magnitudes to thin the edges.

4. **Double Thresholding**
   - Classifies pixels into strong, weak, or suppressed edges based on thresholds.

5. **Edge Tracking by Hysteresis**
   - Connects weak edges to strong edges if they are part of the same edge path.

---

## Pseudo Code for CUDA Kernels

### Gaussian Blur Kernel
```plaintext
Input: d_image (input image), d_blurred (output image), width, height
For each pixel in the image:
    Apply a Gaussian filter in a sliding window
    Save the result to d_blurred

### Sobel filter
Input: d_blurred (blurred image), d_gradient, d_direction, width, height
For each pixel in the image:
    Compute gradient in the x-direction using Sobel filter
    Compute gradient in the y-direction using Sobel filter
    Calculate gradient magnitude and direction
    Save results to d_gradient and d_direction

### Non maximum suppression kernel
Input: d_gradient, d_direction, d_edges, width, height
For each pixel in the image:
    Check the neighboring pixels in the direction of the gradient
    Suppress the pixel if it is not a local maximum
    Save the result to d_edges

### Double Threshold Kernel
Input: d_edges, highThreshold, lowThreshold, width, height
For each pixel in the image:
    Classify the pixel:
        Strong edge if > highThreshold
        Weak edge if between lowThreshold and highThreshold
        Suppressed edge otherwise
    Save the updated edge map to d_edges

###Hystersis Kernel
Input: d_edges, width, height
For each pixel in the image:
    If the pixel is a weak edge:
        Check if it is connected to a strong edge
        Promote to strong edge if connected
    Save the final edge map to d_edges


TO run the code
nvcc -o canny_edge_detection main.cu -lopencv_core -lopencv_imgproc -lopencv_highgui

./canny_edge_detection <input_image>
