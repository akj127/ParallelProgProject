#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(call)                                                      \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__      \
                      << " - " << cudaGetErrorString(err) << std::endl;       \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

// Constants
#define GAUSSIAN_KERNEL_RADIUS 2
#define GAUSSIAN_KERNEL_SIZE (2 * GAUSSIAN_KERNEL_RADIUS + 1)
#define BLOCK_SIZE 16

// Gaussian Kernel Coefficients (1D)
__constant__ float c_GaussianKernel[GAUSSIAN_KERNEL_SIZE] = {
    1 / 273.0f, 7 / 273.0f, 41 / 273.0f, 7 / 273.0f, 1 / 273.0f
};

// CUDA Kernel Definitions

// Horizontal Gaussian Blur Kernel
__global__ void GaussianBlurHorizontalKernel(float* d_input, float* d_output, int width, int height) {
    extern __shared__ float shared_mem[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x + GAUSSIAN_KERNEL_RADIUS;

    if (y >= height) return;

    // Load data into shared memory with halo
    for (int i = -GAUSSIAN_KERNEL_RADIUS; i <= GAUSSIAN_KERNEL_RADIUS; ++i) {
        int idx = min(max(x + i, 0), width - 1);
        shared_mem[tx + i] = d_input[y * width + idx];
    }
    __syncthreads();

    if (x < width) {
        float sum = 0.0f;
        for (int k = -GAUSSIAN_KERNEL_RADIUS; k <= GAUSSIAN_KERNEL_RADIUS; ++k) {
            sum += c_GaussianKernel[k + GAUSSIAN_KERNEL_RADIUS] * shared_mem[tx + k];
        }
        d_output[y * width + x] = sum;
    }
}

// Vertical Gaussian Blur Kernel
__global__ void GaussianBlurVerticalKernel(float* d_input, float* d_output, int width, int height) {
    extern __shared__ float shared_mem[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int ty = threadIdx.y + GAUSSIAN_KERNEL_RADIUS;

    if (x >= width) return;

    // Load data into shared memory with halo
    for (int i = -GAUSSIAN_KERNEL_RADIUS; i <= GAUSSIAN_KERNEL_RADIUS; ++i) {
        int idx = min(max(y + i, 0), height - 1);
        shared_mem[ty + i] = d_input[idx * width + x];
    }
    __syncthreads();

    if (y < height) {
        float sum = 0.0f;
        for (int k = -GAUSSIAN_KERNEL_RADIUS; k <= GAUSSIAN_KERNEL_RADIUS; ++k) {
            sum += c_GaussianKernel[k + GAUSSIAN_KERNEL_RADIUS] * shared_mem[ty + k];
        }
        d_output[y * width + x] = sum;
    }
}

// Optimized Sobel Kernel with Shared Memory
__global__ void SobelSharedKernel(float* d_input, float* d_gradient, float* d_direction, int width, int height) {
    extern __shared__ float shared_mem[];

    int shared_width = BLOCK_SIZE + 2;
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // Load data into shared memory with halo
    if (x < width && y < height) {
        shared_mem[ty * shared_width + tx] = d_input[y * width + x];
    } else {
        shared_mem[ty * shared_width + tx] = 0.0f;
    }

    // Load halo regions
    if (threadIdx.x == 0) {
        int x_left = x - 1;
        if (x_left >= 0 && y < height)
            shared_mem[ty * shared_width + tx - 1] = d_input[y * width + x_left];
        else
            shared_mem[ty * shared_width + tx - 1] = 0.0f;
    }
    if (threadIdx.x == BLOCK_SIZE - 1) {
        int x_right = x + 1;
        if (x_right < width && y < height)
            shared_mem[ty * shared_width + tx + 1] = d_input[y * width + x_right];
        else
            shared_mem[ty * shared_width + tx + 1] = 0.0f;
    }
    if (threadIdx.y == 0) {
        int y_top = y - 1;
        if (x < width && y_top >= 0)
            shared_mem[(ty - 1) * shared_width + tx] = d_input[y_top * width + x];
        else
            shared_mem[(ty - 1) * shared_width + tx] = 0.0f;
    }
    if (threadIdx.y == BLOCK_SIZE - 1) {
        int y_bottom = y + 1;
        if (x < width && y_bottom < height)
            shared_mem[(ty + 1) * shared_width + tx] = d_input[y_bottom * width + x];
        else
            shared_mem[(ty + 1) * shared_width + tx] = 0.0f;
    }

    // Corners
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int x_left = x - 1;
        int y_top = y - 1;
        if (x_left >= 0 && y_top >= 0)
            shared_mem[(ty - 1) * shared_width + tx - 1] = d_input[y_top * width + x_left];
        else
            shared_mem[(ty - 1) * shared_width + tx - 1] = 0.0f;
    }
    if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == 0) {
        int x_right = x + 1;
        int y_top = y - 1;
        if (x_right < width && y_top >= 0)
            shared_mem[(ty - 1) * shared_width + tx + 1] = d_input[y_top * width + x_right];
        else
            shared_mem[(ty - 1) * shared_width + tx + 1] = 0.0f;
    }
    if (threadIdx.x == 0 && threadIdx.y == BLOCK_SIZE - 1) {
        int x_left = x - 1;
        int y_bottom = y + 1;
        if (x_left >= 0 && y_bottom < height)
            shared_mem[(ty + 1) * shared_width + tx - 1] = d_input[y_bottom * width + x_left];
        else
            shared_mem[(ty + 1) * shared_width + tx - 1] = 0.0f;
    }
    if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == BLOCK_SIZE - 1) {
        int x_right = x + 1;
        int y_bottom = y + 1;
        if (x_right < width && y_bottom < height)
            shared_mem[(ty + 1) * shared_width + tx + 1] = d_input[y_bottom * width + x_right];
        else
            shared_mem[(ty + 1) * shared_width + tx + 1] = 0.0f;
    }
    __syncthreads();

    if (x < width && y < height) {
        float Gx = 0.0f;
        float Gy = 0.0f;

        // Compute Gx and Gy using shared memory
        Gx += -1 * shared_mem[(ty - 1) * shared_width + tx - 1];
        Gx += -2 * shared_mem[ty * shared_width + tx - 1];
        Gx += -1 * shared_mem[(ty + 1) * shared_width + tx - 1];
        Gx +=  1 * shared_mem[(ty - 1) * shared_width + tx + 1];
        Gx +=  2 * shared_mem[ty * shared_width + tx + 1];
        Gx +=  1 * shared_mem[(ty + 1) * shared_width + tx + 1];

        Gy += -1 * shared_mem[(ty - 1) * shared_width + tx - 1];
        Gy += -2 * shared_mem[(ty - 1) * shared_width + tx];
        Gy += -1 * shared_mem[(ty - 1) * shared_width + tx + 1];
        Gy +=  1 * shared_mem[(ty + 1) * shared_width + tx - 1];
        Gy +=  2 * shared_mem[(ty + 1) * shared_width + tx];
        Gy +=  1 * shared_mem[(ty + 1) * shared_width + tx + 1];

        float magnitude = sqrtf(Gx * Gx + Gy * Gy);
        float angle = atan2f(Gy, Gx) * (180.0f / M_PI);
        if (angle < 0) angle += 180.0f;

        d_gradient[y * width + x] = magnitude;
        d_direction[y * width + x] = angle;
    }
}

// Normalize Gradient Kernel
__global__ void NormalizeGradientKernel(float* d_gradient, float maxGradient, int width, int height) {
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (x >= width || y >= height) return;

    d_gradient[y * width + x] /= maxGradient;
}

// Non-Maximum Suppression Kernel
__global__ void NonMaxSuppressionKernel(float* d_gradient, float* d_direction, float* d_output, int width, int height) {
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (x >= width || y >= height) return;

    float angle = d_direction[y * width + x];
    float magnitude = d_gradient[y * width + x];
    float neighbor1 = 0.0f, neighbor2 = 0.0f;

    // Determine the orientation
    if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
        // 0 degrees
        if (x > 0) neighbor1 = d_gradient[y * width + x - 1];
        if (x < width - 1) neighbor2 = d_gradient[y * width + x + 1];
    } else if (angle >= 22.5 && angle < 67.5) {
        // 45 degrees
        if (x > 0 && y < height - 1) neighbor1 = d_gradient[(y + 1) * width + x - 1];
        if (x < width - 1 && y > 0) neighbor2 = d_gradient[(y - 1) * width + x + 1];
    } else if (angle >= 67.5 && angle < 112.5) {
        // 90 degrees
        if (y > 0) neighbor1 = d_gradient[(y - 1) * width + x];
        if (y < height - 1) neighbor2 = d_gradient[(y + 1) * width + x];
    } else if (angle >= 112.5 && angle < 157.5) {
        // 135 degrees
        if (x > 0 && y > 0) neighbor1 = d_gradient[(y - 1) * width + x - 1];
        if (x < width - 1 && y < height - 1) neighbor2 = d_gradient[(y + 1) * width + x + 1];
    }

    if (magnitude >= neighbor1 && magnitude >= neighbor2) {
        d_output[y * width + x] = magnitude;
    } else {
        d_output[y * width + x] = 0.0f;
    }
}

// Double Thresholding Kernel
__global__ void DoubleThresholdKernel(float* d_edges, float highThreshold, float lowThreshold, int width, int height) {
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (x >= width || y >= height) return;

    float value = d_edges[y * width + x];
    if (value >= highThreshold) {
        d_edges[y * width + x] = 1.0f; // Strong edge
    } else if (value >= lowThreshold) {
        d_edges[y * width + x] = 0.5f; // Weak edge
    } else {
        d_edges[y * width + x] = 0.0f; // Non-edge
    }
}

// Hysteresis Kernel (Iterative)
__global__ void HysteresisKernel(float* d_edges, int width, int height, bool* d_changed) {
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (x >= width || y >= height) return;

    if (d_edges[y * width + x] != 0.5f) return; // Process only weak edges

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;

            int neighborX = x + dx;
            int neighborY = y + dy;

            if (neighborX >= 0 && neighborX < width && neighborY >= 0 && neighborY < height) {
                if (d_edges[neighborY * width + neighborX] == 1.0f) {
                    d_edges[y * width + x] = 1.0f;
                    *d_changed = true;
                    return;
                }
            }
        }
    }

    d_edges[y * width + x] = 0.0f;
}

// Host Function Declarations
void processSingleScale(float* d_image, float* d_output, float* d_temp, float* d_gradient, float* d_direction, int width, int height);
void combineEdgeMaps(const std::vector<cv::Mat>& edgeMaps, cv::Mat& output);

// Multi-Scale Canny Implementation
void multiScaleCanny(const cv::Mat& inputImage, const std::vector<float>& scales, cv::Mat& outputEdges) {
    int originalWidth = inputImage.cols;
    int originalHeight = inputImage.rows;

    std::vector<cv::Mat> edgeMaps;

    // Allocate device memory once
    float *d_image, *d_blurred, *d_temp, *d_gradient, *d_direction, *d_edges;
    size_t maxWidth = static_cast<size_t>(originalWidth * (*std::max_element(scales.begin(), scales.end())));
    size_t maxHeight = static_cast<size_t>(originalHeight * (*std::max_element(scales.begin(), scales.end())));
    size_t maxSize = maxWidth * maxHeight * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_image, maxSize));
    CUDA_CHECK(cudaMalloc(&d_blurred, maxSize));
    CUDA_CHECK(cudaMalloc(&d_temp, maxSize));
    CUDA_CHECK(cudaMalloc(&d_gradient, maxSize));
    CUDA_CHECK(cudaMalloc(&d_direction, maxSize));
    CUDA_CHECK(cudaMalloc(&d_edges, maxSize));

    for (float scale : scales) {
        // Resize the image for the current scale
        int scaledWidth = static_cast<int>(originalWidth * scale);
        int scaledHeight = static_cast<int>(originalHeight * scale);
        cv::Mat resizedImage;
        cv::resize(inputImage, resizedImage, cv::Size(scaledWidth, scaledHeight), 0, 0, cv::INTER_LINEAR);

        // Copy the resized image to device memory
        std::vector<float> h_resizedImage(scaledWidth * scaledHeight);
        for (int i = 0; i < resizedImage.rows; ++i) {
            for (int j = 0; j < resizedImage.cols; ++j) {
                h_resizedImage[i * scaledWidth + j] = resizedImage.at<uchar>(i, j) / 255.0f;
            }
        }

        CUDA_CHECK(cudaMemcpy(d_image, h_resizedImage.data(), scaledWidth * scaledHeight * sizeof(float), cudaMemcpyHostToDevice));

        // Process single scale
        processSingleScale(d_image, d_edges, d_temp, d_gradient, d_direction, scaledWidth, scaledHeight);

        // Copy the edge map back to host memory
        std::vector<float> h_edges(scaledWidth * scaledHeight);
        CUDA_CHECK(cudaMemcpy(h_edges.data(), d_edges, scaledWidth * scaledHeight * sizeof(float), cudaMemcpyDeviceToHost));

        cv::Mat edgeMap(scaledHeight, scaledWidth, CV_32F, h_edges.data());
        cv::resize(edgeMap, edgeMap, cv::Size(originalWidth, originalHeight), 0, 0, cv::INTER_LINEAR);

        edgeMaps.push_back(edgeMap.clone());
    }

    // Combine edge maps from all scales
    combineEdgeMaps(edgeMaps, outputEdges);

    // Free device memory
    CUDA_CHECK(cudaFree(d_image));
    CUDA_CHECK(cudaFree(d_blurred));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_gradient));
    CUDA_CHECK(cudaFree(d_direction));
    CUDA_CHECK(cudaFree(d_edges));
}

void processSingleScale(float* d_image, float* d_output, float* d_temp, float* d_gradient, float* d_direction, int width, int height) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Shared memory sizes for Gaussian blur kernels
    size_t sharedMemSizeGaussian = (BLOCK_SIZE + 2 * GAUSSIAN_KERNEL_RADIUS) * sizeof(float);
    // Shared memory size for Sobel kernel
    size_t sharedMemSizeSobel = (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2) * sizeof(float);

    // Gaussian Blur - Horizontal Pass
    GaussianBlurHorizontalKernel<<<gridSize, blockSize, sharedMemSizeGaussian>>>(d_image, d_temp, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Gaussian Blur - Vertical Pass
    GaussianBlurVerticalKernel<<<gridSize, blockSize, sharedMemSizeGaussian>>>(d_temp, d_output, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Gradient Computation
    SobelSharedKernel<<<gridSize, blockSize, sharedMemSizeSobel>>>(d_output, d_gradient, d_direction, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy gradient to host to compute max gradient value
    std::vector<float> h_gradient(width * height);
    CUDA_CHECK(cudaMemcpy(h_gradient.data(), d_gradient, width * height * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute maximum gradient magnitude
    float maxGradient = *std::max_element(h_gradient.begin(), h_gradient.end());

    // Normalize gradient magnitudes
    NormalizeGradientKernel<<<gridSize, blockSize>>>(d_gradient, maxGradient, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Non-Maximum Suppression
    NonMaxSuppressionKernel<<<gridSize, blockSize>>>(d_gradient, d_direction, d_output, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy non-max suppressed edges to host to compute thresholds
    std::vector<float> h_suppressed(width * height);
    CUDA_CHECK(cudaMemcpy(h_suppressed.data(), d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute thresholds
    std::vector<float> nonZeroGradients;
    for (float val : h_suppressed) {
        if (val > 0) nonZeroGradients.push_back(val);
    }
    std::sort(nonZeroGradients.begin(), nonZeroGradients.end());

    float highThreshold = 0.0f;
    float lowThreshold = 0.0f;

    if (!nonZeroGradients.empty()) {
        highThreshold = nonZeroGradients[static_cast<int>(0.9 * nonZeroGradients.size())];
        lowThreshold = highThreshold * 0.5f;
    }

    // Double Thresholding
    DoubleThresholdKernel<<<gridSize, blockSize>>>(d_output, highThreshold, lowThreshold, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Edge Tracking by Hysteresis (Iterative)
    bool *d_changed;
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(bool)));

    bool h_changed = true;
    while (h_changed) {
        h_changed = false;
        CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));

        HysteresisKernel<<<gridSize, blockSize>>>(d_output, width, height, d_changed);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(d_changed));
}

void combineEdgeMaps(const std::vector<cv::Mat>& edgeMaps, cv::Mat& output) {
    output = cv::Mat::zeros(edgeMaps[0].size(), CV_32F);

    for (const cv::Mat& edgeMap : edgeMaps) {
        output += edgeMap;
    }

    // Normalize the combined edge map
    double minVal, maxVal;
    cv::minMaxLoc(output, &minVal, &maxVal);
    if (maxVal > 0) {
        output.convertTo(output, CV_32F, 1.0 / maxVal);
    }

    // Apply threshold to get binary edge map
    cv::threshold(output, output, 0.5, 1.0, cv::THRESH_BINARY);
    output.convertTo(output, CV_8U, 255.0);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_file>" << std::endl;
        return -1;
    }

    const char* inputImageFile = argv[1];
    cv::Mat inputImage = cv::imread(inputImageFile, cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }

    // Define scales
    std::vector<float> scales = {0.5f, 1.0f, 1.5f, 2.0f};

    // Perform multi-scale Canny edge detection
    cv::Mat outputEdges;
    multiScaleCanny(inputImage, scales, outputEdges);

    // Save the output
    cv::imwrite("multi_scale_edges.png", outputEdges);

    return 0;
}
