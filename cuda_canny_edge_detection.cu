#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

// Helper macros for CUDA error checking
#define CUDA_CHECK(call)                                                      \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__      \
                      << " - " << cudaGetErrorString(err) << std::endl;       \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

// Host Function Declarations
void loadAndPreprocessImage(const char* filename, float** h_image, int* width, int* height);
void applyGaussianBlur(float* d_image, float* d_blurred, int width, int height);
void computeGradients(float* d_blurred, float* d_gradient, float* d_direction, int width, int height);
void performNonMaxSuppression(float* d_gradient, float* d_direction, float* d_edges, int width, int height);
void applyDoubleThresholding(float* d_edges, float highThreshold, float lowThreshold, int width, int height);
void edgeTrackingByHysteresis(float* d_edges, int width, int height);
void saveOutputImage(const char* filename, float* d_edges, int width, int height);

// CUDA Kernel Declarations
__global__ void GaussianBlurKernel(float* d_image, float* d_blurred, int width, int height);
__global__ void SobelKernel(float* d_blurred, float* d_gradient, float* d_direction, int width, int height);
__global__ void NonMaxSuppressionKernel(float* d_gradient, float* d_direction, float* d_edges, int width, int height);
__global__ void DoubleThresholdKernel(float* d_edges, float highThreshold, float lowThreshold, int width, int height);
__global__ void HysteresisKernel(float* d_edges, int width, int height);

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_file>" << std::endl;
        return -1;
    }

    const char* inputImage = argv[1];
    const char* outputImage = "output_edges.png";

    // Host and Device variables
    float* h_image = nullptr;
    int width, height;

    float *d_image, *d_blurred, *d_gradient, *d_direction, *d_edges;

    // Load and preprocess the input image
    loadAndPreprocessImage(inputImage, &h_image, &width, &height);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_image, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_blurred, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gradient, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_direction, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_edges, width * height * sizeof(float)));

    // Copy input image to device memory
    CUDA_CHECK(cudaMemcpy(d_image, h_image, width * height * sizeof(float), cudaMemcpyHostToDevice));

    // Perform each stage of Canny Edge Detection
    applyGaussianBlur(d_image, d_blurred, width, height);
    computeGradients(d_blurred, d_gradient, d_direction, width, height);
    performNonMaxSuppression(d_gradient, d_direction, d_edges, width, height);
    applyDoubleThresholding(d_edges, 0.2f, 0.1f, width, height);
    edgeTrackingByHysteresis(d_edges, width, height);

    // Save the final output image
    saveOutputImage(outputImage, d_edges, width, height);

    // Free device memory
    CUDA_CHECK(cudaFree(d_image));
    CUDA_CHECK(cudaFree(d_blurred));
    CUDA_CHECK(cudaFree(d_gradient));
    CUDA_CHECK(cudaFree(d_direction));
    CUDA_CHECK(cudaFree(d_edges));

    // Free host memory
    free(h_image);

    return 0;
}

void loadAndPreprocessImage(const char* filename, float** h_image, int* width, int* height) {
    cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Could not load image." << std::endl;
        exit(EXIT_FAILURE);
    }

    *width = img.cols;
    *height = img.rows;
    *h_image = (float*)malloc((*width) * (*height) * sizeof(float));

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            (*h_image)[i * (*width) + j] = img.at<uchar>(i, j) / 255.0f;
        }
    }
}

void saveOutputImage(const char* filename, float* d_edges, int width, int height) {
    float* h_edges = (float*)malloc(width * height * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_edges, d_edges, width * height * sizeof(float), cudaMemcpyDeviceToHost));

    cv::Mat output(height, width, CV_32F, h_edges);
    output.convertTo(output, CV_8U, 255.0);
    cv::imwrite(filename, output);

    free(h_edges);
}

void applyGaussianBlur(float* d_image, float* d_blurred, int width, int height) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    GaussianBlurKernel<<<gridSize, blockSize>>>(d_image, d_blurred, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void computeGradients(float* d_blurred, float* d_gradient, float* d_direction, int width, int height) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    SobelKernel<<<gridSize, blockSize>>>(d_blurred, d_gradient, d_direction, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void performNonMaxSuppression(float* d_gradient, float* d_direction, float* d_edges, int width, int height) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    NonMaxSuppressionKernel<<<gridSize, blockSize>>>(d_gradient, d_direction, d_edges, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void applyDoubleThresholding(float* d_edges, float highThreshold, float lowThreshold, int width, int height) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    DoubleThresholdKernel<<<gridSize, blockSize>>>(d_edges, highThreshold, lowThreshold, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void edgeTrackingByHysteresis(float* d_edges, int width, int height) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    HysteresisKernel<<<gridSize, blockSize>>>(d_edges, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void GaussianBlurKernel(float* d_image, float* d_blurred, int width, int height) {
    // Implement Gaussian blur logic
}

__global__ void SobelKernel(float* d_blurred, float* d_gradient, float* d_direction, int width, int height) {
    // Implement gradient computation logic
}

__global__ void NonMaxSuppressionKernel(float* d_gradient, float* d_direction, float* d_edges, int width, int height) {
    // Implement non-maximum suppression logic
}

__global__ void DoubleThresholdKernel(float* d_edges, float highThreshold, float lowThreshold, int width, int height) {
    // Implement double thresholding logic
}

__global__ void HysteresisKernel(float* d_edges, int width, int height) {
    // Implement edge tracking logic
}
