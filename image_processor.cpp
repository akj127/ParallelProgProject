#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <pthread.h>
#include <mpi.h>
#include <map>
#include <string>

// Profiling data structure
std::map<std::string, double> profilingData;

// Helper function to log timing information
void logProfiling(const std::string& section, const std::chrono::high_resolution_clock::time_point& start) {
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    profilingData[section] += duration;
}

// Output profiling data
void printProfiling() {
    std::cout << "\nProfiling Report:" << std::endl;
    for (const auto& entry : profilingData) {
        std::cout << entry.first << ": " << entry.second << " ms" << std::endl;
    }
}

// 占位实现（与之前一样）
cv::Mat applySobelGradient(const cv::Mat& input) {
    cv::Mat gradX, gradY, gradMag;
    cv::Sobel(input, gradX, CV_64F, 1, 0, 3);
    cv::Sobel(input, gradY, CV_64F, 0, 1, 3);
    cv::magnitude(gradX, gradY, gradMag);
    gradMag.convertTo(gradMag, CV_8U);
    return gradMag;
}

cv::Mat applyNonMaxSuppression(const cv::Mat& gradient) {
    // 占位：实际应实现非极大值抑制，这里直接返回
    return gradient.clone();
}

cv::Mat applyDoubleThreshold(const cv::Mat& input) {
    cv::Mat output;
    cv::threshold(input, output, 50, 255, cv::THRESH_TOZERO);
    cv::threshold(output, output, 150, 255, cv::THRESH_BINARY);
    return output;
}

cv::Mat applyHysteresis(const cv::Mat& input) {
    // 占位：实际应实现滞后阈值
    return input.clone();
}

// Data structure for pthread
struct ThreadData {
    const cv::Mat* input;
    cv::Mat* output;
    int startRow;
    int endRow;
};

// Thread functions for each step
void* threadGaussianBlur(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    cv::GaussianBlur((*data->input)(cv::Range(data->startRow, data->endRow), cv::Range::all()),
                     (*data->output)(cv::Range(data->startRow, data->endRow), cv::Range::all()),
                     cv::Size(5, 5), 1.4);
    pthread_exit(nullptr);
}

void* threadSobel(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    cv::Mat gradX, gradY, gradMag;
    cv::Sobel((*data->input)(cv::Range(data->startRow, data->endRow), cv::Range::all()), gradX, CV_64F, 1, 0, 3);
    cv::Sobel((*data->input)(cv::Range(data->startRow, data->endRow), cv::Range::all()), gradY, CV_64F, 0, 1, 3);
    cv::Mat localMag;
    cv::magnitude(gradX, gradY, localMag);
    localMag.convertTo(localMag, CV_8U);
    localMag.copyTo((*data->output)(cv::Range(data->startRow, data->endRow), cv::Range::all()));
    pthread_exit(nullptr);
}

void* threadNMS(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    // 占位实现，直接拷贝
    (*data->input)(cv::Range(data->startRow, data->endRow), cv::Range::all()).copyTo(
        (*data->output)(cv::Range(data->startRow, data->endRow), cv::Range::all()));
    pthread_exit(nullptr);
}

void* threadDoubleThreshold(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    cv::Mat local = (*data->input)(cv::Range(data->startRow, data->endRow), cv::Range::all());
    cv::Mat out;
    cv::threshold(local, out, 50, 255, cv::THRESH_TOZERO);
    cv::threshold(out, out, 150, 255, cv::THRESH_BINARY);
    out.copyTo((*data->output)(cv::Range(data->startRow, data->endRow), cv::Range::all()));
    pthread_exit(nullptr);
}

void* threadHysteresis(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    // 占位实现，直接拷贝
    (*data->input)(cv::Range(data->startRow, data->endRow), cv::Range::all()).copyTo(
        (*data->output)(cv::Range(data->startRow, data->endRow), cv::Range::all()));
    pthread_exit(nullptr);
}

// Sequential mode with detailed steps
void runSequence(const cv::Mat& image) {
    std::cout << "Running in Sequence mode..." << std::endl;

    auto totalStart = std::chrono::high_resolution_clock::now();

    auto profilingStart = std::chrono::high_resolution_clock::now();
    cv::Mat blurredImage;
    cv::GaussianBlur(image, blurredImage, cv::Size(5, 5), 1.4);
    logProfiling("Gaussian Blur", profilingStart);

    profilingStart = std::chrono::high_resolution_clock::now();
    cv::Mat gradImage = applySobelGradient(blurredImage);
    logProfiling("Sobel Gradient", profilingStart);

    profilingStart = std::chrono::high_resolution_clock::now();
    cv::Mat nmsImage = applyNonMaxSuppression(gradImage);
    logProfiling("Non-Max Suppression", profilingStart);

    profilingStart = std::chrono::high_resolution_clock::now();
    cv::Mat dtImage = applyDoubleThreshold(nmsImage);
    logProfiling("Double Threshold", profilingStart);

    profilingStart = std::chrono::high_resolution_clock::now();
    cv::Mat edges = applyHysteresis(dtImage);
    logProfiling("Hysteresis", profilingStart);

    cv::imwrite("output_sequence.png", edges);

    logProfiling("Total Time (Sequence)", totalStart);
}

// Helper to run a generic pthread step
void runPthreadStep(const cv::Mat& input, cv::Mat& output, void*(*func)(void*), const std::string& stepName) {
    int numThreads = 4;
    int rowsPerThread = input.rows / numThreads;
    pthread_t threads[numThreads];
    ThreadData threadData[numThreads];

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numThreads; ++i) {
        threadData[i] = {&input, &output, i * rowsPerThread, (i == numThreads - 1) ? input.rows : (i + 1) * rowsPerThread};
        pthread_create(&threads[i], nullptr, func, &threadData[i]);
    }
    for (int i = 0; i < numThreads; ++i) {
        pthread_join(threads[i], nullptr);
    }
    logProfiling(stepName, start);
}

// Pthread mode with detailed steps
void runPthreadMode(const cv::Mat& image) {
    std::cout << "Running in Pthread mode..." << std::endl;

    auto totalStart = std::chrono::high_resolution_clock::now();

    cv::Mat blurredImage = image.clone();
    runPthreadStep(image, blurredImage, threadGaussianBlur, "Gaussian Blur (Pthread)");

    cv::Mat gradImage = blurredImage.clone();
    runPthreadStep(blurredImage, gradImage, threadSobel, "Sobel Gradient (Pthread)");

    cv::Mat nmsImage = gradImage.clone();
    runPthreadStep(gradImage, nmsImage, threadNMS, "Non-Max Suppression (Pthread)");

    cv::Mat dtImage = nmsImage.clone();
    runPthreadStep(nmsImage, dtImage, threadDoubleThreshold, "Double Threshold (Pthread)");

    cv::Mat edges = dtImage.clone();
    runPthreadStep(dtImage, edges, threadHysteresis, "Hysteresis (Pthread)");

    cv::imwrite("output_pthread.png", edges);

    logProfiling("Total Time (Pthread)", totalStart);
}


// MPI mode with detailed steps
// 每个进程处理自己的一块图像，执行GaussianBlur、Sobel、NMS、DoubleThreshold、Hysteresis，然后Gather最终结果
// 假定各进程对图像分块在行方向上。
// 每个步骤执行完后，不需要在进程间传输中间结果（因为每个进程有自己的子块），最终只在Hysteresis结果后进行gather。
void runMPIMode(const cv::Mat& image) {
    std::cout << "Running in MPI mode..." << std::endl;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto totalStart = std::chrono::high_resolution_clock::now();

    int rowsPerProcess = image.rows / size;
    int startRow = rank * rowsPerProcess;
    int endRow = (rank == size - 1) ? image.rows : (rank + 1) * rowsPerProcess;

    cv::Mat localInput = image(cv::Range(startRow, endRow), cv::Range::all());

    auto profilingStart = std::chrono::high_resolution_clock::now();
    cv::Mat localBlurred;
    cv::GaussianBlur(localInput, localBlurred, cv::Size(5, 5), 1.4);
    logProfiling("Gaussian Blur (MPI)", profilingStart);

    profilingStart = std::chrono::high_resolution_clock::now();
    cv::Mat localGrad = applySobelGradient(localBlurred);
    logProfiling("Sobel Gradient (MPI)", profilingStart);

    profilingStart = std::chrono::high_resolution_clock::now();
    cv::Mat localNMS = applyNonMaxSuppression(localGrad);
    logProfiling("Non-Max Suppression (MPI)", profilingStart);

    profilingStart = std::chrono::high_resolution_clock::now();
    cv::Mat localDT = applyDoubleThreshold(localNMS);
    logProfiling("Double Threshold (MPI)", profilingStart);

    profilingStart = std::chrono::high_resolution_clock::now();
    cv::Mat localEdges = applyHysteresis(localDT);
    logProfiling("Hysteresis (MPI)", profilingStart);

    // Gather the final edges
    cv::Mat edges;
    if (rank == 0) {
        edges = cv::Mat::zeros(image.size(), image.type());
    }

    MPI_Gather(localEdges.ptr(), (int)localEdges.total(), MPI_UNSIGNED_CHAR,
               edges.ptr(), (int)localEdges.total(), MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        cv::imwrite("output_mpi.png", edges);
    }

    logProfiling("Total Time (MPI)", totalStart);
}


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <mode: 1=sequence, 2=pthread, 3=mpi>" << std::endl;
        return -1;
    }

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Cannot load image!" << std::endl;
        return -1;
    }

    int mode = std::stoi(argv[2]);
    if (mode == 1) {
        runSequence(image);
    } else if (mode == 2) {
        runPthreadMode(image);
    } else if (mode == 3) {
        MPI_Init(&argc, &argv);
        runMPIMode(image);
        MPI_Finalize();
    } else {
        std::cerr << "Invalid mode. Use 1 for sequence, 2 for pthread, 3 for mpi." << std::endl;
        return -1;
    }

    printProfiling();
    return 0;
}
