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

// Data structure for pthread
struct ThreadData {
    cv::Mat input;
    cv::Mat output;
    int startRow;
    int endRow;
};

// Thread function for Gaussian blur
void* applyGaussianBlur(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    cv::GaussianBlur(data->input(cv::Range(data->startRow, data->endRow), cv::Range::all()),
                     data->output(cv::Range(data->startRow, data->endRow), cv::Range::all()),
                     cv::Size(5, 5), 1.4);
    pthread_exit(nullptr);
}

// Thread function for Canny edge detection
void* applyCannyEdgeDetection(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    cv::Canny(data->input(cv::Range(data->startRow, data->endRow), cv::Range::all()),
              data->output(cv::Range(data->startRow, data->endRow), cv::Range::all()),
              50, 150);
    pthread_exit(nullptr);
}

// Sequential mode
void runSequence(const cv::Mat& image) {
    std::cout << "Running in Sequence mode..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    auto profilingStart = start;

    cv::Mat blurredImage;
    cv::GaussianBlur(image, blurredImage, cv::Size(5, 5), 1.4);
    logProfiling("Gaussian Blur (Sequence)", profilingStart);

    profilingStart = std::chrono::high_resolution_clock::now();
    cv::Mat edges;
    cv::Canny(blurredImage, edges, 50, 150);
    logProfiling("Canny Edge Detection (Sequence)", profilingStart);

    cv::imwrite("output_sequence.png", edges);

    logProfiling("Total Sequence Mode", start);
}

// Multithreading mode using pthread
void runPthread(const cv::Mat& image) {
    std::cout << "Running in Pthread mode..." << std::endl;

    const int numThreads = 4;
    pthread_t threads[numThreads];
    ThreadData threadData[numThreads];
    int rowsPerThread = image.rows / numThreads;

    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat blurredImage = image.clone();
    for (int i = 0; i < numThreads; ++i) {
        threadData[i] = {image, blurredImage, i * rowsPerThread,
                         (i == numThreads - 1) ? image.rows : (i + 1) * rowsPerThread};
        pthread_create(&threads[i], nullptr, applyGaussianBlur, &threadData[i]);
    }

    for (int i = 0; i < numThreads; ++i) {
        pthread_join(threads[i], nullptr);
    }
    logProfiling("Gaussian Blur (Pthread)", start);

    start = std::chrono::high_resolution_clock::now();
    cv::Mat edges = blurredImage.clone();
    for (int i = 0; i < numThreads; ++i) {
        threadData[i] = {blurredImage, edges, i * rowsPerThread,
                         (i == numThreads - 1) ? image.rows : (i + 1) * rowsPerThread};
        pthread_create(&threads[i], nullptr, applyCannyEdgeDetection, &threadData[i]);
    }

    for (int i = 0; i < numThreads; ++i) {
        pthread_join(threads[i], nullptr);
    }
    logProfiling("Canny Edge Detection (Pthread)", start);

    cv::imwrite("output_pthread.png", edges);

    logProfiling("Total Pthread Mode", start);
}

// MPI mode
void runMPI(const cv::Mat& image) {
    std::cout << "Running in MPI mode..." << std::endl;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rowsPerProcess = image.rows / size;
    int startRow = rank * rowsPerProcess;
    int endRow = (rank == size - 1) ? image.rows : (rank + 1) * rowsPerProcess;

    cv::Mat localInput = image(cv::Range(startRow, endRow), cv::Range::all());
    cv::Mat localBlurred, localEdges;

    auto start = std::chrono::high_resolution_clock::now();

    cv::GaussianBlur(localInput, localBlurred, cv::Size(5, 5), 1.4);
    cv::Canny(localBlurred, localEdges, 50, 150);

    logProfiling("Processing (MPI)", start);

    cv::Mat edges;
    if (rank == 0) {
        edges = cv::Mat::zeros(image.size(), image.type());
    }
    MPI_Gather(localEdges.ptr(), localEdges.total(), MPI_UNSIGNED_CHAR, edges.ptr(),
               localEdges.total(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cv::imwrite("output_mpi.png", edges);
    }

    logProfiling("Total MPI Mode", start);
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
        runPthread(image);
    } else if (mode == 3) {
        MPI_Init(&argc, &argv);
        runMPI(image);
        MPI_Finalize();
    } else {
        std::cerr << "Invalid mode. Use 1 for sequence, 2 for pthread, 3 for mpi." << std::endl;
        return -1;
    }

    printProfiling();
    return 0;
}
