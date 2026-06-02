# GPU-Accelerated-Image-Processing-System-Using-CUDA-and-OpenCV.
## Project Overview

This project demonstrates a high-performance GPU-based image processing system developed using CUDA and OpenCV. The application utilizes NVIDIA GPU parallel computing to accelerate multiple image processing operations such as grayscale conversion, image blurring, sharpening, and Sobel edge detection.

Traditional CPU-based image processing can become slow when handling large images or real-time applications. To overcome this limitation, this project uses CUDA kernels to execute image operations in parallel on thousands of GPU threads, significantly improving performance and execution speed.

The system loads an input image, transfers it to GPU memory, processes it using CUDA kernels, and generates multiple processed output images. Performance timing is also measured to analyze GPU acceleration efficiency.

## Technologies Used
### Technology	Purpose
CUDA	GPU parallel programming
NVIDIA GPU	Hardware acceleration
C++	Core programming language
OpenCV	Image loading and saving
CUDA Runtime API	GPU memory management
Google Colab / Linux	Development environment
## Features
GPU-based grayscale conversion
CUDA blur filtering
Image sharpening using convolution
Sobel edge detection
Parallel image processing
GPU execution time measurement
Multiple output image generation
CUDA memory management
## Project Structure
```
GPU-Image-Processing/
│
├── main.cu
├── input.jpg
├── grayscale.jpg
├── blur.jpg
├── sharpen.jpg
├── edge.jpg
├── README.md
└── report.pdf
```
## program
```
%%writefile main.cu

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;
using namespace cv;

// ---------------------- GRAYSCALE KERNEL ----------------------
__global__ void grayscaleKernel(unsigned char* input, unsigned char* output,
                                int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;

        unsigned char b = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char r = input[idx + 2];

        output[y * width + x] = (0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// ---------------------- BLUR KERNEL ----------------------
__global__ void blurKernel(unsigned char* input, unsigned char* output,
                           int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
        int sum = 0;

        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                sum += input[(y + ky) * width + (x + kx)];
            }
        }

        output[y * width + x] = sum / 9;
    }
}

// ---------------------- SHARPEN KERNEL ----------------------
__global__ void sharpenKernel(unsigned char* input, unsigned char* output,
                              int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int kernel[3][3] = {
        { 0, -1,  0},
        {-1,  5, -1},
        { 0, -1,  0}
    };

    if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
        int sum = 0;

        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                sum += input[(y + ky) * width + (x + kx)] *
                       kernel[ky + 1][kx + 1];
            }
        }

        output[y * width + x] = min(max(sum, 0), 255);
    }
}

// ---------------------- SOBEL EDGE DETECTION ----------------------
__global__ void sobelKernel(unsigned char* input, unsigned char* output,
                            int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
        int sumX = 0;
        int sumY = 0;

        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int pixel = input[(y + ky) * width + (x + kx)];

                sumX += pixel * Gx[ky + 1][kx + 1];
                sumY += pixel * Gy[ky + 1][kx + 1];
            }
        }

        int magnitude = min((int)sqrtf(sumX * sumX + sumY * sumY), 255);
        output[y * width + x] = magnitude;
    }
}

// ---------------------- MAIN FUNCTION ----------------------
int main() {
    Mat image = imread("input.jpg");

    if (image.empty()) {
        cout << "Error loading image!" << endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
    int channels = image.channels();

    size_t colorSize = width * height * channels * sizeof(unsigned char);
    size_t graySize = width * height * sizeof(unsigned char);

    unsigned char *d_input, *d_gray, *d_blur, *d_sharp, *d_edge;

    cudaMalloc((void**)&d_input, colorSize);
    cudaMalloc((void**)&d_gray, graySize);
    cudaMalloc((void**)&d_blur, graySize);
    cudaMalloc((void**)&d_sharp, graySize);
    cudaMalloc((void**)&d_edge, graySize);

    cudaMemcpy(d_input, image.data, colorSize, cudaMemcpyHostToDevice);

    dim3 threads(16,16);
    dim3 blocks((width + 15)/16, (height + 15)/16);

    auto start = chrono::high_resolution_clock::now();

    // Grayscale
    grayscaleKernel<<<blocks, threads>>>(d_input, d_gray, width, height, channels);
    cudaDeviceSynchronize();

    // Blur
    blurKernel<<<blocks, threads>>>(d_gray, d_blur, width, height);
    cudaDeviceSynchronize();

    // Sharpen
    sharpenKernel<<<blocks, threads>>>(d_gray, d_sharp, width, height);
    cudaDeviceSynchronize();

    // Edge Detection
    sobelKernel<<<blocks, threads>>>(d_gray, d_edge, width, height);
    cudaDeviceSynchronize();

    auto stop = chrono::high_resolution_clock::now();

    vector<unsigned char> h_gray(width * height);
    vector<unsigned char> h_blur(width * height);
    vector<unsigned char> h_sharp(width * height);
    vector<unsigned char> h_edge(width * height);

    cudaMemcpy(h_gray.data(), d_gray, graySize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_blur.data(), d_blur, graySize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sharp.data(), d_sharp, graySize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_edge.data(), d_edge, graySize, cudaMemcpyDeviceToHost);

    Mat grayImage(height, width, CV_8UC1, h_gray.data());
    Mat blurImage(height, width, CV_8UC1, h_blur.data());
    Mat sharpImage(height, width, CV_8UC1, h_sharp.data());
    Mat edgeImage(height, width, CV_8UC1, h_edge.data());

    imwrite("grayscale.jpg", grayImage);
    imwrite("blur.jpg", blurImage);
    imwrite("sharpen.jpg", sharpImage);
    imwrite("edge.jpg", edgeImage);

    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);

    cout << "GPU Execution Time: " << duration.count() << " ms" << endl;
    cout << "Generated Files:" << endl;
    cout << "grayscale.jpg" << endl;
    cout << "blur.jpg" << endl;
    cout << "sharpen.jpg" << endl;
    cout << "edge.jpg" << endl;

    cudaFree(d_input);
    cudaFree(d_gray);
    cudaFree(d_blur);
    cudaFree(d_sharp);
    cudaFree(d_edge);

    return 0;
}
```
## output

<img width="348" height="306" alt="image" src="https://github.com/user-attachments/assets/15abdd98-df72-4a81-b022-3aeaba39037f" />
<img width="351" height="308" alt="image" src="https://github.com/user-attachments/assets/963b7461-4c39-44de-b231-424e662530af" />
<img width="355" height="309" alt="image" src="https://github.com/user-attachments/assets/eb8cad71-a2f4-47ef-91a6-e35aa834d60c" />
<img width="346" height="299" alt="image" src="https://github.com/user-attachments/assets/ed8792e9-ee7b-4c6a-b8ba-78d12985829d" />

## Result 
This project demonstrates the power of GPU computing for accelerating image processing tasks using CUDA. By leveraging thousands of parallel GPU threads, the system achieves efficient and high-speed execution of image operations such as grayscale conversion, blurring, sharpening, and edge detection. The project highlights practical applications of CUDA in computer vision and real-time image analysis systems.
