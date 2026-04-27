#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <random>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        cerr << "CUDA Error: " << cudaGetErrorString(err) << endl; \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void matrixMultiplyKernel(int n, const double* A, const double* B, double* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    vector<int> sizes = {200, 400, 800, 1200, 1600, 2000};
    
    // 8, 16, 32
    int threadsPerBlockSide = 32; 
    
    ofstream dataFile("data_cuda.txt");
    
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(1.0, 10.0);

    for (int n : sizes) {
        cout << "Processing N = " << n << " with CUDA..." << endl;

        size_t matrixSize = n * n * sizeof(double);
        vector<double> h_A(n * n), h_B(n * n), h_C(n * n);

        for (int i = 0; i < n * n; i++) {
            h_A[i] = dis(gen);
            h_B[i] = dis(gen);
        }

        double *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, matrixSize);
        cudaMalloc(&d_B, matrixSize);
        cudaMalloc(&d_C, matrixSize);

        cudaMemcpy(d_A, h_A.data(), matrixSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), matrixSize, cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(threadsPerBlockSide, threadsPerBlockSide);
        dim3 blocksPerGrid((n + threadsPerBlockSide - 1) / threadsPerBlockSide, 
                           (n + threadsPerBlockSide - 1) / threadsPerBlockSide);

        auto start = chrono::high_resolution_clock::now();
        
        matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_A, d_B, d_C);
        cudaDeviceSynchronize(); 
        
        auto end = chrono::high_resolution_clock::now();

        cudaMemcpy(h_C.data(), d_C, matrixSize, cudaMemcpyDeviceToHost);

        double time_spent = chrono::duration<double>(end - start).count();
        cout << "   Time: " << time_spent << "s" << endl;
        
        dataFile << n << " " << time_spent << " " << threadsPerBlockSide << endl;

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    dataFile.close();
    cout << "\nDone! Results saved to data_cuda.txt" << endl;

    return 0;
}
