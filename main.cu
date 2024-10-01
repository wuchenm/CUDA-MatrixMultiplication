#include <iostream>
#include <cstdlib>

// CUDA kernel function for matrix multiplication
__global__ void matrixMulKernel(int *A, int *B, int *C, int M, int N, int P) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < P) {
        int sum = 0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * P + col];
        }
        C[row * P + col] = sum;
    }
}

int main() {
    // Matrix dimensions
    int M = 3; // Rows in matrix A and C
    int N = 3; // Columns in matrix A and rows in matrix B
    int P = 3; // Columns in matrix B and C

    // Allocate memory for matrices on host (CPU)
    int *h_A = (int *)malloc(M * N * sizeof(int));
    int *h_B = (int *)malloc(N * P * sizeof(int));
    int *h_C = (int *)malloc(M * P * sizeof(int));

    // Initialize matrices A and B
    for (int i = 0; i < M * N; ++i) h_A[i] = rand() % 10; // Fill with random numbers
    for (int i = 0; i < N * P; ++i) h_B[i] = rand() % 10; // Fill with random numbers

    // Allocate memory on device (GPU)
    int *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, M * N * sizeof(int));
    cudaMalloc((void **)&d_B, N * P * sizeof(int));
    cudaMalloc((void **)&d_C, M * P * sizeof(int));

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A, M * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * P * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16); // 16x16 threads per block
    dim3 blocksPerGrid((P + 15) / 16, (M + 15) / 16); // Adjust grid size to cover the whole matrix

    // Launch the kernel
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, P);

    // Copy the result matrix C from device to host
    cudaMemcpy(h_C, d_C, M * P * sizeof(int), cudaMemcpyDeviceToHost);

    // Display the result
    std::cout << "Matrix A:\n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_A[i * N + j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nMatrix B:\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            std::cout << h_B[i * P + j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nMatrix C (Result):\n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            std::cout << h_C[i * P + j] << " ";
        }
        std::cout << "\n";
    }

    // Free memory on device and host
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
