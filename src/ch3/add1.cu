#include <math.h>
#include <stdio.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

void __global__ add(const double *x, const double *y, double *z);
void check(const double *z, const int N);

int main(void)
{
    const int N = 100000000;
    const int M = sizeof(double) * N;
    // request for host memory
    double *h_x = (double*) malloc(M);
    double *h_y = (double*) malloc(M);
    double *h_z = (double*) malloc(M);

    for(int i = 0; i < N; ++i){
        h_x[i] = a;
        h_y[i] = b;
    }

    // request for device memory
    double *d_x, *d_y, *d_z;
    cudaMalloc((void **)&d_x, M);
    cudaMalloc((void **)&d_y, M);
    cudaMalloc((void **)&d_z, M);
    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);

    // set for grid and block
    const int block_size = 128;
    const int grid_size = N / block_size;
    add<<<grid_size, block_size>>>(d_x, d_y, d_z);

    // TransferData
    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
    check(h_z, N);

    // Free Memory
    free(h_x);
    free(h_y);
    free(h_z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    return 0;
}

void __global__ add(const double *x, const double *y, double *z){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    z[tid] = x[tid] + y[tid];
}

void check(const double *z, const int N){
    bool has_error = false;
    for (int i = 0; i < N; i++)
    {
        if (fabs(z[i] - c)>EPSILON)
        {
            has_error = true;
        }
        
    }
    printf("%s\n", has_error?"Has errors" : "No errors");
}