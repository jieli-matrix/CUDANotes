#include <stdio.h>

__global__ void hello_from_gpu()
{
    const int bid = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    printf("Hello World from the GPU in block %d thread (%d, %d)!\n", bid, tx, ty);
}

int main(void)
{
    dim3 block_size(2,4)
    hello_from_gpu<<<2, block_size>>>();
    cudaDeviceSynchronize(); 
    return 0;
}