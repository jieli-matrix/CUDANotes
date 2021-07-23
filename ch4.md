# 第4章 CUDA程序的错误检测

- 错误  
    - 编译错误  
    - **运行时错误**(更难排查)

## 4.1 一个检测CUDA运行时错误的宏函数

已使用过的cuda运行时API:  
- cudaMalloc()
- cudaFree()  
- cudaMemcpy()  
cuda运行时API以cuda为前缀，其返回值为cudaError_t，只有返回值为cudaSuccess才表明成功调用了API函数。  
*例程错误检测宏函数*  
以头文件的形式保存，见[error.cuh](src/ch4/error.cuh)

*调用该函数*
使用CHECK(call)替代call，即可在输出中返回相关错误信息。调用实例，见[check1api.cu](src/ch4/check1api.cu)。
*检查核函数*
核函数无返回值，其错误排查通过如下两个语句：  

``` c++
CHECK(cudaGetLastError()); //捕捉错误
CHECK(cudaDeviceSynchronize()); //同步主机和设备
```

cudaMemcpy具有隐式的同步，但在一般情形下，使用cudaGetLastError需要进行显式同步。cudaDeviceSynchronize使得主机调用核函数之后，必须等待核函数执行完毕，才往下走。这会影响程序的性能，一般仅用于程序调试。

## 4.2 使用CUDA-MEMCHECK检查内存错误

memcheck工具可简化为如下调用

``` shell
cuda-memcheck [options] app_name [options]
```

对程序是否进行非法内存访问检查，以[add3if.cu](src/ch4/add3if.cu)为例，其设定数组与block大小如下:

``` c++
const int N = 100000001;
const int block_size = 128;
```
显然，N % block_size = 1 < 32，根据CUDA最小资源调度单位是大小为32的warp线程束，如不加以访问线程限制，则会访问到未分配内存。

``` c++
const int block_size = 128;
const int grid_size = (N + block_size - 1) / block_size;
add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);

void __global__ add(const double *x, const double *y, double *z, const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        z[n] = x[n] + y[n];
    }
    //z[n] = x[n] + y[n];
}
```
当我们去掉if语句，使用CUDA-MEMCHECK则会出现 36 errors报错；当加入if语句后，使用CUDA-MEMCHECK则会返回0 errors。