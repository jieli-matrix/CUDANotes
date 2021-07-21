# 第2章 CUDA中的线程组织
## 2.1 C++中的Hello World程序

- 使用编辑器编写hello.cpp

``` shell
cat hello.cpp
```

``` c++
#include <stdio.h>

int main(void)
{
    printf("Hello World!\n");
    return 0;
}
```

- g++编译

``` shell
g++ hello.cpp -o hello # -o指定输出文件名
```

- 程序运行

``` shell
./hello
Hello World!
```

## 2.2 CUDA中的HelloWorld程序

- CUDA编译流程

CUDA程序的编译器驱动为nvcc，其支持编译纯粹的C++代码；一个标准的CUDA程序，由C++代码和设备代码组成；编译过程中，C++代码交给C++编译器，其负责其余设备代码。CUDA程序源文件的后缀名默认是cu，使用nvcc对hello.cu进行编译

- 核函数

利用的GPU的CUDA程序:  
主机代码 + 设备代码  
主机对设备的调用通过核函数来实现

- 典型工作流

一个典型、简单的CUDA程序结构如下所示  
*主文件代码*
``` c++
int main(void)
{
    // 主机代码
    // 核函数的调用
    // 主机代码
    return 0;
}
```
*核函数代码*

限定符修饰: \__global\__(双下划线)，返回类型必须为空类型void

``` c++
__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}
```

完整代码见[hello1.cu](src/ch2/hello1.cu)

``` c++
// hello1.cu

#include <stdio.h>

__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}

int main(void)
{
    hello_from_gpu<<<1, 1>>>(); // Set Grid and Block
    cudaDeviceSynchronize(); //CUDA Runtime API
    return 0;
}
```

- 要点分析  
    - <<<网格大小,线程块大小>>>  
        根据SIMT思想，对设备线程进行配置与指配：线程构成线程块(Block)；线程块构成网格(Grid)。其中，线程块运行在单个SM上。
    - 同步  
        使用cudaDeviceSynchronize()进行主机与设备的同步，促进缓存区的刷新。

## 2.3 CUDA中的线程组织

- SIMT  
核函数允许指派多线程，当总线程数大于等于计算核心数能够充分利用GPU中的全部计算资源——计算和访存之间合理的重叠。  
- 线程索引  

配置线程的组织结构  
``` c++
<<<grid_size, block_size>>>
// grid_size block_size一般意义下是结构体变量
// 可以是普通的整型变量
```
线程块大小最大为1024，网格大小最大为$2^31 - 1$(对于一维网格而言)。尽管一个核函数中可以指派巨大数目的线程数，但执行时能够同时活跃的线程数是由硬件(SM中的warp线程组与warp调度器)和软件决定的。  
- 单维网格  
    核函数的内建变量    
```c++
    gridDim.x //grid_size
    blockDim.x //block_size
    blockIdx.x //线程在一个网格中的线程块指标
    threadIdx.x //线程在一个线程块中的线程指标
```

- 多维网格

blockIdx与threadIdx为unit3的变量，具有x, y, z三个成员

``` c++
struct __device_builtin__ uint3
{
unsigned int x, y, z;
};
typedef __device_builtin__ struct uint3 uint3;
```

gridDim和blockDim为类型为dim3的变量，具有x, y, z三个成员，定义与unit3变量类似。使用结构体dim3定义"多维"的网格和线程块

``` c++
dim3 grid_size(Gx, Gy, Gz);
dim3 block_size(Bx, By, Bz);
```

如果第三个维度的大小是1，可以定义如下

``` c++
dim3 grid_size(Gx, Gy);
dim3 block_size(Bx, By);
```  

多维线程指标的变化  

``` c++
int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z*blockDim.y*blockDim.x;
```

复合指标

``` c++
int nx = blockDim.x * blockIdx.x + threadIdx.x;
int ny = blockDim.y * blockIdx.y + threadIdx.y;
int nz = blockDim.z * blockIdx.z + threadIdx.z;
```

- 网格与线程块大小的限制

*Constraints on Grid*

``` c++
dim3 grid_size(Gx, Gy, Gz);
Gx <= 2^31 - 1
Gy <= 65535
Gz <= 65535
```

*Constraints on Block*

``` c++
dim3 block_size(Bx, By, Bz);
Bx <= 1024
By <= 1024
Bz <= 64
Bx * By * Bz <= 1024 // very important！
```

## 2.4 CUDA中的头文件

本章程序仅包含c++头文件<stdio.h>，但没有包含任何CUDA相关头文件；使用nvcc编译器驱动编译.cu文件时，将自动包含必要的CUDA头文件，如<cuda.h>和<cuda_runtime.h>。

在使用CUDA进行加速应用程序时，需要包含必要头文件及指定编译链接选项。

## 2.5 nvcc编译CUDA程序

- 代码分离                        
source code -> nvcc: host code & device code
- 编译流程
device code -> PTX code -> binary cubin code

-arch=compute_XY 指定虚拟架构，确定代码中能够使用的CUDA功能
-code=sm_ZW 指定真实架构的**计算能力**，确定可执行文件能够使用的GPU

*真实架构的计算能力>=虚拟架构的计算能力，一般而言，保持相同*

-gencode arch=compute_70, code=sm_70
简化形式
-arch=sm_XY <=> -gencode arch=compute_XY, code=sm_XY

- 即时编译

在运行可执行文件时，从其中保留的PTX代码中，临时编译出一个cubin目标代码。  
使用如下方式指定所保留PTX代码的虚拟架构

-gencode arch=compute_XY, code=compute_XY

该方式不一定能充分利用当前设备的硬件架构，在真实架构运行时，会根据虚拟架构的PTX代码即时编译出适用于当前GPU的目标代码。

- 默认计算能力

目前编译程序时，没有通过编译选项指定计算能力；因为不同CUDA版本的编译器在编译CUDA代码有默认的计算能力：
- CUDA Version <= 6.0, 默认计算能力1.0
- 6.5 <= CUDA Version <= 8.0, 默认计算能力2.0
- 9.0 <= CUDA Version <= 10.2, 默认计算能力3.0
                        
                      