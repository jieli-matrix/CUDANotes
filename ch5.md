# 第5章 获得GPU加速的关键

## 5.1 使用CUDA事件计时

计时例程
``` c++
cudaEvent_t start, stop //定义cuda事件
CHECK(cudaEventCreate(&start)); //进行初始化
CHECK(cudaEventCreate(&stop));
CHECK(cudaEventRecord(start)); //记录start事件
cudaEventQuery(start); //将start事件传入CUDA流

/*
    需要计时的代码块
*/

CHECK(cudaEventRecord(stop));//记录stop事件
CHECK(cudaEventSynchronize(stop));//主机与设备间的同步
float elapsed_time;
CHECK(cudaEventElapsedTime(&elapsed_time, start, stop)); //计算时间差
printf("Time = %g ms.\n", elapsed_time);

CHECK(cudaEventDestroy(start));//销毁CUDA事件
CHECK(cudaEventDestroy(stop));
```

- 使用CUDA事件为C++函数计时

具体例程见[add1cpu.cu](src/ch5/add1cpu.cu)。相比于C++计时函数，其有如下改动  
- 使用.cu后缀
无需增加CUDA头文件
- 使用条件编译选择浮点数精度
``` c++
#ifdef USE_DP
    typedef double real;
    const real EPSILON = 1.0e-15;
#else
    typedef float real;
    const real EPSILON = 1.0e-6f;
#endif
```

当USE_DP有定义时，程序中的real代表double，否则代表float。该宏可通过编译选项定义。  

- trails计时
使用CUDA事件对add函数调用进行计时，平均多次trails时间，并忽略第一次测得的时间。

- 为CUDA函数计时

具体例程见[add2gpu.cu](ssrc/ch5/add2gpu.cu)。经1080Ti测试，单精度浮点数时核函数add所用时间约为3.3ms，双精度浮点数时所用时间约为6.6ms。多机测试后比值相似，均在2左右，说明该性能问题与单精度、双精度浮点数计算峰值无关，主要与访存效率有关。    
在此基础上，计时部分加入数据拷贝与传输时间，发现在1080Ti上，单精度总耗时达到291ms；双精度总耗时达到616ms。这表明核函数运行时间不到数据复制时间的2%，因此如果将CPU与GPU之间的数据传输时间计入，CUDA程序相对于C++程序反而有了性能降低。  
- 使用nvprof进行性能分析

``` shell
nvprof execution_file
nvprof --unified-memory-profiling execution_file
```

nvprof表项
(*待完善*)

## 5.2 影响GPU加速的关键因素

- 数据传输比重(较小)
- 核函数的算术强度(较高)
- 核函数中定义的线程数目(较多)
(*实验部分待完善*)


## 5.3 CUDA中的数学函数库

[数学库文档](http://docs.nvidia.com/cuda/cuda-math-api)  

数学库函数：
    - 内建函数(准确度较低，但效率较高)
    - 数学函数

Note: 使用半精度浮点数内建函数和数学函数需要包含头文件<cuda_fp16.h>。
