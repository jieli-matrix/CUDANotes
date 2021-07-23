#include <math.h>
#include <stdlib.h>
#include <stdio.h>

const double EPSILON = 1.03-15; //宏定义
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
// function declare
void add(const double *x, const double *y, double *z, const int N);
void check(const double *z, const int N);

int main(void)
{
    const int N = 100000000;
    const int M = sizeof(double) * N;
    // request for memory
    double *x = (double*) malloc(M);
    double *y = (double*) malloc(M);
    double *z = (double*) malloc(M);

    // initialize for array x, y
    for (int i = 0; i < N; ++i)
    {
        x[i] = a;
        y[i] = b;
    }

    // vector add
    add(x,y,z, N);
    // error analysis
    check(z, N);

    // memory free
    free(x);
    free(y);
    free(z);
    return 0;
}

void add(const double *x, const double *y, double *z, const int N)
{
    for (int i = 0; i < N; i++)
    {
        z[i] = x[i] + y[i];
    }
    
}

void check(const double *z, const int N)
{
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
