#ifndef __UTILITYH__
#define __UTILITYH__

// c++ headers
#include <cstdio>
#include <algorithm>
#include <climits>
#include <chrono>

// cuda headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "helper_math.h"
#include <CudaUtil.h>

#define M_PI 3.1415926

__device__ float fract(float seed)
{
    return seed - floorf(seed);
}

__device__ float2 fract(float2 input)
{
    return make_float2(fract(input.x), fract(input.y));
}

__device__ float mod(float a, float b)
{
    return a - floorf(a/b)*b;
}

inline int gammaCorrect(float c)
{
    return int(pow(clamp(c, 0.0f, 1.0f), 1 / 2.2) * 255 + .5);
}

#endif