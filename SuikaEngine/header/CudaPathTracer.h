#pragma once

#ifndef __CUDA_PATHTRACER_H_
#define __CUDA_PATHTRACER_H_

#include "linear_algebra.h"
#include "geometry.h"
#include "bvh.h"
#include "Scene.h"

#define BVH_STACK_SIZE 32

#define DBG_PUTS(level, msg) \
    do { if (level <= 1) { puts(msg); fflush(stdout); }} while (0)

// The gateway to CUDA, called from C++ (src/main.cpp)

void RunKernel(size_t textureW, size_t textureH, cudaSurfaceObject_t surfaceObject, cudaSurfaceObject_t textureObject, cudaStream_t streamToRun, int i, float* paras);

/// <summary>
/// Create scene, 
/// By pushing all hitable into queue.
void HostCreateScene(Suika::Scene& scene);

void initCUDAmemoryTriMesh(float* dev_triangle_p);

void CudaFree(float* dev_triangle_p);

void panic(const char* fmt, ...);

float load_object(const char* filename);

void prepCUDAscene();

//void SetEnvironment(cudaTextureObject_t envmap);
void SetEnvironment(cudaSurfaceObject_t envmap);

struct Clock {
	unsigned firstValue;
	Clock() { reset(); }
	void reset() { firstValue = clock(); }
	unsigned readMS() { return (clock() - firstValue) / (CLOCKS_PER_SEC / 1000); }
};

#endif