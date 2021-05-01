#include <CudaFunc.h>
#include "RTUtils.cuh"
#include "Utility.cuh"
#include "HitableList.cuh"
#include "Sphere.cuh"
#include "DevMaterial.cuh"
#include <Debug.h>

// #include "helper_math.h"

int iDivUp(int a, int b) { return a % b != 0 ? a / b + 1 : a / b; }

#define WIDTH 800
#define HEIGHT 600
#define SPP 1
#define BOUNCE 5
#define SPHERE_EPSILON 0.0001f
#define BOX_EPSILON 0.001f
#define RAY_EPSILON 0.05f;
#define M_PI 3.1415926f;

// =============== Worldwide constant ===============
__device__ Hitable *world;
__constant__ DevCamera camera;
__constant__ float iTime;
// ==================================================

__device__ float3 color(const Ray& r,  RandInstance& rnd, int depth  = 0)
{
    HitRecord rec;
    if ((world)->hit(r, 0.001, 100, rec))
    {
        if (depth >= BOUNCE)
           return make_float3(0.0);
        
        Ray scattered;
        float3 attenuation;

        if(rec.mat_ptr->scatter(r, rec, attenuation, scattered, rnd))
        {
            return attenuation * color(scattered, rnd, depth+1);
        }
    }
    else
    {
        float3 unit_direction = normalize(r.dir);
        float tt = 0.5 * (unit_direction.y + 1.);
        return (1.0 - tt) * make_float3(1.0) + tt * make_float3(0.5f, 0.7, 1.0);
    }
}

__device__ float3 RayTrace(const Ray& r, RandInstance& rnd)
{
    int Bounce = 3;
    float3 col = make_float3(0);
    for(int i=0;i<Bounce;i++)
    {
        col = col + color(r,rnd,0);
    }
    col = col / (1. * Bounce);
    return col;
}

__global__ void MainCall(cudaSurfaceObject_t surf, unsigned int width, unsigned int height, int index)
{
    // Get coordinate
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    // If out of bounds, return
    if (y >= height | x >= width) return;
    // Get uniformed pixel position coordinate
    float u = float(x) / width;
    float v = float(y) / height;

    // Init the rand instance for current
    RandInstance rnd(make_float2(u,v),iTime);
    // Prepare the color for the pixel
    float3 color;
    if (index == 0)
    {
        color = make_float3(0,0,0);
        for(int i = 0; i<SPP;i++)
        {
            // Get the ray upon the pixel
            Ray ray = camera.getRay(u + rnd.rand()/width, v + rnd.rand()/height, rnd);
            color = color + RayTrace(ray, rnd);
        }
        color = color / SPP;
    }
    else
    {
        // Init color as the last frame
        float4 pixel4{};
        surf2Dread(&pixel4, surf, x * 16, y);
        color = make_float3(pixel4.x, pixel4.y, pixel4.z);
        // Add color to the init color
        for(int i = 0; i<SPP;i++)
        {
            // Get the ray upon the pixel
            Ray ray = camera.getRay(u + rnd.rand()/width, v + rnd.rand()/height, rnd);
            color = color + RayTrace(ray, rnd) * (1. / (1. * SPP * index));
        }
        color = color / (1 + 1. / index);
    }
    
    // Write to the texture
    float4 pixel = make_float4(color, 1.0);
    surf2Dwrite(pixel, surf, x * 16, y);
}

__global__ void CreateScene(Hitable** hitable, int i)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        hitable[0] = new Sphere(make_float3(0, 0, -1), 0.5, new DevLambertian(make_float3(0.8,0.3,0.3)));
        hitable[1] = new Sphere(make_float3(0, -100.5, -1), 100, new DevLambertian(make_float3(0.8,0.8,0.0)));
        hitable[2] = new Sphere(make_float3(1, 0, -1), 0.5, new DevMetal(make_float3(0.8,0.6,0.2),0.3));
        hitable[3] = new Sphere(make_float3(-1, 0, -1), 0.5, new DevMetal(make_float3(0.8,0.8,0.8),1.));
        world = new HitableList(hitable,4);
    }
}

__global__ void DeleteOnDevice(Hitable** hitable)
{
    delete[](*hitable);
    hitable = nullptr;
}

extern "C" void RunKernel(size_t textureW, size_t textureH, cudaSurfaceObject_t surfaceObject, 
    cudaStream_t streamToRun, int i, float* paras)
{
    // Create Scene
    // -------------------------------
    Hitable** hitable = nullptr;
    cudaMalloc((void**)&hitable, 4 * sizeof(Hitable**));
    CreateScene<<<1,1>>>(hitable, i);

    // Create camera
    // -------------------------------
    float3 lookfrom = make_float3(0.1*paras[0],0.1*paras[1],0.1*paras[2]);
    float3 lookat = make_float3(0.1*paras[3],0.1*paras[4],0.1*paras[5]);
    float dist_to_focus = 10;
    float aperture = 2;

    DevCamera* tmp = new DevCamera(lookfrom, lookat, make_float3(0,1,0), 20, textureW/textureH, aperture, dist_to_focus);
    float animTime = 0.01 * i;
    checkCudaErrors(cudaMemcpyToSymbol(camera, tmp, sizeof(DevCamera),0,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(iTime, &animTime, sizeof(float),0,cudaMemcpyHostToDevice));

    // Emit main call kernel
    // -------------------------------
    auto unit = 10;
    dim3 threads(unit, unit);
    dim3 grid(iDivUp(textureW, unit), iDivUp(textureH, unit));

    MainCall <<<grid, threads, 0, streamToRun >>> (surfaceObject, textureW, textureH, i);
    checkCudaErrors(cudaGetLastError());
}