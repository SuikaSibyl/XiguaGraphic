#include "RTUtils.cuh"
#include "Utility.cuh"
#include "HitableList.cuh"
#include "Sphere.cuh"
#include "DevMaterial.cuh"
#include <Debug.h>
#include <CudaPathTracer.h>
#include <TriangleModel.h>
#include <CudaPrt.h>
#include <Geometry.h>
#include "SphericalHarmonic.cuh"
#include <Mesh.h>

int iDivUp(int a, int b) { return a % b != 0 ? a / b + 1 : a / b; }

#define WIDTH 800
#define HEIGHT 600
#define SPP 1
#define RAYNUM 1
#define BOUNCE 4
#define SPHERE_EPSILON 0.0001f
#define BOX_EPSILON 0.001f
#define RAY_EPSILON 0.05f

#define M_PI 3.1415926535897932384626422832795028841971f
#define TWO_PI 6.2831853071795864769252867665590057683943f
#define NUDGE_FACTOR     1e-3f  // epsilon
#define samps  1 // samples
#define BVH_STACK_SIZE 32
#define SCREEN_DIST (height*2)
// =============== Worldwide constant ===============
__device__ Hitable *world;
__constant__ DevCamera camera;
__constant__ float iTime;
// ==================================================


cudaTextureObject_t cubemap_tex;
__device__ cudaTextureObject_t cu_envmap;
__device__ cudaSurfaceObject_t cu_surfenvmap;

__device__ float2 SampleSphericalMap(float3 v)
{
    float2 uv = make_float2(atan2f(v.z, v.x), asin(v.y));
    uv *= make_float2(0.1591, 0.3183);
    uv += 0.5;
    return uv;
}

__device__ float3 Radiance(const Ray& r,  RandInstance& rnd, cudaSurfaceObject_t textureObject,  int depth  = 0)
{
    HitRecord rec;

    if ((world)->hit(r, 0.001, 100, rec))
    {
        if (depth >= BOUNCE)
           return make_float3(0.0);
        
        Ray scattered; 
        float3 attenuation;
        float3 emission;

        if(rec.mat_ptr->scatter(r, rec, attenuation, emission, scattered, rnd))
        {
            return emission + attenuation * Radiance(scattered, rnd, textureObject, depth + 1);
        }
        else
        {
            return  make_float3(0,0,0);
        }
    }
    else
    {
        float4 pixel4{};
        float2 uv = SampleSphericalMap(r.dir);
        surf2Dread(&pixel4, textureObject, 16 * 3200 * uv.x, 1600 * uv.y);
        return make_float3(pixel4.x, pixel4.y, pixel4.z);
    }
}

__device__ float3 RayTrace(const Ray& r, RandInstance& rnd, cudaSurfaceObject_t textureObject)
{
    // accumulates ray colour with each iteration through bounce loop
    float3 accucolor = make_float3(0.0f, 0.0f, 0.0f); 
    float3 mask = make_float3(1.0f, 1.0f, 1.0f); 

    for(int i=0; i<RAYNUM; i++)
    {
        accucolor += Radiance(r,rnd, textureObject, 0);
    }
    accucolor /= (1. * RAYNUM);
    accucolor = make_float3(powf(accucolor.x, 1.0 / 2.2), powf(accucolor.y, 1.0 / 2.2), powf(accucolor.z, 1.0 / 2.2));
    return accucolor;
}

__global__ void MainCall(cudaSurfaceObject_t surf, cudaSurfaceObject_t textureObject, unsigned int width, unsigned int height, int index)
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
            color = color + RayTrace(ray, rnd, textureObject);
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
            color = color + RayTrace(ray, rnd, textureObject) * (1. / (1. * SPP * index));
        }
        color = color / (1 + 1. / index);
    }
    
    // Write to the texture
    float4 pixel = make_float4(color, 1.0);
    surf2Dwrite(pixel, surf, x * 16, y);
}

__global__ void CreateScene(Hitable** hitable, Suika::CudaTriangleModel** models)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        //hitable[0] = new Sphere(make_float3(0, 1, 1.2), 0.5, new DevLambertian(make_float3(0.1,0.1,1),make_float3(0,0,0)));
        //hitable[1] = new Sphere(make_float3(-0.5, -99.4, -1), 100, new DevLambertian(make_float3(0.5f, 0.0f, 0.0f),make_float3(0,0,0)));
        //hitable[2] = new Sphere(make_float3(1, 1, -1), 0.5, new DevMetal(make_float3(1, 1, 1),make_float3(0,0,0),0));
        //hitable[3] = new Sphere(make_float3(-1, 1, -1), 0.5, new DevMetal(make_float3(0.2,0.9,0.1),make_float3(0,0,0),.4));
        //hitable[4] = new Sphere(make_float3(0, 3, 0), 1, new DevLambertian(make_float3(0.8,0.8,0.8),make_float3(6, 4, 2)));
        //hitable[5] = new TriangleModel(new DevCoat(make_float3(0.9f, 0.4f, 0.0f), make_float3(0, 0, 0), 0), models[0]);
        hitable[0] = new TriangleModel(new DevLambertian(make_float3(1, 1, 1), make_float3(0, 0, 0)), models[0]);
        world = new HitableList(hitable,1);
    }
}

__global__ void DeleteOnDevice(Hitable** hitable)
{
    delete[](*hitable);
    hitable = nullptr;
}

bool g_bFirstTime = true;

void SetEnvironment(cudaSurfaceObject_t envmap)
{
    cu_surfenvmap = envmap;
}

void HostCreateScene(Suika::Scene& scene)
{
    Suika::CudaTriangleModel* d_model = scene.models[0]->GetDeviceVersion();
    checkCudaErrors(cudaGetLastError());

    // Create Scene
    // -------------------------------
    Hitable** hitable = nullptr;
    cudaMalloc((void**)&hitable, 1 * sizeof(Hitable*));

    Suika::CudaTriangleModel** d_models = nullptr;
    cudaMalloc((void**)&d_models, 1 * sizeof(Suika::CudaTriangleModel*));
    cudaMemcpy(&(d_models[0]), &(d_model), sizeof(Suika::CudaTriangleModel*), cudaMemcpyHostToDevice);

    CreateScene << <1, 1 >> > (hitable, d_models);
    cudaDeviceSynchronize();
}

void RunKernel(size_t textureW, size_t textureH, cudaSurfaceObject_t surfaceObject, cudaSurfaceObject_t textureObject,
    cudaStream_t streamToRun, int i, float* paras)
{
    // Create camera
    // -------------------------------
    //Update Camera
    float3 lookfrom = make_float3(paras[0], paras[1], paras[2]);
    float3 lookat = make_float3(paras[3], paras[4], paras[5]);
    float dist_to_focus = 5;
    float aperture = .5;

    DevCamera* tmp = new DevCamera(lookfrom, lookat, make_float3(0, 1, 0), 45, textureW / textureH, aperture, dist_to_focus);
    float animTime = 0.01 * i;
    checkCudaErrors(cudaMemcpyToSymbol(camera, tmp, sizeof(DevCamera), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(iTime, &animTime, sizeof(float),0,cudaMemcpyHostToDevice));

    // Emit main call kernel
    // -------------------------------
    auto unit = 8;
    dim3 threads(unit, unit);
    dim3 grid(iDivUp(textureW, unit), iDivUp(textureH, unit));

    MainCall <<<grid, threads, 0, streamToRun >>> (surfaceObject, textureObject, textureW, textureH, i);
    checkCudaErrors(cudaGetLastError());
}

// ================================================================
// PRT part
// ================================================================
__global__ void PrecomputeTransfer(float* cudaPRTVertices, float* cudaPRTransfer, int order, int num)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;;
    if (tid > num) return;

    float3 pos = make_float3(cudaPRTVertices[6 * tid + 0], cudaPRTVertices[6 * tid + 1], cudaPRTVertices[6 * tid + 2]);
    float3 norm = make_float3(cudaPRTVertices[6 * tid + 3], cudaPRTVertices[6 * tid + 4], cudaPRTVertices[6 * tid + 5]);

    // This is the approach demonstrated in [1] and is useful for arbitrary
    // functions on the sphere that are represented analytically.
    int sample_count = 32;

    float step = M_PI;
    for (int i = 0; i < sample_count; i++)
    {
        for (int j = 0; j < sample_count; j++)
        {
            float phi = (-1 + 2.0 * i / sample_count) * M_PI;
            float theta = (1.0 * j / sample_count) * M_PI;

            float x = sin(theta) * cos(phi);
            float y = sin(theta) * sin(phi);
            float z = cos(theta);

            float3 dir = make_float3(x, y, z);
            dir = normalize(dir);

            //Hit Test
            float func_value = 0;
            Ray ray(pos, dir);
            HitRecord rec;
            if (!(world)->hit(ray, 0.001, 100, rec))
            {
                func_value = fmaxf(dot(dir, norm), 0);
            }

            //evaluate the SH basis functions up to band O, scale them by the
            //function's value and accumulate them over all generated samples
            for (int l = 0; l <= order; l++) {
                for (int m = -l; m <= l; m++) {
                    float sh = EvalHardCodedSH(l, m, dir);
                    cudaPRTransfer[16 * tid + GetIndex(l, m)] += func_value * sh;
                }
            }
        }
    }
    for (int l = 0; l <= order; l++) {
        for (int m = -l; m <= l; m++) {
            cudaPRTransfer[16 * tid + GetIndex(l, m)] /= sample_count * sample_count;
        }
    }
}

void RunPrecomputeTransfer(PRTransfer* prt)
{
    float* cudaPRTransfer;
    float* cudaPRTVertices;

    // copy vertex data to CUDA global memory
    checkCudaErrors(cudaMalloc((void**)&cudaPRTVertices, prt->mVerticesNum * 6 * sizeof(float)));
    checkCudaErrors(cudaMemcpy(cudaPRTVertices, prt->pVertexNormal, prt->mVerticesNum * 6 * sizeof(float), cudaMemcpyHostToDevice))

    // malloc the memory on GPU
    checkCudaErrors(cudaMalloc((void**)&cudaPRTransfer, prt->TransferSize()));

    // Emit main call kernel
    // -------------------------------

    int num = prt->VertexNum();
    dim3 threads(128);
    dim3 grid(iDivUp(num, 128));
    PrecomputeTransfer << <grid, threads >> > (cudaPRTVertices, cudaPRTransfer, 3, num);

    checkCudaErrors(cudaMemcpy(prt->pTransferData, cudaPRTransfer, prt->TransferSize(), cudaMemcpyDeviceToHost));

    // Free all cuda memory
    checkCudaErrors(cudaFree(cudaPRTransfer));
    checkCudaErrors(cudaFree(cudaPRTVertices));

    cudaDeviceSynchronize();
}