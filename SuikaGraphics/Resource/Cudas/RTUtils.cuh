#ifndef __RTUTILH__
#define __RTUTILH__

#include "Utility.cuh"
#include "RandUtils.cuh"

class DevMaterial;

class Ray 
{
public:
    __device__ Ray() :
        pos(make_float3(0)), dir(make_float3(0)) {}

    __device__ Ray(float3 ipos, float3 idir) :
        pos(ipos), dir(idir) {}

    __device__ float3 point_at_parameter(float t) const
    {
        return pos + dir * t;
    }

    float3 pos;
    float3 dir;
};

class DevCamera
{
public:
    __device__ DevCamera(){}

    __host__ __device__ DevCamera(unsigned int width, unsigned int height)
    {
        float ratio = 1.0*width/height;
        lower_left_corner = make_float3(-ratio,-1,-1);
        horizontal = make_float3(ratio*2.,0,0);
        vertical = make_float3(0,2,0);
        origin = make_float3(0,0,0);
    }

    __host__ __device__ DevCamera(float3 lookfrom, float3 lookat, float3 vup, float vfov, float aspect, float aperture, float focus_dist)
    {
        lens_radius = aperture / 2;
        float theta = vfov * M_PI / 180;
        float half_height = tan(theta/2);
        float half_width = aspect * half_height;
        origin = lookfrom;
        w = normalize(lookfrom - lookat);
        u = normalize(cross(vup,w));
        v = cross(w,u);
        lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
        horizontal = 2 * half_width * focus_dist * u;
        vertical = 2 * half_height * focus_dist * v;
    }

    __device__ Ray getRay(float s, float t, RandInstance& rnd) const
    {
        float3 rd = lens_radius * rnd.random_in_unit_disk();
        float3 offset = u * rd.x + v * rd.y;
        offset = offset * 0;
        return Ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
    }

    __device__ Ray Dowhatever(float u, float v, float i) const
    {
        return Ray(origin, make_float3(0));
    }

    float3 lower_left_corner;
    float3 horizontal;
    float3 vertical;
    float3 origin;
    float3 u,v,w;

    float lens_radius;
};

__device__ Ray Dowhatever2(DevCamera* cam, float u, float v, int i)
{
    float3 test = cam->lower_left_corner + u * cam->vertical - cam->origin;
    float3 test2 = make_float3(1);
    Ray ray;
    ray.dir = test;
}

struct HitRecord
{
    float t;
    float3 p;
    float3 normal;
    DevMaterial *mat_ptr;
};

class Hitable
{
public:
    __device__ Hitable(DevMaterial* mat):material(mat) {}
    __device__ virtual ~Hitable() {}
    __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const = 0;

    DevMaterial* material;
};

class DevMaterial
{
public:
    __device__ virtual bool scatter(const Ray& r_in, const HitRecord &rec, float3& attenuation, Ray& scattered, RandInstance& seed) const = 0;
};

#endif