#ifndef SPHEREH
#define SPHEREH

#include "RTUtils.cuh"

class Sphere:public Hitable
{
public:
    __device__ Sphere(DevMaterial* mat):Hitable(mat){}
    __device__ ~Sphere(){}
    __device__ Sphere(float3 cen, float r, DevMaterial* mat):Hitable(mat),center(cen),radius(r){};
    __device__ bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const;

    float3 center;
    float radius;
};

__device__ bool Sphere::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const
{   
    float3 oc = r.pos - center;
    float a = dot(r.dir, r.dir);
    float b = dot(oc, r.dir);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - a * c;

    if(discriminant>0)
    {
        float temp = (-b - sqrtf(discriminant)) / a;
        if(temp < t_max && temp > t_min)
        {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = normalize((rec.p-center)/radius);
            rec.mat_ptr = material;
            return true;
        }
        temp  = (-b + sqrtf(discriminant))/a;
        if(temp < t_max && temp > t_min)
        {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = normalize((rec.p-center)/radius);
            rec.mat_ptr = material;
            return true;
        }
    }
    return false;
}
#endif