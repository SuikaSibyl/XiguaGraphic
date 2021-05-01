#include "RTUtils.cuh"

class DevLambertian: public DevMaterial
{
public:
    __device__ DevLambertian(const float3& a):albedo(a){}

    __device__ bool scatter(const Ray& r_in, const HitRecord &rec, float3& attenuation, Ray& scattered, RandInstance& rnd) const
    {
        float3 target = rec.p + rec.normal + rnd.random_in_unit_sphere();
        scattered = Ray(rec.p, target-rec.p);
        attenuation = albedo;
        return (dot(scattered.dir, rec.normal)>0);
    }
    
    float3 albedo;
};

class DevMetal: public DevMaterial
{
public:
    __device__ DevMetal(const float3& a, float f):albedo(a), fuzz(f){}

    __device__ bool scatter(const Ray& r_in, const HitRecord &rec, float3& attenuation, Ray& scattered, RandInstance& rnd) const
    {
        float3 reflected = normalize(reflect(normalize(r_in.dir), rec.normal));
        scattered = Ray(rec.p, reflected);// +fuzz * rnd.random_in_unit_sphere());
        attenuation = albedo;
        return (dot(scattered.dir, rec.normal)>0);
    }
    
    float3 albedo;
    float fuzz;
};