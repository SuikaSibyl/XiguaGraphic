#include "RTUtils.cuh"

class DevLambertian: public DevMaterial
{
public:
    __device__ DevLambertian(const float3& a, const float3& e):albedo(a), emission(e){}

    __device__ bool scatter(const Ray& r_in, const HitRecord &rec, float3& attenuation, float3& emission, Ray& scattered, RandInstance& rnd) const
    {
        float3 target = rec.p + rec.normal + rnd.random_in_unit_sphere();
        scattered = Ray(rec.p, target-rec.p);
        attenuation = albedo * 1.5;
        emission = this -> emission;
        attenuation = albedo;
        return (dot(scattered.dir, rec.normal)>0);
    }
    
    float3 albedo;
    float3 emission;
};

class DevMetal: public DevMaterial
{
public:
    __device__ DevMetal(const float3& a, const float3& e, float f):
        albedo(a), emission(e), fuzz(f){}

    __device__ bool scatter(const Ray& r_in, const HitRecord &rec, float3& attenuation, float3& emission, Ray& scattered, RandInstance& rnd) const
    {
        float3 reflected = normalize(reflect(normalize(r_in.dir), rec.normal));
        scattered = Ray(rec.p, reflected + fuzz * rnd.random_in_unit_sphere());
        attenuation = albedo * 1.5;
        emission = this -> emission;
        attenuation = albedo;
        return (dot(scattered.dir, rec.normal)>0);
    }
    
    float3 albedo;
    float3 emission;
    float fuzz;
};

class DevCoat : public DevMaterial
{
public:
    __device__ DevCoat(const float3& a, const float3& e, float f) :
        albedo(a), emission(e), fuzz(f) {}

    __device__ bool scatter(const Ray& r_in, const HitRecord& rec, float3& attenuation, float3& emission, Ray& scattered, RandInstance& rnd) const
    {
        float rouletteRandomFloat = rnd.rand();
        float threshold = 0.2f;
        bool reflectFromSurface = (rouletteRandomFloat < threshold);

        if (reflectFromSurface) { // calculate perfectly specular reflection

            // Ray reflected from the surface. Trace a ray in the reflection direction.
            // TODO: Use Russian roulette instead of simple multipliers! (Selecting between diffuse sample and no sample (absorption) in this case.)
            float3 reflected = normalize(reflect(normalize(r_in.dir), rec.normal));
            scattered = Ray(rec.p, reflected + fuzz * rnd.random_in_unit_sphere());
            attenuation = albedo * 1.5 + make_float3(1, 1, 1);
            attenuation /= 2;
            attenuation = albedo;
            emission = this->emission;
            return (dot(scattered.dir, rec.normal) > 0);
        }

        else {  // calculate perfectly diffuse reflection
            float3 target = rec.p + rec.normal + rnd.random_in_unit_sphere();
            scattered = Ray(rec.p, target - rec.p);
            attenuation = albedo * 1.5;
            emission = this->emission;
            attenuation = albedo;
            return (dot(scattered.dir, rec.normal) > 0);
        }
    }

    float3 albedo;
    float3 emission;
    float fuzz;
};