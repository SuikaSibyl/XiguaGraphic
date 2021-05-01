#ifndef __RANDUTILH__
#define __RANDUTILH__

#include "Utility.cuh"

#define kMultiplier1 1140671485 
#define kMultiplier2 65793

#define kIncrement1	12820163 
#define kIncrement2 4282663

#define kModulo1 16777216
#define kModulo2 8388608

__device__ float hash(float seed)
{
    return fract(sin(seed)*43758.5453);
}

__device__ float fragRandOffset( float2 fragCoord , float a)
{
	return (hash( dot( fragCoord, make_float2(12.9898, 78.233) ) + 1113.1*a ));
}

class RandInstance
{
public:
    __device__ RandInstance(float2 fragCoord, const float iTime)
    {
        fragCoord += fract(make_float2(float(iTime*383.0),(iTime*787.0))/953.0)*953.0;
        g_CurrentRand1 = int(fragRandOffset(fragCoord,1.0)*float(kMultiplier1));
        g_CurrentRand2 = int(fragRandOffset(fragCoord,1.0)*float(kMultiplier2));
    }

    __device__ float rand()
    {
        int mul1 = kMultiplier1;
        int mul2 = kMultiplier2;
        int inc1 = kIncrement1;
        int inc2 = kIncrement2;
        float mod1 = float(kModulo1);
        float mod2 = float(kModulo2); 

        // move both internal generators on to their next number
        g_CurrentRand1 = int(mod(float(g_CurrentRand1*mul1 + inc1),mod1));
        g_CurrentRand2 = int(mod(float(g_CurrentRand2*mul2 + inc2),mod2));
        
        // combine them to get something that is hopefully more random
        return fract(float(g_CurrentRand1 - g_CurrentRand2)/mod1);
    }

    __device__ float3 random_in_unit_sphere()
    {	
        float phi = (rand()*2.0-1.0)*M_PI;
        float costheta = rand()*2.0-1.0;
        float u = rand();
        
        //float theta = acos( costheta );
        float sintheta = sqrt(1.0-costheta*costheta);
        float r = powf(u ,1.0/3.0);
        
        float x = r * sintheta * cos( phi );
        float y = r * sintheta * sin( phi );
        float z = r * costheta;
        
        return make_float3(x,y,z);
    }

    __device__ float3 random_in_unit_disk()
    {
        float theta = rand()*2.0*M_PI;
        float r = sqrt(rand());
    
        return make_float3(r*sin(theta),r*cos(theta),0.0);
    }

    int g_CurrentRand1;
    int g_CurrentRand2;
};

#endif