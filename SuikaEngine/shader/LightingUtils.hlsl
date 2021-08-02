#include "ShaderLibrary/Common.hlsl"

float CalcAttenuation(float d, float falloffStart, float falloffEnd)
{
    // Linear falloff.
    return saturate((falloffEnd-d) / (falloffEnd - falloffStart)); 
}

// Schlick gives an approximation to Fresnel reflectance
// (see pg. 233 “Real-Time Rendering 3rd Ed.”).
// R0 = ( (n-1)/(n+1) )^2, where n is the index of refraction.
float3 SchlickFresnel(float3 R0, float3 normal, float3 lightVec)
{
    float cosIncidentAngle = saturate(dot(normal, lightVec));

    float f0 = 1.0f - cosIncidentAngle;
    float3 reflectPercent = R0 + (1.0f - R0)* (f0*f0*f0*f0*f0);

    return reflectPercent;
}

struct Material 
{
    float4 DiffuseAlbedo;
    float3 FresnelR0;
    // Shininess is inverse of roughness: Shininess = 1- roughness.
    float Shininess; 
};

float3 BlinnPhong(float3 lightStrength, float3 lightVec, float3 normal, float3 toEye, Material mat)
{
    // Derive m from the shininess, which is derived from the roughness.
    const float m = mat.Shininess * 256.0f;
    float3 halfVec = normalize(toEye + lightVec);
    float roughnessFactor = (m + 8.0f)*pow(max(dot(halfVec, normal), 0.0f), m) / 8.0f;
    float3 fresnelFactor = SchlickFresnel(mat.FresnelR0, halfVec, lightVec);

    // Our spec formula goes outside [0,1] range, but we are doing 
    // LDR rendering. So scale it down a bit.
    float3 specAlbedo = fresnelFactor * roughnessFactor;//镜面反射反照率=菲尼尔因子*粗糙度因子
    specAlbedo = specAlbedo / (specAlbedo + float3(1,1,1));
    float3 diff_Spec = lightStrength * (mat.DiffuseAlbedo.rgb + specAlbedo); //漫反射+高光反射=入射光量*总的反照率
    return diff_Spec;
}

float3 ComputerDirectionalLight(Light light, Material mat, float3 normal, float3 toEye)
{
    float3 lightVec = -light.Direction; //光向量和光源指向顶点的向量相反
    float3 lightStrength = max(dot(normal, lightVec), 0.0f) * light.Strength; //方向光单位面积上的辐照度
    
    //平行光的漫反射+高光反射
    return BlinnPhong(lightStrength, lightVec, normal, toEye, mat);
}

float CalcShadowFactor(float4 shadowPosH, Texture2D shadowmap)
{
    // Complete projection by doing division by w.
    shadowPosH.xyz /= shadowPosH.w;

    // Depth in NDC space.
    float depth = shadowPosH.z;

    uint width, height, numMips;
    shadowmap.GetDimensions(0, width, height, numMips);

    // Texel size.
    float dx = 1.0f / (float)width;

    float percentLit = 0.0f;
    const float2 offsets[9] =
    {
        float2(-dx,  -dx), float2(0.0f,  -dx), float2(dx,  -dx),
        float2(-dx, 0.0f), float2(0.0f, 0.0f), float2(dx, 0.0f),
        float2(-dx,  +dx), float2(0.0f,  +dx), float2(dx,  +dx)
    };

    [unroll]
    for(int i = 0; i < 9; ++i)
    {
        percentLit += shadowmap.SampleCmpLevelZero(gsamShadow,
            shadowPosH.xy + offsets[i], depth).r;
    }
    
    return percentLit / 9.0f;
}
