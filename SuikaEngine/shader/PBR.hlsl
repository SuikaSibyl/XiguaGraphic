#ifndef __SHADER_PBR__
#define __SHADER_PBR__

#include "Utility.hlsl"
#include "LightingUtils.hlsl"

// [D] Normal Distribution Function::Trowbridge-Reitz GGX
// ==================================================
float DistributionGGX(float3 N, float3 H, float roughness)
{
    float a      = roughness*roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom    = a2;
    float denom  = (NdotH2 * (a2 - 1.0) + 1.0);
    denom        = PI * denom * denom;

    return nom / denom;
}

// [F] Fresnel Equation::Fresnel-Schlick Approximation
// ==================================================
float3 fresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}  

float3 FresnelSchlick(float metallic, float3 albedo, float3 H, float3 V)
{
    float3 F0 = float3(0.04,0.04,0.04); 
    F0      = LerpFloat3(F0, albedo, metallic);

    return fresnelSchlick(max(dot(H, V), 0.0), F0);
}  

// [G] Geometry Function::Smith’s Schlick-GGX
// ==================================================
float GeometrySchlickGGX(float NdotV, float roughness, bool isIBL)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    if(isIBL)
    {
        float a = roughness;
        k = (a * a) / 2.0;
    }
    
    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

float GeometrySmith(float3 N, float3 V, float3 L, float roughness, bool isIBL)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    
    float a      = roughness*roughness;
    float ggx1 = GeometrySchlickGGX(NdotV, a, isIBL);
    float ggx2 = GeometrySchlickGGX(NdotL, a, isIBL);

    return ggx1 * ggx2;
}

//
//
float4 CookTorrance(float metallic,float roughness, float ao, float3 albedo, float3 N, float3 H, float3 V, float3 L)
{
    float3 Lo = float3(0.0, 0.0, 0.0);
    
    float3 F  = FresnelSchlick(metallic, albedo, H, V);
    float NDF = DistributionGGX(N, H, roughness);       
    float G   = GeometrySmith(N, V, L, roughness, false);
    
    float3 nominator    = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001; 
    float3 specular     = nominator / denominator;  

    float3 kS = F;
    float3 kD = float3(1.0, 1.0, 1.0) - kS;

    kD *= 1.0 - metallic;  

    float NdotL = max(dot(N, L), 0.0);        
    // Lo += (kD * albedo / PI + specular) * radiance * NdotL;

    float3 ambient = float3(0.03, 0.03, 0.03) * albedo * ao;
    float3 color   = ambient + Lo;

    return float4(ReinhardHDR(color), 1.0);
}

#endif