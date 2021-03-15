struct Light 
{
    float3 Strength;
    float FalloffStart; // point/spot light only
    float3 Direction; // directional/spot light only
    float FalloffEnd; // point/spot light only
    float3 Position; // point light only
    float SpotPower; // spot light only 
};

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
    float3 FresnelR0; // Shininess is inverse of roughness: Shininess = 1- roughness.
    float Shininess; 
};

// float3 BlinnPhong(float3 lightStrength, float3 lightVec, float3 normal, float3 toEye, Material mat)
// {
//     // Derive m from the shininess, which is derived from the roughness.
//     const float m = mat.Shininess * 256.0f;
//     float3 halfVec = normalize(toEye + lightVec);
//     float roughnessFactor = (m + 8.0f)*pow(max(dot(halfVec, normal), 0.0f), m) / 8.0f;
//     float3 fresnelFactor = SchlickFresnel(mat.FresnelR0, halfVec, lightVec);
//     // Our spec formula goes outside [0,1] range, but we are doing 
//     // LDR rendering. So scale it down a bit.
//     specAlbedo = specAlbedo / (specAlbedo + 1.0f);
//     return (mat.DiffuseAlbedo.rgb + specAlbedo) * lightStrength;
// }