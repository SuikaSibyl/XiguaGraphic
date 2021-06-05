#include "Common.hlsl"
#include "Random.hlsl"
#include "PBR.hlsl"

struct VertexOut
{
	float4 PosH  : SV_POSITION;
    float3 WorldPos : POSITION;
    float3 WorldNormal : NORMAL;
    float2 uv: TEXCOORD;
	float4 Color  : COLOR;
};

VertexOut VS(VertexIn vin)
{
	VertexOut vout;

    //使用结构化缓冲区数组（结构化缓冲区是由若干类型数据所组成的数组）
    MaterialData matData = gMaterialData[materialIndex];

	vout.WorldPos = vin.PosL;
	vout.PosH = float4(vin.PosL,1);
    vout.uv = vin.TexC;
    // vout.uv = float2(vin.TexC.x, 1-vin.TexC.y);
    vout.uv = float2(vin.TexC.x, vin.TexC.y);
    
    return vout;
}

float4 PS(VertexOut pin) : SV_Target
{
    MaterialData matData = gMaterialData[materialIndex];
    uint index = matData.gDiffuseMapIndex;
    float4 albedo = gDiffuseMap[8].Sample(gSamAnisotropicWarp, pin.uv);
    // float3 depth = albedo.rrr;
    // return float4(depth,1.0);
    return albedo;

    // float NdotV = pin.uv.x;
    // float roughness = pin.uv.y;

    // float3 V;
    // V.x = sqrt(1.0 - NdotV*NdotV);
    // V.y = 0.0;
    // V.z = NdotV;

    // float A = 0.0;
    // float B = 0.0;

    // float3 N = float3(0.0, 0.0, 1.0);

    // // const uint SAMPLE_COUNT = 1024;
    // [unroll(1024)]
    // for(uint i = 0; i < 1024; ++i)
    // {
    //     float2 Xi = Hammersley(i, 1024);
    //     float3 H  = ImportanceSampleGGX(Xi, N, roughness);
    //     float3 L  = normalize(2.0 * dot(V, H) * H - V);

    //     float NdotL = max(L.z, 0.0);
    //     float NdotH = max(H.z, 0.0);
    //     float VdotH = max(dot(V, H), 0.0);

    //     if(NdotL > 0.0)
    //     {
    //         float G = GeometrySmith(N, V, L, roughness,true);
    //         float G_Vis = (G * VdotH) / (NdotH * NdotV);
    //         float Fc = pow(1.0 - VdotH, 5.0);

    //         A += (1.0 - Fc) * G_Vis;
    //         B += Fc * G_Vis;
    //     }
    // }
    // A /= float(1024);
    // B /= float(1024);
    // return float4(A, B,0,1);
}