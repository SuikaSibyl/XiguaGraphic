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
    float2 uv = float2(pin.uv.x,1-pin.uv.y);
    float4 albedo = gDiffuseMap[11].Sample(gSamAnisotropicWarp, uv);
    float4 normal = gDiffuseMap[12].Sample(gSamAnisotropicWarp, uv);
    float depth = gDiffuseMap[14].Sample(gSamAnisotropicWarp, uv).rrr;
    
    return normal;
}