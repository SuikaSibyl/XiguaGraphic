#include "Common.hlsl"

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
    
    return vout;
}

float4 PS(VertexOut pin) : SV_Target
{
    MaterialData matData = gMaterialData[materialIndex];
    uint index = matData.gDiffuseMapIndex;
    float4 albedo = gDiffuseMap[index].Sample(gSamPointWrap, pin.uv);

    return albedo;
}