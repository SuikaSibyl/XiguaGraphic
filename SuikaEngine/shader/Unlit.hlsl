#include "ShaderLibrary/Common.hlsl"

struct VertexOut
{
	float4 PosH  : SV_POSITION;
	float4 Color  : COLOR;
};

VertexOut VS(VertexIn vin)
{
	VertexOut vout;

    MaterialData matData = gMaterialData[materialIndex];
    
	float3 PosW = mul(float4(vin.PosL, 1.0f), gWorld).xyz;
	vout.PosH = mul(float4(PosW, 1.0f), gViewProj);
    vout.Color = matData.gDiffuseAlbedo;

    return vout;
}

float4 PS(VertexOut pin) : SV_Target
{
    return pin.Color;
}