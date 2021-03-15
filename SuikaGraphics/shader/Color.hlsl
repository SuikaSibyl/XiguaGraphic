#include "LightingUtils.hlsl"
#define MaxLights 16

cbuffer cbPerObject : register(b0)
{
	float4x4 gWorld;
};

cbuffer cbPass : register(b1)
{
	float4x4 gViewProj;
	float gTime;
	Light gLights[MaxLights];
};

cbuffer cbMaterial : register(b2)
{
    float4 gDiffuseAlbedo;
    float3 gFresnelR0;
    float gRoughness;
	float4 gMatTransform;
};

struct VertexIn
{
	float3 PosL  : POSITION;
    float3 Nomral : NORMAL;
};

struct VertexOut
{
	float4 PosH  : SV_POSITION;
    float4 Color : COLOR;
};

VertexOut VS(VertexIn vin)
{
	VertexOut vout;

	float3 PosW = mul(float4(vin.PosL, 1.0f), gWorld).xyz;
	vout.PosH = mul(float4(PosW, 1.0f), gViewProj);
	
	// Just pass vertex color into the pixel shader.
	float3 color = vin.Nomral/2+float3(.5,.5,.5);
    vout.Color = float4(color, 1.0f);
    
    return vout;
}

float4 PS(VertexOut pin) : SV_Target
{
	float4 finalColor = pin.Color * ((sin(gTime) + 2) / 2);
    return float4(gLights[0].Strength,1.0);
}