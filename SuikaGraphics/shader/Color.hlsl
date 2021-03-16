#include "LightingUtils.hlsl"
#define MaxLights 16

Texture2D gDiffuseMap : register(t0);//所有漫反射贴图

//6个不同类型的采样器
SamplerState gSamPointWrap : register(s0);
SamplerState gSamPointClamp : register(s1);
SamplerState gSamLinearWarp : register(s2);
SamplerState gSamLinearClamp : register(s3);
SamplerState gSamAnisotropicWarp : register(s4);
SamplerState gSamAnisotropicClamp : register(s5);

cbuffer cbPerObject : register(b0)
{
	float4x4 gWorld;
};

cbuffer cbPass : register(b1)
{
	float4x4 gViewProj;
	float3 gEyePosW;
	float gTime;
	float4 gAmbientLight;
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
    float3 Normal : NORMAL;
    float2 TexC : TEXCOORD;
};

struct VertexOut
{
	float4 PosH  : SV_POSITION;
    float3 WorldPos : POSITION;
    float3 WorldNormal : NORMAL;
    float2 uv: TEXCOORD;
};

VertexOut VS(VertexIn vin)
{
	VertexOut vout;

	float3 PosW = mul(float4(vin.PosL, 1.0f), gWorld).xyz;
	vout.WorldPos = PosW;
	float3x3 world = (float3x3)gWorld;
    vout.WorldNormal = mul(vin.Normal, (float3x3)gWorld).xyz;
	vout.PosH = mul(float4(PosW, 1.0f), gViewProj);
	
	// Just pass vertex color into the pixel shader.
	vout.WorldNormal = vin.Normal/2+float3(.5,.5,.5);
    vout.uv = vin.TexC;
    return vout;
}

float4 PS(VertexOut pin) : SV_Target
{
    float4 diffuseAlbedo = gDiffuseMap.Sample(gSamPointWrap, pin.uv) * gDiffuseAlbedo;
    
    float3 worldNormal = normalize(pin.WorldNormal);
    float3 worldView = normalize(gEyePosW - pin.WorldPos);
    
    Material mat = { float4(1, pin.uv, 1), gFresnelR0, gRoughness };
	 float3 shadowFactor = 1.0f;//暂时使用1.0，不对计算产生影响
    //直接光照
    float4 directLight = float4(ComputerDirectionalLight(gLights[0], mat, worldNormal, worldView),1);
    // //环境光照
    // float4 ambient = gAmbientLight * gDiffuseAlbedo;
	
    // float4 finalCol = ambient + directLight;
    // finalCol.a = gDiffuseAlbedo.a;
    
    // return finalCol;
	float4 finalColor = float4(gLights[0].Strength,1.0) * ((sin(gTime) + 2) / 2);
    return diffuseAlbedo;
}