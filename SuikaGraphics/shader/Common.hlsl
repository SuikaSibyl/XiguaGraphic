#include "LightingUtils.hlsl"
#include "PBR.hlsl"
#define MaxLights 16

struct MaterialData
{
    float4 gDiffuseAlbedo; //材质反照率
    float3 gFresnelR0; //RF(0)值，即材质的反射属性
    float gRoughness; //材质的粗糙度
    float4x4 gMatTransform; //UV动画变换矩阵
    uint gDiffuseMapIndex;//纹理数组索引
    uint gMatPad0;
    uint gMatPad1;
    uint gMatPad2;
};

TextureCube gCubeMap : register(t0);
// An array of textures, which is only supported in shader model 5.1+.
// Unlike Texture2DArray, the textures in this array can be different
// sizes and formats, making it more flexible than texture arrays.
Texture2D gDiffuseMap[4] : register(t1);//所有漫反射贴图

// Put in space1, so the texture array does not overlap with these.
// The texture array above will occupy registers t0, t1, …, t6 in
// space0.
//材质数据的结构化缓冲区，使用t0的space1空间
StructuredBuffer<MaterialData> gMaterialData : register(t0, space1);

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
	float4x4 gTexTrans;
    uint materialIndex;
    uint pad1;
    uint pad2;
    uint pad3;
};

cbuffer cbPass : register(b1)
{
	float4x4 gViewProj;
	float3 gEyePosW;
	float gTime;
	float4 gAmbientLight;
	Light gLights[MaxLights];
};

struct VertexIn
{
	float3 PosL  : POSITION;
    float3 Normal : NORMAL;
    float2 TexC : TEXCOORD;
};