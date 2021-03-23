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

	float3 PosW = mul(float4(vin.PosL, 1.0f), gWorld).xyz;
	vout.WorldPos = PosW;
	float3x3 world = (float3x3)gWorld;
    vout.WorldNormal = mul(vin.Normal, (float3x3)gWorld).xyz;
	vout.PosH = mul(float4(PosW, 1.0f), gViewProj);
	
	// Just pass vertex color into the pixel shader.
    vout.uv = vin.TexC;
    //计算UV坐标的静态偏移（相当于MAX中编辑UV）
    float4 texCoord = mul(float4(vin.TexC, 0.0f, 1.0f), gTexTrans);
    vout.uv = texCoord.xy;
    vout.Color = matData.gDiffuseAlbedo;
    return vout;
}

float4 PS(VertexOut pin) : SV_Target
{
    MaterialData matData = gMaterialData[materialIndex];
    float4 dark = float4(0.117,0.117,0.117,1 );

    float4 albedo = gDiffuseMap[matData.gDiffuseMapIndex].Sample(gSamPointWrap, pin.uv);
    
    float3 N = normalize(pin.WorldNormal);
    float3 V = normalize(gEyePosW - pin.WorldPos);

    float metallic  =   0;
    float roughness =   0.2;
    float ao      =   0;

    float4 skyCol = gCubeMap.Sample(gSamPointWrap, -V);

    float3 Lo = float3(0.0,0.0,0.0);

    for(int i=0;i<4;i++)
    {
        float3 L = normalize(gLights[i].Position - pin.WorldPos);
        float3 H = normalize(V + L);
        
        float distance    = length(gLights[i].Position - pin.WorldPos);
        float attenuation = 1.0;// / (distance * distance);
        float3 radiance   = gLights[i].Strength * attenuation; 
        
        float3 F  = FresnelSchlick(metallic, albedo.rgb, H, V);
        float NDF = DistributionGGX(N, H, roughness);       
        float G   = GeometrySmith(N, V, L, roughness);
        
        float3 nominator    = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001; 
        float3 specular     = nominator / denominator;  

        float3 kS = F;
        float3 kD = float3(1.0, 1.0, 1.0) - kS;

        kD *= 1.0 - metallic;  

        float NdotL = max(dot(N, L), 0.0);
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;
    }
    
    float3 ambient = float3(0.03, 0.03, 0.03) * albedo * ao;
    float3 color   = ambient + Lo;

#ifdef ALPHA_TEST
    clip(albedo.a - 0.1f);
#endif
    // albedo.a = albedo.a*0.7;
    // float3 worldNormal = normalize(pin.WorldNormal);
    // float3 worldView = normalize(gEyePosW - pin.WorldPos);
    // float3 worldPosToEye = gEyePosW - pin.WorldPos;
    // float distPosToEye = length(gEyePosW - pin.WorldPos);
    
    // // return finalCol;
	// float4 finalColor = float4(gLights[0].Strength,1.0) * ((sin(gTime) + 2) / 2);
    
    // float s = saturate((distPosToEye) * 0.001f);
    // float4 finalCol = lerp(diffuseAlbedo, dark, s);

    return float4(ReinhardHDR(albedo), 1.0);
}