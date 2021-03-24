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

float4 getCubeMap(float3 direction)
{
    float2 uv = SampleSphericalMap(normalize(direction)); // make sure to normalize localPos
    return gDiffuseMap[6].Sample(gSamPointWrap, uv);
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
        float attenuation = 1.0 / (distance * distance);
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
    
    float3 F0 = float3(0.04,0.04,0.04); 
    float3 Ks = fresnelSchlick(max(dot(N, V), 0.0), F0);
    float3 Kd = 1.0 - Ks;
    float3 irradiance = getCubeMap(N);
    float3 diffuse    = irradiance * albedo;
    float3 ambient    = (Kd * diffuse) * 1; 
    // float3 ambient = float3(0.03, 0.03, 0.03) * albedo * ao;
    float3 color   = ambient + Lo;

#ifdef ALPHA_TEST
    clip(albedo.a - 0.1f);
#endif

    return float4(ReinhardHDR(color), 0.7);
}