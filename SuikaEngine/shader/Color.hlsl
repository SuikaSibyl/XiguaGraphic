#include "Common.hlsl"
#include "PBR.hlsl"
#include "PRT.hlsl"

struct VertexOut
{
	float4 PosH  : SV_POSITION;
    float3 WorldPos : POSITION0;
    float4 ShadowPosH : POSITION1;
    float3 WorldNormal : NORMAL;
    float2 uv: TEXCOORD;
	float4 Color  : COLOR;
    float3 AO : POSITION2;
};

VertexOut VS(VertexIn vin)
{
	VertexOut vout;

    //使用结构化缓冲区数组（结构化缓冲区是由若干类型数据所组成的数组）
    MaterialData matData = gMaterialData[materialIndex];

	float3 PosW = mul(float4(vin.PosL, 1.0f), gWorld).xyz;
	vout.WorldPos = PosW;
    vout.WorldNormal = mul(vin.Normal, (float3x3)gWorld).xyz;
	vout.PosH = mul(float4(PosW, 1.0f), gViewProj);
	
	// Just pass vertex color into the pixel shader.
    vout.uv = vin.TexC;
    //计算UV坐标的静态偏移（相当于MAX中编辑UV）
    float4 texCoord = mul(float4(vin.TexC, 0.0f, 1.0f), gTexTrans);
    vout.uv = texCoord.xy;
    vout.Color = matData.gDiffuseAlbedo;
    // Generate projective tex-coords to project shadow map onto scene.
    vout.ShadowPosH =  mul(float4(PosW, 1.0f), gShadowTransform);//mul(PosW, gShadowTransform);
    float3 ver[3] = {float3(5,0,0), float3(0,5,0), float3(0,0,5)};
    float4x4 LightCoeff = ComputeCoefficients(PosW, ver, 3);
    
    float AO = vin.Transfer[0][0];// * LightCoeff[0][0] +
            // vin.Transfer[0][1] * LightCoeff[0][1] -
            // vin.Transfer[0][2] * LightCoeff[0][2] -
            // vin.Transfer[0][3] * LightCoeff[0][3];// +
            // vin.Transfer[1][0] * LightCoeff[1][0] +
            // vin.Transfer[1][1] * LightCoeff[1][1] +
            // vin.Transfer[1][2] * LightCoeff[1][2] +
            // vin.Transfer[1][3] * LightCoeff[1][3] +
            // vin.Transfer[2][0] * LightCoeff[2][0] +
            // vin.Transfer[2][1] * LightCoeff[2][1] +
            // vin.Transfer[2][2] * LightCoeff[2][2] +
            // vin.Transfer[2][3] * LightCoeff[2][3] +
            // vin.Transfer[3][0] * LightCoeff[3][0] +
            // vin.Transfer[3][1] * LightCoeff[3][1] +
            // vin.Transfer[3][2] * LightCoeff[3][2] +
            // vin.Transfer[3][3] * LightCoeff[3][3];

    vout.AO= 5 * float3( AO,AO,AO);
    return vout;
}

float3 SampleCubeMapArray(TextureCubeArray cubearray, float3 direction, float layer)
{
    float layerf = floor(layer);
    float layerc = layerf+1;
    float3 sample1 = cubearray.Sample(gSamLinearWarp, float4(direction,layerf));
    float3 sample2 = cubearray.Sample(gSamLinearWarp, float4(direction,layerc));
    return LerpFloat3(sample1,sample2, layer-layerf);
}

float4 PS(VertexOut pin) : SV_Target
{
    MaterialData matData = gMaterialData[materialIndex];
    float4 dark = float4(0.117,0.117,0.117,1 );

    // Only the first light casts a shadow.
    // float3 shadowFactor = float3(1.0f, 1.0f, 1.0f);
    float shadowFactor = CalcShadowFactor(pin.ShadowPosH, gDiffuseMap[12]);

    uint index = matData.gDiffuseMapIndex;
    float4 albedo = matData.gDiffuseAlbedo;
    if(index!=-1)
    {
        albedo = gDiffuseMap[index].Sample(gSamPointWrap, pin.uv);
    }
    
    float3 N = normalize(pin.WorldNormal);
    float3 V = normalize(gEyePosW - pin.WorldPos);

    float metallic  =   matData.gMetalness;
    float roughness =   matData.gRoughness;
    float ao      =   1;

    float3 Lo = float3(0.0,0.0,0.0);

    float3 F  = FresnelSchlick(metallic, albedo.rgb, N, V);
    float3 kS = F;
    float3 kD = float3(1.0, 1.0, 1.0) - kS;

    for(int i=0;i<1;i++)
    {
        // float3 L = normalize(gLights[i].Position - pin.WorldPos);
        float3 L = -normalize(gLights[i].Direction);
        float3 H = normalize(V + L);
        
        // float distance    = length(gLights[i].Position - pin.WorldPos);
        // float attenuation = 1.0 / (distance * distance);
        float   NdotL       =   max(dot(N, L), 0.0);
        float3  radiance    =   gLights[i].Strength * NdotL * shadowFactor; 
        
        float NDF = DistributionGGX(N, H, roughness);       
        float G   = GeometrySmith(N, V, L, roughness, false);
        
        float3 nominator    = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001; 
        float3 specular     = nominator / denominator;  

        kD *= 1.0 - metallic;  
        
        Lo += (kD * albedo / PI + specular) * radiance;
    }
    
    float3 R = reflect(-V, N);   
    const float MAX_REFLECTION_LOD = 5.0;
    float3 prefilteredColor = GammaToLinearSpace(SampleCubeMapArray(gCubeMapArray, R,  roughness * MAX_REFLECTION_LOD));
    float2 envBRDF  = gDiffuseMap[0].Sample(gSamPointWrap, float2(max(dot(N, V), 0.0), roughness)).rg;
    float3 specular = prefilteredColor * (F * envBRDF.x + envBRDF.y);

    float3 irradiance = GammaToLinearSpace(gCubeMap.Sample(gSamPointWrap, N));
    float3 diffuse = irradiance * albedo;
    float3 ambient = (diffuse * kD + specular) * ao;

    float3 color   = ambient + Lo + matData.gEmission;  

// #ifdef ALPHA_TEST
//     clip(albedo.a - 0.1f);
// #endif

    // color*=3;
    // color = color / (color + float3(1.0,1.0,1.0));
    color = pow(color, float3(1.0/2.2,1.0/2.2,1.0/2.2)); 
    
    return float4(pin.AO, 1);
    // return float4(radiance, 0.7);
}