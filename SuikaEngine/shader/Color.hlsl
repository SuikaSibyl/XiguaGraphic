#include "Common.hlsl"
#include "PBR.hlsl"
#include "PRT.hlsl"

#define MODE_TRANSLUCENT
// #define MODE_OPAQUE
// #define MODE_MANYLIGHT

struct VertexOut
{
	float4 PosH  : SV_POSITION;
    float3 WorldPos : POSITION0;
    float4 ShadowPosH : POSITION1;
    float3 WorldNormal : NORMAL;
    float3 ViewNormal : NORMAL1;
    float2 uv: TEXCOORD;
	float4 Color  : COLOR;
    float4 AO : COLOR2;
    float type: PSIZE;
};

float SHMulEnv(float4x4 a, float4x4 b)
{
    // n = 0
    float res = a[0][0] * b[0][0] +
    // n = 1
               a[0][1] * b[0][1] +
               a[0][2] * b[0][2] +
               a[0][3] * b[0][3]+
    // n = 2
            a[1][0] * b[1][0] +
            a[1][1] * b[1][1] +
            a[1][2] * b[1][2] +
            a[1][3] * b[1][3] +
            a[2][0] * b[2][0] +
    // n = 3
            a[2][1] * b[2][1] +
            a[2][2] * b[2][2] +
            a[2][3] * b[2][3] +
            a[3][0] * b[3][0] +
            a[3][1] * b[3][1] +
            a[3][2] * b[3][2] +
            a[3][3] * b[3][3];

    return res;
}

float SHMulPoly(float4x4 a, float4x4 b)
{
    // n = 0
    float res = a[0][0] * b[0][0] +
    // n = 1
            a[1][0] * b[0][1] +
            a[2][0] * b[0][2] +
            a[3][0] * b[0][3];// +
    // n = 2
            a[0][1] * b[1][0] +
            a[1][1] * b[1][1] +
            a[2][1] * b[1][2] +
            a[3][1] * b[1][3] +
            a[0][2] * b[2][0] +
    // n = 3
            a[1][2] * b[2][1] +
            a[2][2] * b[2][2] +
            a[3][2] * b[2][3] +
            a[0][3] * b[3][0] +
            a[1][3] * b[3][1] +
            a[2][3] * b[3][2] +
            a[3][3] * b[3][3];

            return res;
}

float SHMulPolyLow(float4x4 a, float4x4 b)
{
    // n = 0
    float res = a[0][0] * b[0][0] +
    // n = 1
            a[1][0] * b[0][1] +
            a[2][0] * b[0][2] +
            a[3][0] * b[0][3];

            return res;
}

VertexOut VS(VertexIn vin)
{
    // Hard coded environment map SH
    float4x4 env_r = 
    {
        0.0893863961, 0.0252154265, 0.00724511826, -0.0125375418,
        0.0110503519, 0.00140159857, -0.0226063933, 0.000372407550,
        0.0505704880, -0.000926197856, 0.00141673617, 0.00214727456,
        -0.00147280795, 0.0248643998, 0.00743547687, -0.0130441217
    };
    float4x4 env_g = 
    {
        0.0724972561, -0.0210544541, 0.00932176691, -0.0286795162,
        0.0143158725, -0.00498958258, -0.0220207255, -0.00895528495,
        0.0477836616, -0.0308262501, 0.00416620541, -0.00207177131,
        0.00169866264, 0.0225809347, 0.00247310218, -0.0194749162
    };
    float4x4 env_b = 
    {
        0.0998886228, -0.0648806915, 0.0167694464, -0.0330084115,
        0.0205197018, -0.0135007529, -0.0269548111, -0.0164335147,
        0.0504177511, -0.0627221614, 0.00669931155, -0.00647997810,
        0.00500204507, 0.0190090276, -0.000605879235, -0.0156639982
    };
	VertexOut vout;

    //使用结构化缓冲区数组（结构化缓冲区是由若干类型数据所组成的数组）
    MaterialData matData = gMaterialData[materialIndex];
	float3 PosW = mul(float4(vin.PosL, 1.0f), gWorld).xyz;
	vout.WorldPos = PosW;
    vout.WorldNormal = mul(vin.Normal, (float3x3)gWorld).xyz;
    vout.ViewNormal = mul(vout.WorldNormal, (float3x3)(gViewInverse)).xyz;
	vout.PosH = mul(float4(PosW, 1.0f), gViewProj);
    
	// Just pass vertex color into the pixel shader.
    vout.uv = vin.TexC;
    //计算UV坐标的静态偏移（相当于MAX中编辑UV）
    float4 texCoord = mul(float4(vin.TexC, 0.0f, 1.0f), gTexTrans);
    vout.uv = texCoord.xy;
    vout.Color = matData.gDiffuseAlbedo;

    uint type = matData.gMatType;
    if(type==1)
    {
        vout.type = 1;
        vout.Color = float4(vin.Transfer[0][0],vin.Transfer[0][1],vin.Transfer[0][2], 0);
        return vout;
    }
	
    vout.type = 0;

    // Generate projective tex-coords to project shadow map onto scene.
    vout.ShadowPosH =  mul(float4(PosW, 1.0f), gShadowTransform);//mul(PosW, gShadowTransform);
    // float3 ver[3] = {float3(-2,2,-2), float3(-2,-2,-2), float3(-2,2,2)};
    // float4x4 ao = ComputeCoefficients(PosW, ver, 3);
    MyStruct LightCoeff = PositionInterpo(PosW, gPolygonSHData);

    // =========================================
    // Opaque Light Weights
    // =========================================
#ifdef MODE_OPAQUE
    float r = SHMulEnv(vin.Transfer, env_r) * 2 * 3.14 + SHMulPoly(vin.Transfer, LightCoeff.SHCoeff_r) * 6;
    float g = SHMulEnv(vin.Transfer, env_g) * 2 * 3.14 + SHMulPoly(vin.Transfer, LightCoeff.SHCoeff_g) * 6;
    float b = SHMulEnv(vin.Transfer, env_b) * 2 * 3.14 + SHMulPoly(vin.Transfer, LightCoeff.SHCoeff_b) * 6;
    r*=2.5;
    g*=2.5;
    b*=2.5;
#endif

    // // =========================================
    // // Translucent Weights
    // // =========================================
#ifdef MODE_TRANSLUCENT
    float r = SHMulEnv(vin.Transfer, env_r) * 2 * 3.14 + SHMulPolyLow(vin.Transfer, LightCoeff.SHCoeff_r) * 1;
    float g = SHMulEnv(vin.Transfer, env_g) * 2 * 3.14 + SHMulPolyLow(vin.Transfer, LightCoeff.SHCoeff_g) * 1;
    float b = SHMulEnv(vin.Transfer, env_b) * 2 * 3.14 + SHMulPolyLow(vin.Transfer, LightCoeff.SHCoeff_b) * 1;

    r += SHMulPolyLow(vin.Thiness, LightCoeff.SHCoeff_r) * 2;
    g += SHMulPolyLow(vin.Thiness, LightCoeff.SHCoeff_g) * 2;
    b += SHMulPolyLow(vin.Thiness, LightCoeff.SHCoeff_b) * 2;
    
    r*=4;
    g*=4;
    b*=4;
#endif
    // // =========================================

    // // =========================================
    // // Many Light Weights
    // // =========================================
#ifdef MODE_MANYLIGHT
    float r = SHMulEnv(vin.Transfer, env_r) * 2 * 3.14 + SHMulPoly(vin.Transfer, LightCoeff.SHCoeff_r) * 8;
    float g = SHMulEnv(vin.Transfer, env_g) * 2 * 3.14 + SHMulPoly(vin.Transfer, LightCoeff.SHCoeff_g) * 8;
    float b = SHMulEnv(vin.Transfer, env_b) * 2 * 3.14 + SHMulPoly(vin.Transfer, LightCoeff.SHCoeff_b) * 8;

    r/=12;
    g/=12;
    b/=12;
#endif
    // // =========================================
    vout.AO = float4(r,g,b, vin.Transfer[0][0]);
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

struct PS_OUTPUT
{
    float4 Color: SV_Target0;
    float4 Normal: SV_Target1;
};

PS_OUTPUT PS(VertexOut pin) : SV_Target
{
    MaterialData matData = gMaterialData[materialIndex];
    PS_OUTPUT output;
    output.Normal = float4(normalize(pin.ViewNormal)*0.5 + float3(0.5,0.5,0.5), 1);

    if(matData.gMatType > 0)
    {
        // float3 color = pow(pin.Color.rgb, float3(1.0/2.2,1.0/2.2,1.0/2.2)); 
        // output.Color = float4(color, 1);
        output.Color = pin.Color;
        return output;
    }

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

    color = pin.AO.rgb;//+ pin.AO.w * 10 * ambient;
    // color = pow(color, float3(1.0/2.2,1.0/2.2,1.0/2.2)); 

    // float4 test = pin.WorldPos;
    // test.xyz /= test.w; // 透视划分
    output.Color = float4(pin.AO.xyz, 1);
    return output;
    // return float4(radiance, 0.7);
}