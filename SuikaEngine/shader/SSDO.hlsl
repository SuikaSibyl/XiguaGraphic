#include "ShaderLibrary/Common.hlsl"
#include "Random.hlsl"
#include "PBR.hlsl"

#define NEAR 1
#define FAR 1000
#define W 1280
#define H 567
#define TANFOVD2 0.414213562373

#define kernelSize 16
#define radius 0.5

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

// Fetch Frag Pos in Vies Space
float3 GetFragPos(float depth, float2 uv)
{
    float zc = 1.001/(1.001-depth);
    uv = -2 * (uv-float2(0.5,0.5));
    uv.y = -uv.y;
    return float3(-uv.x*TANFOVD2*zc *2.25749559, -uv.y*TANFOVD2*zc ,zc);
}   

// Fetch Frag Depth (in NDC space, range in [0,1])
// uv range in [0,1]
float GetFragDepth(float2 uv)
{
    if(uv.y<0 || uv.y > 1)
    {
        return 100;
    }
    float depth = gDiffuseMap[15].Sample(gSamAnisotropicWarp, uv).rrr;
    float zc = 1.001/(1.001-depth);
    return zc;
}

// Fetch Frag Color
// uv range in [0,1]
float3 GetFragColor(float2 uv)
{
    float3 color = gDiffuseMap[11].Sample(gSamAnisotropicClamp, uv).rgb;
    return color;
}

// Fetch normal range in [-1,1]
// uv range in [0,1]
float3 GetFragNormal(float2 uv)
{
    float3 normal = gDiffuseMap[12].Sample(gSamAnisotropicClamp, uv).rgb;
    normal = (normal-0.5)*2;
    return normalize(normal);
}

float4 PS(VertexOut pin) : SV_Target
{
    float3 samples[] =
    {
        float3(0.497709, -0.447092, 0.499634),
        float3(0.145427, 0.164949, 0.0223372),
        float3(-0.402936, -0.19206, 0.316554),
        float3(0.135109, -0.89806, 0.401306),
        float3(0.540875, 0.577609, 0.557006),
        float3(0.874615, 0.41973, 0.146465),
        float3(-0.0188978, -0.504141, 0.618431),
        float3(-0.00298402, -0.00169127, 0.00333421),
        float3(0.438746, -0.408985, 0.222553),
        float3(0.323672, 0.266571, 0.27902),
        float3(-0.261392, 0.167732, 0.184589),
        float3(0.440034, -0.292085, 0.430474),
        float3(0.435821, -0.171226, 0.573847),
        float3(-0.117331, -0.0274799, 0.40452),
        float3(-0.174974, -0.173549, 0.174403),
        float3(-0.22543, 0.143145, 0.169986),
        float3(-0.112191, 0.0920681, 0.0342291),
        float3(0.448674, 0.685331, 0.0673666),
        float3(-0.257349, -0.527384, 0.488827),
        float3(-0.464402, -0.00938766, 0.473935),
        float3(-0.0553817, -0.174926, 0.102575),
        float3(0.0163094, -0.0247947, 0.0211469),
        float3(-0.0357804, -0.319047, 0.326624),
        float3(0.435365, -0.0369896, 0.662937),
        float3(0.339125, 0.56041, 0.472273),
        float3(0.00165474, 0.00189482, 0.00127085),
        float3(-0.421643, 0.263322, 0.409346),
        float3(-0.0171094, -0.459828, 0.622265),
        float3(-0.273823, 0.126528, 0.823235),
        float3(-0.00968538, 0.0108071, 0.0102621),
        float3(-0.364436, 0.478037, 0.558969),
        float3(0.15067, 0.333067, 0.191465),
        float3(0.414059, -0.0692679, 0.401582),
        float3(-0.484817, -0.458746, 0.367069),
        float3(-0.530125, -0.589921, 0.16319),
        float3(-0.118435, 0.235465, 0.202611),
        float3(-0.00666287, -0.0052001, 0.010577),
        float3(-0.241253, -0.454733, 0.747212),
        float3(-0.541038, 0.757421, 0.213657),
        float3(-0.0633459, 0.66141, 0.73048),
        float3(0.458887, -0.599781, 0.24389),
        float3(0.116971, 0.222313, 0.688396),
        float3(-0.268377, 0.244657, 0.574693),
        float3(0.304252, -0.129121, 0.453988),
        float3(0.100759, -0.433708, 0.282605),
        float3(-0.343713, -0.0738141, 0.0292256),
        float3(0.251075, 0.0834831, 0.238692),
        float3(-0.0756226, 0.0950082, 0.0954248),
        float3(-0.0389006, -0.133558, 0.361451),
        float3(-0.226506, 0.315615, 0.00827583),
        float3(0.244327, 0.354923, 0.0673253),
        float3(0.0447351, 0.568618, 0.243966),
        float3(0.119581, -0.446107, 0.0971173),
        float3(0.316438, -0.328146, 0.270037),
        float3(0.51475, 0.448266, 0.714832),
        float3(-0.727464, 0.385414, 0.393764),
        float3(0.537968, 0.00715645, 0.149009),
        float3(0.450305, 0.00440889, 0.105299),
        float3(0.39208, 0.0368202, 0.212718),
        float3(-0.0958963, 0.592978, 0.0653918),
        float3(0.973455, -0.00306814, 0.112386),
        float3(0.496669, -0.841329, 0.00418623),
        float3(0.441751, -0.163923, 0.489625),
        float3(-0.455431, -0.698782, 0.191856),
    };

    float3 noises[] = 
    {
        float3(-0.729046, 0.629447, 0),
        float3(0.670017, 0.811584, 0),
        float3(0.937736, -0.746026, 0),
        float3(-0.557932, 0.826752, 0),
        float3(-0.383666, 0.264719, 0),
        float3(0.0944412, -0.804919, 0),
        float3(-0.623236, -0.443004, 0),
        float3(0.985763, 0.093763, 0),
        float3(0.992923, 0.915014, 0),
        float3(0.93539, 0.929777, 0),
        float3(0.451678, -0.684774, 0),
        float3(0.962219, 0.941186, 0),
        float3(-0.780276, 0.914334, 0),
        float3(0.596212, -0.0292487, 0),
        float3(-0.405941, 0.600561, 0),
        float3(-0.990433, -0.716227, 0),
    };

    float2 uv = float2(pin.uv.x,1-pin.uv.y);
    float4 albedo = gDiffuseMap[11].Sample(gSamAnisotropicWarp, uv);
    float3 normal = gDiffuseMap[12].Sample(gSamAnisotropicWarp, uv).rgb;
    normal = (normal-0.5)*2;
    normal = normalize(normal);
    float depth = gDiffuseMap[15].Sample(gSamAnisotropicWarp, uv).rrr;
    
    float3 fragPos = GetFragPos(depth, uv);

    int unoise = uv.x*1280;
    int vnoise = uv.y*567;
    float3 randomVec = noises[(unoise%4)*4 + (vnoise%4)].xzy;
    
    float3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    float3 bitangent = cross(normal, tangent);
    float3x3 TBN = float3x3(tangent, bitangent, normal);
    
    float occlusion = 0.0;
    float3 indirect = float3(0,0,0);
    float samp_dep = 0.0;
    if(albedo.w==0 || depth>0.99)
    {

    }
    else
    {
        for(int i = 0; i < kernelSize; ++i)
        {
            float3 samp = mul(samples[i], TBN); // 切线->观察空间
            float3 sample_pos = fragPos + samp * radius; 

            float4 offset = float4(sample_pos, 1.0);
            offset = mul(offset, gProjection);
            offset.xyz /= offset.w; // 透视划分
            offset.xy = offset.xy * 0.5 + 0.5; // 变换到0.0 - 1.0的值域
            offset.xy = float2(offset.x,1-offset.y);

            float sampleDepth = GetFragDepth(offset.xy);
            float distance = (abs(sample_pos.z - sampleDepth)*length(sample_pos)/abs(sample_pos.z));
            float range = radius / distance;
            float rangeCheck = smoothstep(0.0, 1.0, range);
            if(rangeCheck<0.9) rangeCheck=0.0;
            occlusion += (sampleDepth <= sample_pos.z ? 1.0 : 0.0) * rangeCheck;
            indirect += (sampleDepth <= sample_pos.z ? GetFragColor(offset.xy) * max(0,dot(normalize(sample_pos-fragPos), normal)) * max(0,dot(normalize(fragPos-sample_pos), GetFragNormal(offset.xy)))* rangeCheck  : 0.0);    
        }
    }
    
    occlusion = 1.0 - (occlusion/ kernelSize);
    indirect/= kernelSize;
    return albedo + float4(indirect*2.14,1);
}