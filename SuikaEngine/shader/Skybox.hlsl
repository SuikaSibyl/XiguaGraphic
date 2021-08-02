#include "ShaderLibrary/Common.hlsl"
#include "Utility.hlsl"
#include "Random.hlsl"

struct VertexOut
{
	float4 PosH : SV_POSITION;
	float3 PosL : POSITION;
};

VertexOut VS(VertexIn vin)
{
	VertexOut vout;
    
	// Use local vertex position as cubemap lookup vector.
	vout.PosL = vin.PosL;
    
	// Transform to world space.
	float4 posW = mul(float4(vin.PosL, 1.0f), gWorld);
	// Always center sky about camera.
	posW.xyz += gEyePosW;
	// Set z = w so that z/w = 1 (i.e., skydome always on far plane).
	vout.PosH = mul(posW, gViewProj).xyww;
    
    return vout;
}

float3 GetIBLDiffuse(float3 position)
{
    float3 normal = normalize(position);
    float3 irradiance = float3(0.0,0.0,0.0);
    float3 up    = float3(0.0, 1.0, 0.0);
    float3 right = cross(up, normal);
    up         = cross(normal, right);
    float sampleDelta = 0.025;
    float nrSamples = 0.0; 

    for(float phi = 0.0; phi < 2.0 * PI; phi += sampleDelta)
    {
        for(float theta = 0.0; theta < 0.5 * PI; theta += sampleDelta)
        {
            // spherical to cartesian (in tangent space)
            float3 tangentSample = float3(sin(theta) * cos(phi),  sin(theta) * sin(phi), cos(theta));
            // tangent space to world
            float3 sampleVec = tangentSample.x * right + tangentSample.y * up + tangentSample.z * normal; 

            irradiance += gCubeMap.Sample(gSamLinearWarp, sampleVec).rgb * cos(theta) * sin(theta);
            nrSamples++;
        }
    }
    irradiance = PI * irradiance * (1.0 / float(nrSamples));
    return irradiance;
}

float3 GetIBLReflection(float3 position, float roughness)
{
    float3 N = normalize(position);    
    float3 R = N;
    float3 V = R;

    float totalWeight = 1.0;   
    float3 prefilteredColor = float3(0.0,0.0,0.0);

	[unroll(2048)]
    for(uint i = 0; i < 2048; ++i)
    {
        float2 Xi = Hammersley(i, 2048);
        float3 H  = ImportanceSampleGGX(Xi, N, roughness);
        float3 L  = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = max(dot(N, L), 0.0);
        if(NdotL > 0.0)
        {
			// float2 loc_uv = SampleSphericalMap(normalize(L));
    		// float4 loc_color = gDiffuseMap[5].Sample(gSamPointWrap, loc_uv);
    		float4 loc_color = gCubeMap.Sample(gSamPointWrap, normalize(L));
            prefilteredColor += loc_color.rgb * NdotL;
            totalWeight      += NdotL;
        }
    }
    prefilteredColor = prefilteredColor / totalWeight;
    return prefilteredColor;
}

float4 PS(VertexOut pin) : SV_Target
{
    float2 uv = SampleSphericalMap(normalize(pin.PosL)); // make sure to normalize localPos
    float4 color = gDiffuseMap[7].Sample(gSamPointWrap, uv);
    float3 col = pow(color.xyz, float3(1.0/2.2,1.0/2.2,1.0/2.2)); 
    // float3 prefilteredColor = GetIBLReflection(normalize(pin.PosL),1);
    return float4(col,1);
	// float3 envColor = color.rgb;
    // envColor = envColor / (envColor + float3(1.0,1.0,1.0));
    // envColor = pow(envColor, float3(1.0/2.2,1.0/2.2,1.0/2.2)); 
    // float3 color = gCubeMap.Sample(gSamPointWrap, normalize(pin.PosL));
    // return color;
}