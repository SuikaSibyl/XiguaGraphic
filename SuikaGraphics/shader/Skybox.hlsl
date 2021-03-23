#include "Common.hlsl"

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

float4 PS(VertexOut pin) : SV_Target
{
    float2 uv = SampleSphericalMap(normalize(pin.PosL)); // make sure to normalize localPos
    float4 color = gDiffuseMap[5].Sample(gSamPointWrap, uv);

    // float4 albedo = gDiffuseMap[5].Sample(gSamPointWrap, pin.PosL.xy);
    
    return color;
}